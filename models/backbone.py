"""
DINOv2 Backbone Model for Few-Shot Object Detection

This module implements the DINOv2 backbone used for feature extraction
in the two-stage fine-tuning approach for few-shot object detection.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import torchvision.transforms as transforms
import numpy as np


class DINOv2Backbone(nn.Module):
    """
    DINOv2 backbone for feature extraction.
    
    This model uses the pretrained DINOv2 vision transformer as a feature extractor
    for object detection tasks. It extracts multi-scale features suitable for RPN
    and ROI pooling operations.
    """
    
    def __init__(self, model_name: str = "dinov2_vitb14", pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        
        # Load pretrained DINOv2 model
        if pretrained:
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
        else:
            # For cases where we want to load from local checkpoint
            raise NotImplementedError("Local checkpoint loading not implemented yet")
        
        # Freeze backbone parameters (will be unfrozen in Stage I training)
        for param in self.dinov2.parameters():
            param.requires_grad = False
            
        # Get feature dimensions based on model type
        self.feature_dim = self._get_feature_dim()
        
        # Create feature projection layers for different scales
        # These help adapt DINOv2 features for object detection
        self.feature_proj = nn.Conv2d(self.feature_dim, 256, kernel_size=1)
        self.feature_norm = nn.GroupNorm(32, 256)
        
    def _get_feature_dim(self) -> int:
        """Get the feature dimension based on model type."""
        if "vits14" in self.model_name:
            return 384
        elif "vitb14" in self.model_name:
            return 768
        elif "vitl14" in self.model_name:
            return 1024
        elif "vitg14" in self.model_name:
            return 1536
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
    
    def freeze_backbone(self):
        """Freeze backbone parameters (used in Stage II)."""
        for param in self.dinov2.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters (used in Stage I)."""
        for param in self.dinov2.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DINOv2 backbone.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            
        Returns:
            Dictionary containing feature maps at different scales
        """
        batch_size, _, height, width = x.shape
        
        # DINOv2 expects specific input preprocessing
        # Ensure input is properly normalized (DINOv2 uses ImageNet normalization)
        
        # Extract features using DINOv2
        with torch.set_grad_enabled(self.training):
            try:
                # Try the standard forward_features method
                features = self.dinov2.forward_features(x)
            except Exception as e:
                print(f"Warning: forward_features failed ({e}), trying alternative method")
                # Fallback to regular forward pass
                features = self.dinov2(x)
            
        # Handle different DINOv2 output formats
        if isinstance(features, dict):
            # Some DINOv2 versions return a dictionary
            if 'x_norm_patchtokens' in features:
                features = features['x_norm_patchtokens']
            elif 'x_prenorm' in features:
                features = features['x_prenorm']
            elif 'last_hidden_state' in features:
                features = features['last_hidden_state']
            else:
                # Take the first tensor value if it's a dict
                features = list(features.values())[0]
        elif hasattr(features, 'last_hidden_state'):
            # Handle transformers-style output
            features = features.last_hidden_state
        
        # Ensure we have a tensor
        if not isinstance(features, torch.Tensor):
            raise ValueError(f"Unexpected DINOv2 output type: {type(features)}")
        
        # Handle different tensor shapes
        if len(features.shape) == 2:
            # If (B, feature_dim), assume it's already pooled
            # Reshape to (B, feature_dim, 1, 1) for consistency
            features = features.unsqueeze(-1).unsqueeze(-1)
            feat_h, feat_w = 1, 1
        elif len(features.shape) == 3:
            # DINOv2 returns features as (B, N_patches + 1, feature_dim) or (B, N_patches, feature_dim)
            # Only remove first token if it looks like a CLS token (i.e., we have more patches than expected)
            
            # Calculate expected number of patches
            patch_size = 14
            expected_patches = (height // patch_size) * (width // patch_size)
            current_patches = features.shape[1]
            
            # If we have exactly expected_patches + 1, remove the first token (likely CLS)
            if current_patches == expected_patches + 1:
                features = features[:, 1:]  # Remove CLS token
                print(f"Debug: Removed CLS token, {current_patches} -> {features.shape[1]} patches")
            # If we already have the expected number (or close), don't remove anything
            elif abs(current_patches - expected_patches) <= 2:
                print(f"Debug: Keeping {current_patches} patches (expected {expected_patches})")
            # If significantly different, still remove first if > expected 
            elif current_patches > expected_patches:
                features = features[:, 1:]  # Remove what might be CLS token
                print(f"Debug: Removed potential CLS token, {current_patches} -> {features.shape[1]} patches")
            
            # Calculate patch grid dimensions
            patch_size = 14  # DINOv2 uses 14x14 patches
            feat_h = height // patch_size
            feat_w = width // patch_size
            
            # Verify the number of patches matches
            expected_patches = feat_h * feat_w
            actual_patches = features.shape[1]
            
            if actual_patches != expected_patches:
                # Try to find the best grid dimensions
                # First try square grid
                sqrt_patches = int(np.sqrt(actual_patches))
                if sqrt_patches * sqrt_patches == actual_patches:
                    feat_h = feat_w = sqrt_patches
                else:
                    # Try to find rectangular grids that work with actual_patches
                    found_grid = False
                    
                    # Check dimensions around the expected size, preferring balanced grids
                    best_score = float('inf')
                    best_h, best_w = feat_h, feat_w
                    
                    for h in range(max(1, feat_h - 5), feat_h + 6):
                        if actual_patches % h == 0:
                            w = actual_patches // h
                            # Score based on distance from expected dims and aspect ratio balance
                            dist_score = abs(h - feat_h) + abs(w - feat_w)
                            aspect_score = abs(h - w) / max(h, w)  # Prefer more square grids
                            total_score = dist_score + aspect_score * 10
                            
                            if total_score < best_score:
                                best_score = total_score
                                best_h, best_w = h, w
                                found_grid = True
                    
                    # Use the best grid found, or try all factor pairs if none found
                    if found_grid:
                        feat_h, feat_w = best_h, best_w
                    else:
                        # Try all factors of actual_patches
                        factors = []
                        for i in range(1, int(np.sqrt(actual_patches)) + 1):
                            if actual_patches % i == 0:
                                factors.append((i, actual_patches // i))
                        
                        if factors:
                            # Choose the factor pair closest to square
                            feat_h, feat_w = min(factors, key=lambda x: abs(x[0] - x[1]))
                            found_grid = True
                        else:
                            # Last resort: use closest square and handle padding/truncation later
                            feat_h = feat_w = sqrt_patches
                    
                    if found_grid:
                        print(f"Info: Using {feat_h}x{feat_w} grid for {actual_patches} patches")
                    else:
                        print(f"Warning: Using {feat_h}x{feat_w} grid for {actual_patches} patches (will pad/truncate)")
            
            # Reshape to spatial format: (B, feature_dim, H, W)
            try:
                features = features.transpose(1, 2).reshape(batch_size, self.feature_dim, feat_h, feat_w)
            except RuntimeError as e:
                # If reshape fails, handle mismatched dimensions
                target_patches = feat_h * feat_w
                current_patches = features.shape[1]
                
                if current_patches > target_patches:
                    # Truncate extra patches
                    features = features[:, :target_patches]
                    print(f"Warning: Truncated {current_patches - target_patches} patches to fit {feat_h}x{feat_w} grid")
                elif current_patches < target_patches:
                    # Pad with zeros
                    pad_size = target_patches - current_patches
                    padding = torch.zeros(batch_size, pad_size, self.feature_dim, device=features.device, dtype=features.dtype)
                    features = torch.cat([features, padding], dim=1)
                    print(f"Warning: Padded {pad_size} patches to fit {feat_h}x{feat_w} grid")
                
                features = features.transpose(1, 2).reshape(batch_size, self.feature_dim, feat_h, feat_w)
        elif len(features.shape) == 4:
            # Already in (B, C, H, W) format
            feat_h, feat_w = features.shape[2], features.shape[3]
        else:
            raise ValueError(f"Unexpected feature tensor shape: {features.shape}")
        
        # Apply projection and normalization
        features = self.feature_proj(features)
        features = self.feature_norm(features)
        features = torch.relu(features)
        
        return {
            "features": features,
            "spatial_dims": (feat_h, feat_w),
            "stride": height // feat_h if feat_h > 0 else 14
        }


class FeatureExtractor(nn.Module):
    """
    Feature extractor that combines backbone features with ROI pooling.
    
    This module takes the backbone features and region proposals to extract
    fixed-size feature vectors for classification and regression.
    """
    
    def __init__(self, feature_dim: int = 256, output_size: int = 7):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_size = output_size
        
        # ROI Align layer for extracting fixed-size features
        from torchvision.ops import RoIAlign
        self.roi_align = RoIAlign(
            output_size=(output_size, output_size),
            spatial_scale=1.0,  # Will be adjusted based on feature map stride
            sampling_ratio=2
        )
        
        # Additional processing layers
        self.conv1 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, feature_dim)
        self.norm2 = nn.GroupNorm(32, feature_dim)
        
        # Global average pooling to get fixed-size features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, features: torch.Tensor, rois: torch.Tensor, spatial_scale: float) -> torch.Tensor:
        """
        Extract ROI features from backbone feature maps.
        
        Args:
            features: Feature maps from backbone (B, C, H, W)
            rois: Region proposals (N, 5) where first column is batch index
            spatial_scale: Scale factor from input image to feature map
            
        Returns:
            ROI features (N, feature_dim)
        """
        # Create RoIAlign with the correct spatial scale to avoid inplace operations
        from torchvision.ops import RoIAlign
        roi_align = RoIAlign(
            output_size=(self.output_size, self.output_size),
            spatial_scale=spatial_scale,
            sampling_ratio=2
        )
        
        # Extract ROI features
        roi_features = roi_align(features, rois)
        
        # Additional processing
        roi_features = torch.relu(self.norm1(self.conv1(roi_features)))
        roi_features = torch.relu(self.norm2(self.conv2(roi_features)))
        
        # Global pooling to get fixed-size features
        roi_features = self.global_pool(roi_features)
        roi_features = roi_features.flatten(1)
        
        return roi_features


def get_image_transforms(is_training: bool = True) -> transforms.Compose:
    """
    Get image preprocessing transforms for DINOv2.
    
    Args:
        is_training: Whether to include training augmentations
        
    Returns:
        Composed transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((518, 518)),  # DINOv2 typical input size
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


if __name__ == "__main__":
    # Test the backbone model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    backbone = DINOv2Backbone(model_name="dinov2_vitb14").to(device)
    
    # Test input
    x = torch.randn(2, 3, 518, 518).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = backbone(x)
        print(f"Feature shape: {output['features'].shape}")
        print(f"Spatial dimensions: {output['spatial_dims']}")
        print(f"Stride: {output['stride']}")