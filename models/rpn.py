"""
Region Proposal Network (RPN) and ROI components for few-shot object detection.

This module implements the RPN that generates object proposals and the ROI pooling
mechanism for extracting features from proposed regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


class AnchorGenerator:
    """
    Generate anchor boxes at multiple scales and aspect ratios.
    """
    
    def __init__(self, 
                 scales: List[float] = [0.5, 1.0, 2.0],
                 aspect_ratios: List[float] = [0.5, 1.0, 2.0],
                 anchor_size: int = 32):
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.anchor_size = anchor_size
        
    def generate_anchors(self, feature_height: int, feature_width: int, 
                        stride: int, device: torch.device) -> torch.Tensor:
        """
        Generate anchor boxes for a feature map.
        
        Args:
            feature_height: Height of the feature map
            feature_width: Width of the feature map
            stride: Stride of the feature map relative to input image
            device: Device to create anchors on
            
        Returns:
            Anchors tensor of shape (N, 4) in (x1, y1, x2, y2) format
        """
        # Create base anchors
        base_anchors = []
        for scale in self.scales:
            for ratio in self.aspect_ratios:
                size = self.anchor_size * scale
                h = size * np.sqrt(ratio)
                w = size / np.sqrt(ratio)
                base_anchors.append([-w/2, -h/2, w/2, h/2])
        
        base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=device)
        
        # Create grid of anchor centers
        shift_x = torch.arange(0, feature_width, dtype=torch.float32, device=device) * stride
        shift_y = torch.arange(0, feature_height, dtype=torch.float32, device=device) * stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        
        shifts = torch.stack([shift_x.flatten(), shift_y.flatten(), 
                             shift_x.flatten(), shift_y.flatten()], dim=1)
        
        # Generate all anchors
        anchors = base_anchors.view(1, -1, 4) + shifts.view(-1, 1, 4)
        anchors = anchors.view(-1, 4)
        
        return anchors


class RPNHead(nn.Module):
    """
    Region Proposal Network Head for generating object proposals.
    """
    
    def __init__(self, in_channels: int = 256, num_anchors: int = 9):
        super().__init__()
        self.num_anchors = num_anchors
        
        # Shared convolutional layer
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
        # Classification head (objectness)
        self.cls_head = nn.Conv2d(in_channels, num_anchors, 1)
        
        # Regression head (box deltas)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize layer weights."""
        for layer in [self.conv, self.cls_head, self.reg_head]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through RPN head.
        
        Args:
            features: Feature maps from backbone (B, C, H, W)
            
        Returns:
            Tuple of (objectness_scores, bbox_deltas)
            - objectness_scores: (B, num_anchors, H, W)
            - bbox_deltas: (B, num_anchors*4, H, W)
        """
        x = F.relu(self.conv(features))
        
        objectness = self.cls_head(x)
        bbox_deltas = self.reg_head(x)
        
        return objectness, bbox_deltas


class RPN(nn.Module):
    """
    Complete Region Proposal Network.
    """
    
    def __init__(self, 
                 in_channels: int = 256,
                 anchor_scales: List[float] = [0.5, 1.0, 2.0],
                 anchor_ratios: List[float] = [0.5, 1.0, 2.0],
                 nms_threshold: float = 0.7,
                 score_threshold: float = 0.5,
                 max_proposals: int = 1000):
        super().__init__()
        
        self.anchor_generator = AnchorGenerator(
            scales=anchor_scales,
            aspect_ratios=anchor_ratios
        )
        
        self.rpn_head = RPNHead(
            in_channels=in_channels,
            num_anchors=len(anchor_scales) * len(anchor_ratios)
        )
        
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_proposals = max_proposals
        
    def forward(self, features: torch.Tensor, image_size: Tuple[int, int], 
                stride: int) -> Dict[str, torch.Tensor]:
        """
        Forward pass through RPN.
        
        Args:
            features: Feature maps from backbone (B, C, H, W)
            image_size: Original image size (H, W)
            stride: Feature map stride
            
        Returns:
            Dictionary containing proposals and scores
        """
        batch_size, _, feat_h, feat_w = features.shape
        device = features.device
        
        # Generate anchors
        anchors = self.anchor_generator.generate_anchors(
            feat_h, feat_w, stride, device
        )
        
        # Get RPN predictions
        objectness, bbox_deltas = self.rpn_head(features)
        
        # Reshape predictions
        objectness = objectness.permute(0, 2, 3, 1).reshape(batch_size, -1)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        
        # Apply deltas to anchors to get proposals
        proposals_list = []
        scores_list = []
        
        for i in range(batch_size):
            # Apply sigmoid to objectness scores
            scores = torch.sigmoid(objectness[i])
            
            # Decode bbox deltas
            proposals = self._decode_boxes(anchors, bbox_deltas[i])
            
            # Clip to image boundaries
            proposals = self._clip_boxes(proposals, image_size)
            
            # Filter by score threshold
            keep = scores >= self.score_threshold
            proposals = proposals[keep]
            scores = scores[keep]
            
            # Apply NMS
            if len(proposals) > 0:
                keep_nms = self._nms(proposals, scores, self.nms_threshold)
                proposals = proposals[keep_nms[:self.max_proposals]]
                scores = scores[keep_nms[:self.max_proposals]]
            
            proposals_list.append(proposals)
            scores_list.append(scores)
        
        return {
            "proposals": proposals_list,
            "scores": scores_list,
            "objectness_raw": objectness,
            "bbox_deltas": bbox_deltas
        }
    
    def _decode_boxes(self, anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """
        Decode bounding box deltas relative to anchors.
        
        Args:
            anchors: Anchor boxes (N, 4)
            deltas: Predicted deltas (N, 4)
            
        Returns:
            Decoded boxes (N, 4)
        """
        # Convert anchors to center format
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights
        
        # Apply deltas
        dx, dy, dw, dh = deltas.unbind(dim=1)
        
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        
        # Convert back to corner format
        pred_boxes = torch.stack([
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h
        ], dim=1)
        
        return pred_boxes
    
    def _clip_boxes(self, boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """Clip boxes to image boundaries."""
        height, width = image_size
        # Create new tensor to avoid inplace operations
        clipped_boxes = torch.stack([
            torch.clamp(boxes[:, 0], 0, width),
            torch.clamp(boxes[:, 1], 0, height),
            torch.clamp(boxes[:, 2], 0, width),
            torch.clamp(boxes[:, 3], 0, height)
        ], dim=1)
        return clipped_boxes
    
    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, 
             threshold: float) -> torch.Tensor:
        """Apply Non-Maximum Suppression."""
        from torchvision.ops import nms
        return nms(boxes, scores, threshold)


class ROIPooling(nn.Module):
    """
    ROI Pooling layer that extracts fixed-size features from variable-size regions.
    """
    
    def __init__(self, output_size: Tuple[int, int] = (7, 7), spatial_scale: float = 1.0):
        super().__init__()
        from torchvision.ops import RoIAlign
        
        self.roi_align = RoIAlign(
            output_size=output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=2
        )
    
    def forward(self, features: torch.Tensor, rois: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract ROI features.
        
        Args:
            features: Feature maps (B, C, H, W)
            rois: List of ROI boxes for each batch item
            
        Returns:
            Pooled features (total_rois, C, output_h, output_w)
        """
        # Prepare ROIs in the format expected by RoIAlign
        # Format: (batch_idx, x1, y1, x2, y2)
        roi_list = []
        for batch_idx, batch_rois in enumerate(rois):
            if len(batch_rois) > 0:
                batch_indices = torch.full((len(batch_rois), 1), batch_idx, 
                                         dtype=torch.float32, device=batch_rois.device)
                roi_list.append(torch.cat([batch_indices, batch_rois], dim=1))
        
        if len(roi_list) == 0:
            # No ROIs, return empty tensor
            return torch.empty(0, features.shape[1], *self.roi_align.output_size, 
                             device=features.device)
        
        all_rois = torch.cat(roi_list, dim=0)
        
        # Apply ROI pooling
        pooled_features = self.roi_align(features, all_rois)
        
        return pooled_features


if __name__ == "__main__":
    # Test RPN and ROI components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test RPN
    rpn = RPN().to(device)
    features = torch.randn(2, 256, 32, 32).to(device)
    
    with torch.no_grad():
        rpn_output = rpn(features, image_size=(512, 512), stride=16)
        print(f"Number of proposals: {[len(p) for p in rpn_output['proposals']]}")
    
    # Test ROI Pooling
    roi_pool = ROIPooling().to(device)
    
    # Create some dummy ROIs
    rois = [
        torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]], 
                    dtype=torch.float32, device=device),
        torch.tensor([[20, 20, 60, 60]], dtype=torch.float32, device=device)
    ]
    
    with torch.no_grad():
        pooled = roi_pool(features, rois)
        print(f"Pooled features shape: {pooled.shape}")