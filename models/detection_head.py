"""
Box Classification and Regression heads for few-shot object detection.

This module implements the final classification and bounding box regression
heads that operate on ROI features to predict object classes and refine
bounding box coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class BoxClassifier(nn.Module):
    """
    Box classification head that predicts object classes from ROI features.
    
    This head can be frozen during Stage II fine-tuning while keeping
    the feature extractor frozen and only training class-specific layers.
    """
    
    def __init__(self, 
                 feature_dim: int = 256,
                 num_classes: int = 80,
                 hidden_dim: int = 1024,
                 dropout: float = 0.5):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Shared feature processing layers
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.cls_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights."""
        for layer in [self.fc1, self.fc2, self.cls_head]:
            if isinstance(layer, nn.Linear):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)
    
    def forward(self, roi_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            roi_features: ROI features (N, feature_dim)
            
        Returns:
            Class logits (N, num_classes + 1)
        """
        x = F.relu(self.fc1(roi_features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        cls_logits = self.cls_head(x)
        
        return cls_logits


class BoxRegressor(nn.Module):
    """
    Box regression head that refines bounding box coordinates.
    
    This head predicts deltas to adjust the initial proposals for
    better localization accuracy.
    """
    
    def __init__(self, 
                 feature_dim: int = 256,
                 num_classes: int = 80,
                 hidden_dim: int = 1024,
                 dropout: float = 0.5):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Shared feature processing layers
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Regression head (4 coordinates per class)
        self.reg_head = nn.Linear(hidden_dim, (num_classes + 1) * 4)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights."""
        for layer in [self.fc1, self.fc2]:
            if isinstance(layer, nn.Linear):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)
        
        # Initialize regression head with smaller weights
        torch.nn.init.normal_(self.reg_head.weight, std=0.001)
        torch.nn.init.constant_(self.reg_head.bias, 0)
    
    def forward(self, roi_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through regression head.
        
        Args:
            roi_features: ROI features (N, feature_dim)
            
        Returns:
            Box deltas (N, (num_classes + 1) * 4)
        """
        x = F.relu(self.fc1(roi_features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        bbox_deltas = self.reg_head(x)
        
        return bbox_deltas


class FewShotClassifier(nn.Module):
    """
    Few-shot classifier for novel classes.
    
    This module implements a prototypical network approach where
    novel classes are represented by prototypes computed from
    support examples.
    """
    
    def __init__(self, feature_dim: int = 256, distance_metric: str = "cosine"):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.distance_metric = distance_metric
        
        # Prototype storage
        self.register_buffer('prototypes', torch.empty(0, feature_dim))
        self.register_buffer('prototype_labels', torch.empty(0, dtype=torch.long))
        
        # Temperature parameter for scaling similarities
        self.temperature = nn.Parameter(torch.tensor(10.0))
        
    def update_prototypes(self, support_features: torch.Tensor, 
                         support_labels: torch.Tensor):
        """
        Update prototypes based on support examples.
        
        Args:
            support_features: Features from support examples (N, feature_dim)
            support_labels: Labels for support examples (N,)
        """
        unique_labels = torch.unique(support_labels)
        prototypes = []
        prototype_labels = []
        
        for label in unique_labels:
            mask = support_labels == label
            prototype = support_features[mask].mean(dim=0)
            prototypes.append(prototype)
            prototype_labels.append(label)
        
        self.prototypes = torch.stack(prototypes)
        self.prototype_labels = torch.tensor(prototype_labels, device=self.prototypes.device)
    
    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        Classify query features using prototypes.
        
        Args:
            query_features: Features from query examples (N, feature_dim)
            
        Returns:
            Similarity scores to each prototype (N, num_prototypes)
        """
        if len(self.prototypes) == 0:
            return torch.empty(len(query_features), 0, device=query_features.device)
        
        # Normalize features
        query_features = F.normalize(query_features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        
        if self.distance_metric == "cosine":
            # Cosine similarity
            similarities = torch.mm(query_features, prototypes.t())
        elif self.distance_metric == "euclidean":
            # Negative squared euclidean distance
            similarities = -torch.cdist(query_features, prototypes, p=2) ** 2
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Scale by temperature
        similarities = similarities * self.temperature
        
        return similarities


class DetectionHead(nn.Module):
    """
    Complete detection head combining classification and regression.
    
    This module can operate in two modes:
    1. Base training: Uses standard classifier for base classes
    2. Few-shot fine-tuning: Uses few-shot classifier for novel classes
    """
    
    def __init__(self, 
                 feature_dim: int = 256,
                 num_base_classes: int = 80,
                 hidden_dim: int = 1024,
                 dropout: float = 0.5):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_base_classes = num_base_classes
        
        # Base classifier for Stage I training
        self.base_classifier = BoxClassifier(
            feature_dim=feature_dim,
            num_classes=num_base_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Box regressor
        self.box_regressor = BoxRegressor(
            feature_dim=feature_dim,
            num_classes=num_base_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Few-shot classifier for Stage II fine-tuning
        self.few_shot_classifier = FewShotClassifier(feature_dim=feature_dim)
        
        # Mode flag
        self.few_shot_mode = False
    
    def set_few_shot_mode(self, mode: bool):
        """Switch between base training and few-shot modes."""
        self.few_shot_mode = mode
    
    def freeze_feature_layers(self):
        """Freeze feature processing layers for Stage II."""
        for param in self.base_classifier.parameters():
            if 'cls_head' not in [name for name, _ in self.base_classifier.named_parameters()
                                if param is getattr(self.base_classifier, name.split('.')[0], None)]:
                param.requires_grad = False
        
        for param in self.box_regressor.parameters():
            if 'reg_head' not in [name for name, _ in self.box_regressor.named_parameters()
                                if param is getattr(self.box_regressor, name.split('.')[0], None)]:
                param.requires_grad = False
    
    def forward(self, roi_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through detection head.
        
        Args:
            roi_features: ROI features (N, feature_dim)
            
        Returns:
            Dictionary containing classification and regression outputs
        """
        # Always compute box regression
        bbox_deltas = self.box_regressor(roi_features)
        
        if self.few_shot_mode:
            # Use few-shot classifier
            cls_scores = self.few_shot_classifier(roi_features)
        else:
            # Use base classifier
            cls_scores = self.base_classifier(roi_features)
        
        return {
            "cls_scores": cls_scores,
            "bbox_deltas": bbox_deltas
        }


def compute_detection_loss(predictions: Dict[str, torch.Tensor],
                          targets: Dict[str, torch.Tensor],
                          loss_weights: Dict[str, float] = None) -> Dict[str, torch.Tensor]:
    """
    Compute detection losses for classification and regression.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        loss_weights: Weights for different loss components
        
    Returns:
        Dictionary of losses
    """
    if loss_weights is None:
        loss_weights = {"cls": 1.0, "reg": 1.0}
    
    # Classification loss
    cls_loss = F.cross_entropy(predictions["cls_scores"], targets["labels"])
    
    # Regression loss (only for positive samples)
    pos_mask = targets["labels"] > 0
    if pos_mask.sum() > 0:
        pos_bbox_deltas = predictions["bbox_deltas"][pos_mask]
        pos_targets = targets["bbox_targets"][pos_mask]
        reg_loss = F.smooth_l1_loss(pos_bbox_deltas, pos_targets)
    else:
        reg_loss = torch.tensor(0.0, device=predictions["cls_scores"].device)
    
    # Total loss
    total_loss = (loss_weights["cls"] * cls_loss + 
                 loss_weights["reg"] * reg_loss)
    
    return {
        "total_loss": total_loss,
        "cls_loss": cls_loss,
        "reg_loss": reg_loss
    }


if __name__ == "__main__":
    # Test detection heads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test base classifier
    classifier = BoxClassifier(feature_dim=256, num_classes=20).to(device)
    roi_features = torch.randn(100, 256).to(device)
    
    with torch.no_grad():
        cls_scores = classifier(roi_features)
        print(f"Classification scores shape: {cls_scores.shape}")
    
    # Test few-shot classifier
    few_shot_cls = FewShotClassifier(feature_dim=256).to(device)
    
    # Create support examples
    support_features = torch.randn(10, 256).to(device)
    support_labels = torch.tensor([0, 0, 1, 1, 1, 2, 2, 0, 1, 2]).to(device)
    
    few_shot_cls.update_prototypes(support_features, support_labels)
    
    # Test query
    query_features = torch.randn(5, 256).to(device)
    with torch.no_grad():
        similarities = few_shot_cls(query_features)
        print(f"Few-shot similarities shape: {similarities.shape}")
    
    # Test complete detection head
    detection_head = DetectionHead(feature_dim=256, num_base_classes=20).to(device)
    
    with torch.no_grad():
        output = detection_head(roi_features)
        print(f"Detection head output keys: {output.keys()}")
        print(f"Cls scores shape: {output['cls_scores'].shape}")
        print(f"Bbox deltas shape: {output['bbox_deltas'].shape}")