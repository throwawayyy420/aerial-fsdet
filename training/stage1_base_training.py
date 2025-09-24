"""
Stage I: Base Training Script

This script implements the first stage of the two-stage fine-tuning approach
where the complete model (backbone + RPN + detection heads) is trained on
abundant base class data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import os
import json
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import logging

# Import model components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import DINOv2Backbone, FeatureExtractor, get_image_transforms
from models.rpn import RPN, ROIPooling
from models.detection_head import DetectionHead, compute_detection_loss


class BaseDetectionDataset(Dataset):
    """
    Dataset for base class training with abundant data.
    
    This dataset should be adapted to your specific data format.
    For demonstration, we'll use a COCO-like format.
    """
    
    def __init__(self, 
                 data_dir: str,
                 annotations_file: str,
                 transforms: Optional[transforms.Compose] = None,
                 max_objects: int = 50):
        self.data_dir = data_dir
        self.transforms = transforms or get_image_transforms(is_training=True)
        self.max_objects = max_objects
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.images = self.annotations['images']
        self.annotation_data = self.annotations['annotations']
        
        # Build image_id to annotations mapping
        self.img_to_anns = {}
        for ann in self.annotation_data:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        
        # Load image
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations
        anns = self.img_to_anns.get(img_id, [])
        
        # Extract boxes and labels
        boxes = []
        labels = []
        
        for ann in anns[:self.max_objects]:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        # Pad if necessary
        while len(boxes) < self.max_objects:
            boxes.append([0, 0, 1, 1])  # Dummy box
            labels.append(0)  # Background class
        
        # Apply transforms
        if self.transforms:
            image = self.transforms(image)
        
        return {
            'image': image,
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'image_id': img_id,
            'original_size': (img_info['height'], img_info['width'])
        }


class FewShotDetector(nn.Module):
    """
    Complete few-shot detection model combining all components.
    """
    
    def __init__(self, 
                 num_classes: int = 80,
                 backbone_model: str = "dinov2_vitb14",
                 feature_dim: int = 256):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = DINOv2Backbone(model_name=backbone_model, pretrained=True)
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            feature_dim=feature_dim,
            output_size=7
        )
        
        # RPN
        self.rpn = RPN(
            in_channels=feature_dim,
            anchor_scales=[0.5, 1.0, 2.0],
            anchor_ratios=[0.5, 1.0, 2.0]
        )
        
        # ROI pooling
        self.roi_pooling = ROIPooling(
            output_size=(7, 7),
            spatial_scale=1.0
        )
        
        # Detection head
        self.detection_head = DetectionHead(
            feature_dim=feature_dim,
            num_base_classes=num_classes,
            hidden_dim=1024
        )
    
    def forward(self, images: torch.Tensor, targets: Optional[Dict] = None) -> Dict:
        """
        Forward pass through the complete model.
        
        Args:
            images: Input images (B, 3, H, W)
            targets: Ground truth targets (for training)
            
        Returns:
            Dictionary containing predictions and losses (if training)
        """
        batch_size = images.shape[0]
        
        # Extract backbone features
        backbone_output = self.backbone(images)
        features = backbone_output["features"]
        spatial_dims = backbone_output["spatial_dims"]
        stride = backbone_output["stride"]
        
        # Generate proposals with RPN
        image_size = (images.shape[2], images.shape[3])
        rpn_output = self.rpn(features, image_size, stride)
        
        # Extract ROI features
        spatial_scale = 1.0 / stride
        roi_features_list = []
        
        for i in range(batch_size):
            proposals = rpn_output["proposals"][i]
            if len(proposals) > 0:
                # Format ROIs for ROI pooling
                batch_indices = torch.full((len(proposals), 1), i, 
                                         dtype=torch.float32, device=proposals.device)
                rois = torch.cat([batch_indices, proposals], dim=1)
                
                # Extract features using the proper method with spatial scale
                roi_feats = self.feature_extractor(features, rois, spatial_scale)
                roi_features_list.append(roi_feats)
        
        if len(roi_features_list) > 0:
            all_roi_features = torch.cat(roi_features_list, dim=0)
        else:
            # No proposals, create dummy features
            all_roi_features = torch.empty(0, self.feature_extractor.feature_dim, 
                                         device=features.device)
        
        # Get detection predictions
        detection_output = self.detection_head(all_roi_features)
        
        result = {
            "proposals": rpn_output["proposals"],
            "roi_features": all_roi_features,
            "cls_scores": detection_output["cls_scores"],
            "bbox_deltas": detection_output["bbox_deltas"],
            "rpn_objectness": rpn_output["objectness_raw"],
            "rpn_bbox_deltas": rpn_output["bbox_deltas"]
        }
        
        # Compute losses if targets are provided
        if targets is not None and self.training:
            # This is a simplified loss computation
            # In practice, you'd need proper target assignment
            losses = self._compute_losses(result, targets)
            result.update(losses)
        
        return result
    
    def _compute_losses(self, predictions: Dict, targets: Dict) -> Dict:
        """Compute training losses."""
        # Simplified loss computation for demonstration
        # In practice, you'd need proper positive/negative sampling
        # and target assignment based on IoU with ground truth
        
        total_loss = torch.tensor(0.0, device=predictions["cls_scores"].device)
        
        # Dummy loss computation - replace with proper implementation
        if len(predictions["cls_scores"]) > 0:
            # Classification loss
            dummy_labels = torch.zeros(len(predictions["cls_scores"]), 
                                     dtype=torch.long, 
                                     device=predictions["cls_scores"].device)
            cls_loss = torch.nn.functional.cross_entropy(
                predictions["cls_scores"], dummy_labels
            )
            
            # Regression loss
            dummy_targets = torch.zeros_like(predictions["bbox_deltas"])
            reg_loss = torch.nn.functional.smooth_l1_loss(
                predictions["bbox_deltas"], dummy_targets
            )
            
            total_loss = cls_loss + reg_loss
        
        return {
            "total_loss": total_loss,
            "cls_loss": total_loss * 0.5,
            "reg_loss": total_loss * 0.5
        }


def train_stage1(args):
    """
    Main training function for Stage I.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = FewShotDetector(
        num_classes=args.num_classes,
        backbone_model=args.backbone_model,
        feature_dim=args.feature_dim
    ).to(device)
    
    # Unfreeze backbone for Stage I training
    model.backbone.unfreeze_backbone()
    
    # Create dataset and dataloader
    # Note: You'll need to provide your own dataset
    # This is a placeholder that expects COCO-format annotations
    if os.path.exists(args.train_annotations):
        train_dataset = BaseDetectionDataset(
            data_dir=args.train_data_dir,
            annotations_file=args.train_annotations,
            transforms=get_image_transforms(is_training=True)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda batch: batch  # Custom collate function needed
        )
    else:
        logger.warning("Training data not found. Creating dummy data for demonstration.")
        train_loader = create_dummy_dataloader(args, device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.epochs // 3, 2 * args.epochs // 3],
        gamma=0.1
    )
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Process batch (simplified for demonstration)
            if isinstance(batch, list):
                # Handle list of samples
                images = torch.stack([item['image'] for item in batch]).to(device)
                targets = {
                    'boxes': [item['boxes'].to(device) for item in batch],
                    'labels': [item['labels'].to(device) for item in batch]
                }
            else:
                # Handle tensor batch
                images, targets = batch
                images = images.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(images, targets if model.training else None)
            
            # Compute loss
            if "total_loss" in output:
                loss = output["total_loss"]
            else:
                # Dummy loss for demonstration
                loss = torch.tensor(1.0, device=device, requires_grad=True)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{epoch_loss/(batch_idx+1):.4f}"
            })
        
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.save_dir, 
                f"stage1_checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss / len(train_loader)
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, "stage1_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model: {final_model_path}")


def create_dummy_dataloader(args, device):
    """Create dummy dataloader for demonstration purposes."""
    class DummyDataset(Dataset):
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            return (
                torch.randn(3, 518, 518),
                {
                    'boxes': torch.tensor([[10, 10, 100, 100], [200, 200, 300, 300]]),
                    'labels': torch.tensor([1, 2])
                }
            )
    
    return DataLoader(DummyDataset(), batch_size=args.batch_size, shuffle=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stage I Base Training")
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=20,
                       help='Number of base classes')
    parser.add_argument('--backbone_model', type=str, default='dinov2_vitb14',
                       help='DINOv2 model variant')
    parser.add_argument('--feature_dim', type=int, default=256,
                       help='Feature dimension')
    
    # Data parameters
    parser.add_argument('--train_data_dir', type=str, default='data/train',
                       help='Training data directory')
    parser.add_argument('--train_annotations', type=str, default='data/annotations/train.json',
                       help='Training annotations file')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Save parameters
    parser.add_argument('--save_dir', type=str, default='checkpoints/stage1',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Start training
    train_stage1(args)