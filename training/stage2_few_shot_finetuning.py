"""
Stage II: Few-Shot Fine-Tuning Script

This script implements the second stage of the two-stage fine-tuning approach
where the feature extractor is frozen and only the classification and regression
heads are fine-tuned on few-shot novel class data.
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
import random
from collections import defaultdict

# Import model components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import DINOv2Backbone, FeatureExtractor, get_image_transforms
from models.rpn import RPN, ROIPooling
from models.detection_head import DetectionHead, compute_detection_loss
from training.stage1_base_training import FewShotDetector


class FewShotDataset(Dataset):
    """
    Dataset for few-shot learning with support and query sets.
    
    This dataset creates episodes where each episode contains:
    - Support set: Few examples of novel classes (K-shot)
    - Query set: Examples to classify using the support prototypes
    """
    
    def __init__(self, 
                 data_dir: str,
                 annotations_file: str,
                 n_way: int = 5,
                 k_shot: int = 5,
                 n_query: int = 15,
                 transforms: Optional[transforms.Compose] = None):
        self.data_dir = data_dir
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.transforms = transforms or get_image_transforms(is_training=False)
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.images = self.annotations['images']
        self.annotation_data = self.annotations['annotations']
        
        # Build class to images mapping
        self.class_to_images = defaultdict(list)
        self.img_id_to_info = {img['id']: img for img in self.images}
        
        for ann in self.annotation_data:
            img_id = ann['image_id']
            class_id = ann['category_id']
            self.class_to_images[class_id].append({
                'image_id': img_id,
                'bbox': ann['bbox'],
                'image_info': self.img_id_to_info[img_id]
            })
        
        # Filter classes that have enough examples
        self.valid_classes = [
            cls for cls, examples in self.class_to_images.items()
            if len(examples) >= (k_shot + n_query)
        ]
        
        if len(self.valid_classes) < n_way:
            raise ValueError(f"Not enough classes with sufficient examples. "
                           f"Need {n_way} classes with at least {k_shot + n_query} examples each.")
    
    def __len__(self):
        return 1000  # Number of episodes
    
    def __getitem__(self, idx):
        """
        Create a few-shot learning episode.
        
        Returns:
            Dictionary containing support and query sets
        """
        # Sample N classes for this episode
        episode_classes = random.sample(self.valid_classes, self.n_way)
        
        support_images = []
        support_boxes = []
        support_labels = []
        query_images = []
        query_boxes = []
        query_labels = []
        
        for class_idx, class_id in enumerate(episode_classes):
            # Sample examples for this class
            class_examples = random.sample(
                self.class_to_images[class_id], 
                self.k_shot + self.n_query
            )
            
            # Split into support and query
            support_examples = class_examples[:self.k_shot]
            query_examples = class_examples[self.k_shot:]
            
            # Process support examples
            for example in support_examples:
                image, box = self._load_example(example)
                support_images.append(image)
                support_boxes.append(box)
                support_labels.append(class_idx)  # Use episode-specific label
            
            # Process query examples
            for example in query_examples:
                image, box = self._load_example(example)
                query_images.append(image)
                query_boxes.append(box)
                query_labels.append(class_idx)  # Use episode-specific label
        
        return {
            'support_images': torch.stack(support_images),
            'support_boxes': torch.stack(support_boxes),
            'support_labels': torch.tensor(support_labels, dtype=torch.long),
            'query_images': torch.stack(query_images),
            'query_boxes': torch.stack(query_boxes),
            'query_labels': torch.tensor(query_labels, dtype=torch.long),
            'episode_classes': episode_classes
        }
    
    def _load_example(self, example: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess an individual example."""
        # Load image
        img_info = example['image_info']
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Get bounding box
        x, y, w, h = example['bbox']
        box = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
        
        # Apply transforms
        if self.transforms:
            image = self.transforms(image)
        
        return image, box


def extract_roi_features(model: FewShotDetector, images: torch.Tensor, 
                        boxes: torch.Tensor) -> torch.Tensor:
    """
    Extract ROI features from images and bounding boxes.
    
    Args:
        model: The detection model
        images: Input images (N, 3, H, W)
        boxes: Bounding boxes (N, 4)
        
    Returns:
        ROI features (N, feature_dim)
    """
    model.eval()
    with torch.no_grad():
        # Extract backbone features
        backbone_output = model.backbone(images)
        features = backbone_output["features"]
        stride = backbone_output["stride"]
        
        # Scale boxes to feature map coordinates
        spatial_scale = 1.0 / stride
        scaled_boxes = boxes * spatial_scale
        
        # Create ROI format (batch_idx, x1, y1, x2, y2)
        batch_indices = torch.arange(len(images), dtype=torch.float32, device=images.device)
        batch_indices = batch_indices.unsqueeze(1)
        rois = torch.cat([batch_indices, scaled_boxes], dim=1)
        
        # Extract ROI features using the proper method with spatial scale
        roi_features = model.feature_extractor(features, rois, spatial_scale)
        
        return roi_features


def train_stage2(args):
    """
    Main training function for Stage II few-shot fine-tuning.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load Stage I model
    if not os.path.exists(args.stage1_model_path):
        logger.error(f"Stage I model not found at {args.stage1_model_path}")
        logger.info("Please run Stage I training first or provide dummy model for demonstration")
        
        # Create dummy model for demonstration
        model = FewShotDetector(
            num_classes=args.num_base_classes,
            backbone_model=args.backbone_model,
            feature_dim=args.feature_dim
        ).to(device)
    else:
        model = FewShotDetector(
            num_classes=args.num_base_classes,
            backbone_model=args.backbone_model,
            feature_dim=args.feature_dim
        ).to(device)
        
        # Load pretrained weights
        checkpoint = torch.load(args.stage1_model_path, map_location=device)
        model.load_state_dict(checkpoint)
        logger.info(f"Loaded Stage I model from {args.stage1_model_path}")
    
    # Freeze feature extractor (backbone + early layers)
    model.backbone.freeze_backbone()
    model.detection_head.freeze_feature_layers()
    
    # Switch to few-shot mode
    model.detection_head.set_few_shot_mode(True)
    
    # Create dataset and dataloader
    if os.path.exists(args.novel_annotations):
        dataset = FewShotDataset(
            data_dir=args.novel_data_dir,
            annotations_file=args.novel_annotations,
            n_way=args.n_way,
            k_shot=args.k_shot,
            n_query=args.n_query,
            transforms=get_image_transforms(is_training=True)
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # One episode at a time
            shuffle=True,
            num_workers=args.num_workers
        )
    else:
        logger.warning("Novel class data not found. Creating dummy episodes for demonstration.")
        dataloader = create_dummy_episode_dataloader(args, device)
    
    # Create optimizer (only for unfrozen parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    model.train()
    # Keep backbone in eval mode since it's frozen
    model.backbone.eval()
    
    total_episodes = len(dataloader)
    accuracy_history = []
    
    for episode_idx, episode in enumerate(tqdm(dataloader, desc="Fine-tuning")):
        # Move episode to device
        support_images = episode['support_images'].squeeze(0).to(device)
        support_boxes = episode['support_boxes'].squeeze(0).to(device)
        support_labels = episode['support_labels'].squeeze(0).to(device)
        query_images = episode['query_images'].squeeze(0).to(device)
        query_boxes = episode['query_boxes'].squeeze(0).to(device)
        query_labels = episode['query_labels'].squeeze(0).to(device)
        
        # Extract features for support and query sets
        support_features = extract_roi_features(model, support_images, support_boxes)
        query_features = extract_roi_features(model, query_images, query_boxes)
        
        # Update prototypes using support set
        model.detection_head.few_shot_classifier.update_prototypes(
            support_features, support_labels
        )
        
        # Forward pass on query set
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            # Get similarity scores
            similarities = model.detection_head.few_shot_classifier(query_features)
            
            # Compute classification loss
            if len(similarities) > 0:
                loss = nn.functional.cross_entropy(similarities, query_labels)
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Compute accuracy for this episode
        if len(similarities) > 0:
            predictions = torch.argmax(similarities, dim=1)
            accuracy = (predictions == query_labels).float().mean().item()
            accuracy_history.append(accuracy)
        
        # Log progress
        if (episode_idx + 1) % args.log_interval == 0:
            avg_accuracy = np.mean(accuracy_history[-args.log_interval:]) if accuracy_history else 0.0
            logger.info(f"Episode {episode_idx + 1}/{total_episodes}, "
                       f"Loss: {loss.item():.4f}, "
                       f"Accuracy: {avg_accuracy:.4f}")
        
        # Save checkpoint
        if (episode_idx + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.save_dir, 
                f"stage2_checkpoint_episode_{episode_idx + 1}.pth"
            )
            torch.save({
                'episode': episode_idx + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy_history': accuracy_history
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, "stage2_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    
    # Log final results
    final_accuracy = np.mean(accuracy_history) if accuracy_history else 0.0
    logger.info(f"Final average accuracy: {final_accuracy:.4f}")
    logger.info(f"Saved final model: {final_model_path}")


def create_dummy_episode_dataloader(args, device):
    """Create dummy episode dataloader for demonstration."""
    class DummyEpisodeDataset(Dataset):
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            n_support = args.n_way * args.k_shot
            n_query = args.n_way * args.n_query
            
            return {
                'support_images': torch.randn(n_support, 3, 518, 518),
                'support_boxes': torch.randint(10, 500, (n_support, 4)).float(),
                'support_labels': torch.repeat_interleave(
                    torch.arange(args.n_way), args.k_shot
                ),
                'query_images': torch.randn(n_query, 3, 518, 518),
                'query_boxes': torch.randint(10, 500, (n_query, 4)).float(),
                'query_labels': torch.repeat_interleave(
                    torch.arange(args.n_way), args.n_query
                ),
                'episode_classes': list(range(args.n_way))
            }
    
    return DataLoader(DummyEpisodeDataset(), batch_size=1, shuffle=True)


def evaluate_few_shot(model: FewShotDetector, dataloader: DataLoader, device: torch.device):
    """
    Evaluate few-shot performance on test episodes.
    
    Args:
        model: Trained few-shot detection model
        dataloader: Test episode dataloader
        device: Device to run evaluation on
        
    Returns:
        Average accuracy across all test episodes
    """
    model.eval()
    model.detection_head.set_few_shot_mode(True)
    
    accuracies = []
    
    with torch.no_grad():
        for episode in tqdm(dataloader, desc="Evaluating"):
            # Move episode to device
            support_images = episode['support_images'].squeeze(0).to(device)
            support_boxes = episode['support_boxes'].squeeze(0).to(device)
            support_labels = episode['support_labels'].squeeze(0).to(device)
            query_images = episode['query_images'].squeeze(0).to(device)
            query_boxes = episode['query_boxes'].squeeze(0).to(device)
            query_labels = episode['query_labels'].squeeze(0).to(device)
            
            # Extract features
            support_features = extract_roi_features(model, support_images, support_boxes)
            query_features = extract_roi_features(model, query_images, query_boxes)
            
            # Update prototypes
            model.detection_head.few_shot_classifier.update_prototypes(
                support_features, support_labels
            )
            
            # Get predictions
            similarities = model.detection_head.few_shot_classifier(query_features)
            
            if len(similarities) > 0:
                predictions = torch.argmax(similarities, dim=1)
                accuracy = (predictions == query_labels).float().mean().item()
                accuracies.append(accuracy)
    
    return np.mean(accuracies) if accuracies else 0.0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stage II Few-Shot Fine-Tuning")
    
    # Model parameters
    parser.add_argument('--num_base_classes', type=int, default=20,
                       help='Number of base classes from Stage I')
    parser.add_argument('--backbone_model', type=str, default='dinov2_vitb14',
                       help='DINOv2 model variant')
    parser.add_argument('--feature_dim', type=int, default=256,
                       help='Feature dimension')
    parser.add_argument('--stage1_model_path', type=str, 
                       default='checkpoints/stage1/stage1_final_model.pth',
                       help='Path to Stage I pretrained model')
    
    # Few-shot parameters
    parser.add_argument('--n_way', type=int, default=5,
                       help='Number of novel classes per episode')
    parser.add_argument('--k_shot', type=int, default=5,
                       help='Number of support examples per class')
    parser.add_argument('--n_query', type=int, default=15,
                       help='Number of query examples per class')
    
    # Data parameters
    parser.add_argument('--novel_data_dir', type=str, default='data/novel',
                       help='Novel class data directory')
    parser.add_argument('--novel_annotations', type=str, 
                       default='data/annotations/novel.json',
                       help='Novel class annotations file')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate for fine-tuning')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default='checkpoints/stage2',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Save checkpoint every N episodes')
    parser.add_argument('--log_interval', type=int, default=20,
                       help='Log progress every N episodes')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Start fine-tuning
    train_stage2(args)