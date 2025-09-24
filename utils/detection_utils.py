"""
Utility functions for the few-shot object detection system.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import json
import os


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of bounding boxes.
    
    Args:
        box1: Boxes of shape (..., 4) in [x1, y1, x2, y2] format
        box2: Boxes of shape (..., 4) in [x1, y1, x2, y2] format
    
    Returns:
        IoU values
    """
    # Get intersection coordinates
    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])
    
    # Compute intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Compute areas of both boxes
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    
    # Compute union area
    union = area1 + area2 - intersection
    
    # Compute IoU
    iou = intersection / (union + 1e-6)
    
    return iou


def assign_targets(proposals: torch.Tensor, 
                  gt_boxes: torch.Tensor, 
                  gt_labels: torch.Tensor,
                  pos_threshold: float = 0.5,
                  neg_threshold: float = 0.4) -> Dict[str, torch.Tensor]:
    """
    Assign ground truth targets to proposals based on IoU.
    
    Args:
        proposals: Proposed boxes (N, 4)
        gt_boxes: Ground truth boxes (M, 4)
        gt_labels: Ground truth labels (M,)
        pos_threshold: IoU threshold for positive samples
        neg_threshold: IoU threshold for negative samples
    
    Returns:
        Dictionary containing assigned targets
    """
    if len(proposals) == 0:
        return {
            'labels': torch.empty(0, dtype=torch.long, device=proposals.device),
            'bbox_targets': torch.empty(0, 4, device=proposals.device),
            'bbox_weights': torch.empty(0, 4, device=proposals.device)
        }
    
    # Compute IoU matrix
    ious = compute_iou(proposals.unsqueeze(1), gt_boxes.unsqueeze(0))  # (N, M)
    
    # Find best GT for each proposal
    max_ious, best_gt_indices = torch.max(ious, dim=1)  # (N,)
    
    # Assign labels
    labels = torch.zeros(len(proposals), dtype=torch.long, device=proposals.device)
    
    # Positive samples
    pos_mask = max_ious >= pos_threshold
    labels[pos_mask] = gt_labels[best_gt_indices[pos_mask]]
    
    # Negative samples
    neg_mask = max_ious < neg_threshold
    labels[neg_mask] = 0  # Background class
    
    # Compute bbox targets for positive samples
    bbox_targets = torch.zeros_like(proposals)
    bbox_weights = torch.zeros_like(proposals)
    
    if pos_mask.sum() > 0:
        pos_proposals = proposals[pos_mask]
        pos_gt_boxes = gt_boxes[best_gt_indices[pos_mask]]
        
        # Compute deltas
        bbox_targets[pos_mask] = compute_bbox_deltas(pos_proposals, pos_gt_boxes)
        bbox_weights[pos_mask] = 1.0
    
    return {
        'labels': labels,
        'bbox_targets': bbox_targets,
        'bbox_weights': bbox_weights
    }


def compute_bbox_deltas(boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute bounding box regression deltas.
    
    Args:
        boxes: Source boxes (N, 4)
        gt_boxes: Target boxes (N, 4)
    
    Returns:
        Regression deltas (N, 4)
    """
    # Convert to center format
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights
    
    # Compute deltas
    dx = (gt_ctr_x - ctr_x) / widths
    dy = (gt_ctr_y - ctr_y) / heights
    dw = torch.log(gt_widths / widths)
    dh = torch.log(gt_heights / heights)
    
    deltas = torch.stack([dx, dy, dw, dh], dim=1)
    
    return deltas


def apply_bbox_deltas(boxes: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """
    Apply bounding box regression deltas to boxes.
    
    Args:
        boxes: Source boxes (N, 4)
        deltas: Regression deltas (N, 4)
    
    Returns:
        Transformed boxes (N, 4)
    """
    # Convert to center format
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    
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


def visualize_episode(support_images: torch.Tensor,
                     support_boxes: torch.Tensor,
                     support_labels: torch.Tensor,
                     query_images: torch.Tensor,
                     query_boxes: torch.Tensor,
                     query_labels: torch.Tensor,
                     class_names: Optional[List[str]] = None,
                     save_path: Optional[str] = None):
    """
    Visualize a few-shot learning episode.
    
    Args:
        support_images: Support images (K*N, 3, H, W)
        support_boxes: Support boxes (K*N, 4)
        support_labels: Support labels (K*N,)
        query_images: Query images (Q*N, 3, H, W)
        query_boxes: Query boxes (Q*N, 4)
        query_labels: Query labels (Q*N,)
        class_names: Optional class names
        save_path: Path to save visualization
    """
    n_support = len(support_images)
    n_query = len(query_images)
    
    # Create figure
    fig, axes = plt.subplots(2, max(n_support, n_query), 
                            figsize=(15, 8))
    
    # Define colors for different classes
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot support examples
    for i in range(n_support):
        ax = axes[0, i] if n_support > 1 else axes[0]
        
        # Convert tensor to numpy
        img = support_images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
        
        ax.imshow(img)
        
        # Draw bounding box
        box = support_boxes[i].cpu().numpy()
        label = support_labels[i].item()
        color = colors[label % len(colors)]
        
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                               linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        class_name = class_names[label] if class_names else f'Class {label}'
        ax.text(box[0], box[1] - 5, f'Support: {class_name}', 
               color=color, fontweight='bold')
        
        ax.set_title(f'Support {i+1}')
        ax.axis('off')
    
    # Plot query examples
    for i in range(n_query):
        ax = axes[1, i] if n_query > 1 else axes[1]
        
        # Convert tensor to numpy
        img = query_images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
        
        ax.imshow(img)
        
        # Draw bounding box
        box = query_boxes[i].cpu().numpy()
        label = query_labels[i].item()
        color = colors[label % len(colors)]
        
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                               linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        class_name = class_names[label] if class_names else f'Class {label}'
        ax.text(box[0], box[1] - 5, f'Query: {class_name}', 
               color=color, fontweight='bold')
        
        ax.set_title(f'Query {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_coco_format_annotation(image_dir: str, 
                                 output_file: str,
                                 class_mapping: Dict[str, int]):
    """
    Create COCO format annotation file from a directory structure.
    
    Expected structure:
    image_dir/
    ├── class1/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── class2/
    │   ├── img3.jpg
    │   └── img4.jpg
    
    Args:
        image_dir: Directory containing class subdirectories
        output_file: Path to output JSON file
        class_mapping: Mapping from class names to IDs
    """
    images = []
    annotations = []
    
    img_id = 1
    ann_id = 1
    
    for class_name in os.listdir(image_dir):
        class_path = os.path.join(image_dir, class_name)
        
        if not os.path.isdir(class_path) or class_name not in class_mapping:
            continue
        
        class_id = class_mapping[class_name]
        
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(class_path, img_name)
            
            # Load image to get dimensions
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except:
                continue
            
            # Add image entry
            images.append({
                'id': img_id,
                'file_name': os.path.join(class_name, img_name),
                'width': width,
                'height': height
            })
            
            # Create dummy bounding box (you'll need to provide real annotations)
            # This creates a box covering 80% of the image centered
            margin_x = width * 0.1
            margin_y = height * 0.1
            bbox = [margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y]
            
            # Add annotation entry
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': class_id,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0
            })
            
            img_id += 1
            ann_id += 1
    
    # Create categories
    categories = [
        {'id': class_id, 'name': class_name} 
        for class_name, class_id in class_mapping.items()
    ]
    
    # Create final annotation structure
    coco_format = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"Created COCO annotation file with {len(images)} images and {len(annotations)} annotations")


def compute_map(detections: List[Dict], 
               ground_truths: List[Dict],
               iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP) for object detection.
    
    Args:
        detections: List of detection dictionaries
        ground_truths: List of ground truth dictionaries
        iou_threshold: IoU threshold for positive detection
    
    Returns:
        Dictionary with mAP metrics
    """
    # This is a simplified mAP computation
    # For production use, consider using pycocotools
    
    if not detections or not ground_truths:
        return {'mAP': 0.0, 'mAP50': 0.0, 'mAP75': 0.0}
    
    # Group by image and class
    det_by_img_cls = {}
    gt_by_img_cls = {}
    
    for det in detections:
        key = (det['image_id'], det['class'])
        if key not in det_by_img_cls:
            det_by_img_cls[key] = []
        det_by_img_cls[key].append(det)
    
    for gt in ground_truths:
        key = (gt['image_id'], gt['class'])
        if key not in gt_by_img_cls:
            gt_by_img_cls[key] = []
        gt_by_img_cls[key].append(gt)
    
    # Compute AP for each class
    all_keys = set(list(det_by_img_cls.keys()) + list(gt_by_img_cls.keys()))
    
    aps = []
    for key in all_keys:
        dets = det_by_img_cls.get(key, [])
        gts = gt_by_img_cls.get(key, [])
        
        if not gts:
            continue
        
        # Sort detections by confidence
        dets = sorted(dets, key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Match detections to ground truths
        tp = []
        fp = []
        
        gt_matched = [False] * len(gts)
        
        for det in dets:
            det_box = torch.tensor(det['bbox'])
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gts):
                if gt_matched[gt_idx]:
                    continue
                
                gt_box = torch.tensor(gt['bbox'])
                iou = compute_iou(det_box.unsqueeze(0), gt_box.unsqueeze(0)).item()
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp.append(1)
                fp.append(0)
                gt_matched[best_gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Compute AP
        ap = 0
        for i in range(1, len(recalls)):
            ap += (recalls[i] - recalls[i-1]) * precisions[i]
        
        aps.append(ap)
    
    mean_ap = np.mean(aps) if aps else 0.0
    
    return {
        'mAP': mean_ap,
        'mAP50': mean_ap,  # Simplified - should compute at IoU=0.5
        'mAP75': mean_ap,  # Simplified - should compute at IoU=0.75
        'num_classes': len(aps)
    }


if __name__ == "__main__":
    # Test IoU computation
    box1 = torch.tensor([[10, 10, 50, 50]])
    box2 = torch.tensor([[20, 20, 60, 60]])
    
    iou = compute_iou(box1, box2)
    print(f"IoU: {iou.item():.3f}")
    
    # Test bbox delta computation
    deltas = compute_bbox_deltas(box1, box2)
    restored = apply_bbox_deltas(box1, deltas)
    print(f"Original box2: {box2}")
    print(f"Restored box2: {restored}")
    print(f"Restoration error: {torch.abs(box2 - restored).max().item():.6f}")