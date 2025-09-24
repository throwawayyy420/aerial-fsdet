"""
Example script to run the complete few-shot detection pipeline.

This script demonstrates how to use the two-stage training approach
and test the final model on new data.
"""

import torch
import os
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt

# Import our modules
from models.backbone import DINOv2Backbone, get_image_transforms
from training.stage1_base_training import FewShotDetector
from training.stage2_few_shot_finetuning import extract_roi_features
from demo.few_shot_demo import FewShotDetectionDemo
from utils.detection_utils import visualize_episode


def create_sample_dataset():
    """Create a simple sample dataset for testing."""
    print("Creating sample dataset...")
    
    # Create directories
    os.makedirs("data/sample/train", exist_ok=True)
    os.makedirs("data/sample/novel", exist_ok=True)
    os.makedirs("data/sample/annotations", exist_ok=True)
    
    # Create simple colored rectangles as training data
    from PIL import ImageDraw
    
    # Base classes: red rectangles, blue circles
    colors_shapes = [
        ("red_rect", (255, 0, 0), "rectangle"),
        ("blue_circle", (0, 0, 255), "circle"),
        ("green_rect", (0, 255, 0), "rectangle"),
    ]
    
    images_data = []
    annotations_data = []
    
    img_id = 1
    ann_id = 1
    
    # Generate training images
    for class_id, (class_name, color, shape) in enumerate(colors_shapes, 1):
        for i in range(10):  # 10 images per class
            # Create image
            img = Image.new('RGB', (300, 300), 'white')
            draw = ImageDraw.Draw(img)
            
            # Random position and size
            import random
            x = random.randint(50, 150)
            y = random.randint(50, 150)
            size = random.randint(50, 100)
            
            if shape == "rectangle":
                draw.rectangle([x, y, x + size, y + size], 
                             fill=color, outline='black', width=2)
                bbox = [x, y, size, size]
            else:  # circle
                draw.ellipse([x, y, x + size, y + size], 
                           fill=color, outline='black', width=2)
                bbox = [x, y, size, size]
            
            # Save image
            filename = f"{class_name}_{i:03d}.png"
            img.save(f"data/sample/train/{filename}")
            
            # Add to annotations
            images_data.append({
                "id": img_id,
                "file_name": filename,
                "width": 300,
                "height": 300
            })
            
            annotations_data.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_id,
                "bbox": bbox,
                "area": size * size,
                "iscrowd": 0
            })
            
            img_id += 1
            ann_id += 1
    
    # Save training annotations
    train_annotations = {
        "images": images_data,
        "annotations": annotations_data,
        "categories": [
            {"id": i, "name": name} 
            for i, (name, _, _) in enumerate(colors_shapes, 1)
        ]
    }
    
    with open("data/sample/annotations/train.json", 'w') as f:
        json.dump(train_annotations, f, indent=2)
    
    # Create novel classes for few-shot learning
    novel_colors_shapes = [
        ("yellow_rect", (255, 255, 0), "rectangle"),
        ("purple_circle", (128, 0, 128), "circle"),
    ]
    
    novel_images_data = []
    novel_annotations_data = []
    
    img_id = 1
    ann_id = 1
    
    for class_id, (class_name, color, shape) in enumerate(novel_colors_shapes, 1):
        for i in range(15):  # 15 images per novel class (5 support + 10 query)
            # Create image
            img = Image.new('RGB', (300, 300), 'white')
            draw = ImageDraw.Draw(img)
            
            # Random position and size
            x = random.randint(50, 150)
            y = random.randint(50, 150)
            size = random.randint(50, 100)
            
            if shape == "rectangle":
                draw.rectangle([x, y, x + size, y + size], 
                             fill=color, outline='black', width=2)
                bbox = [x, y, size, size]
            else:  # circle
                draw.ellipse([x, y, x + size, y + size], 
                           fill=color, outline='black', width=2)
                bbox = [x, y, size, size]
            
            # Save image
            filename = f"{class_name}_{i:03d}.png"
            img.save(f"data/sample/novel/{filename}")
            
            # Add to annotations
            novel_images_data.append({
                "id": img_id,
                "file_name": filename,
                "width": 300,
                "height": 300
            })
            
            novel_annotations_data.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_id,
                "bbox": bbox,
                "area": size * size,
                "iscrowd": 0
            })
            
            img_id += 1
            ann_id += 1
    
    # Save novel annotations
    novel_annotations = {
        "images": novel_images_data,
        "annotations": novel_annotations_data,
        "categories": [
            {"id": i, "name": name} 
            for i, (name, _, _) in enumerate(novel_colors_shapes, 1)
        ]
    }
    
    with open("data/sample/annotations/novel.json", 'w') as f:
        json.dump(novel_annotations, f, indent=2)
    
    print("Sample dataset created!")
    print(f"Training: {len(images_data)} images, {len(colors_shapes)} classes")
    print(f"Novel: {len(novel_images_data)} images, {len(novel_colors_shapes)} classes")


def run_stage1_training():
    """Run Stage I base training with sample data."""
    print("\n" + "="*50)
    print("STAGE I: BASE TRAINING")
    print("="*50)
    
    # Enable anomaly detection for gradient debugging
    torch.autograd.set_detect_anomaly(True)
    
    # Create checkpoint directory
    os.makedirs("checkpoints/stage1", exist_ok=True)
    
    # Import and run training
    from training.stage1_base_training import train_stage1
    
    class Args:
        num_classes = 3
        backbone_model = 'dinov2_vitb14'
        feature_dim = 256
        train_data_dir = 'data/sample/train'
        train_annotations = 'data/sample/annotations/train.json'
        batch_size = 2
        epochs = 5  # Reduced for demo
        learning_rate = 1e-4
        weight_decay = 1e-4
        num_workers = 2
        save_dir = 'checkpoints/stage1'
        save_interval = 2
    
    args = Args()
    
    try:
        train_stage1(args)
        print("Stage I training completed!")
    except Exception as e:
        print(f"Stage I training failed: {e}")
        import traceback
        traceback.print_exc()
        print("Creating dummy model for demonstration...")
        
        # Create and save a dummy model
        model = FewShotDetector(
            num_classes=args.num_classes,
            backbone_model=args.backbone_model,
            feature_dim=args.feature_dim
        )
        
        torch.save(model.state_dict(), 'checkpoints/stage1/stage1_final_model.pth')
        print("Dummy model saved!")


def run_stage2_finetuning():
    """Run Stage II few-shot fine-tuning."""
    print("\n" + "="*50)
    print("STAGE II: FEW-SHOT FINE-TUNING")
    print("="*50)
    
    # Create checkpoint directory
    os.makedirs("checkpoints/stage2", exist_ok=True)
    
    from training.stage2_few_shot_finetuning import train_stage2
    
    class Args:
        num_base_classes = 3
        backbone_model = 'dinov2_vitb14'
        feature_dim = 256
        stage1_model_path = 'checkpoints/stage1/stage1_final_model.pth'
        n_way = 2
        k_shot = 5
        n_query = 10
        novel_data_dir = 'data/sample/novel'
        novel_annotations = 'data/sample/annotations/novel.json'
        learning_rate = 1e-2
        weight_decay = 1e-3
        num_workers = 2
        save_dir = 'checkpoints/stage2'
        save_interval = 20
        log_interval = 10
    
    args = Args()
    
    try:
        train_stage2(args)
        print("Stage II fine-tuning completed!")
    except Exception as e:
        print(f"Stage II fine-tuning failed: {e}")
        print("This is expected without proper training data")


def run_demo():
    """Run the enhanced interactive demo with click-and-drag bounding boxes."""
    print("\n" + "="*50)
    print("ENHANCED INTERACTIVE DEMO")
    print("="*50)
    print("🎯 NEW: Click-and-drag bounding box selection!")
    print("📋 Instructions:")
    print("  1. Upload image → 2. Click two corners → 3. Add class name → 4. Done!")
    print()
    
    # Check if trained model exists
    model_path = 'checkpoints/stage2/stage2_final_model.pth'
    if not os.path.exists(model_path):
        model_path = None
        print("No trained model found. Using random initialization.")
    
    print("Starting enhanced demo...")
    print("Open http://localhost:7860 in your browser")
    
    from demo.few_shot_demo import create_gradio_interface
    
    interface = create_gradio_interface()
    interface.launch(share=False, server_name="0.0.0.0", server_port=7860)


def test_model_inference():
    """Test model inference on sample images."""
    print("\n" + "="*50)
    print("MODEL INFERENCE TEST")
    print("="*50)
    
    # Create demo instance
    model_path = 'checkpoints/stage2/stage2_final_model.pth'
    if not os.path.exists(model_path):
        model_path = None
    
    demo = FewShotDetectionDemo(model_path=model_path)
    
    # Load a sample image
    if os.path.exists("data/sample/novel"):
        import glob
        sample_images = glob.glob("data/sample/novel/*.png")[:3]
        
        if sample_images:
            print("Testing inference on sample images...")
            
            for img_path in sample_images:
                image = Image.open(img_path)
                print(f"\nProcessing: {os.path.basename(img_path)}")
                
                # Add as support example (for demo purposes)
                bbox = [50, 50, 150, 150]  # Dummy bbox
                demo.update_prototypes([(image, bbox, 0)])
                
                # Test detection
                detections = demo.detect_objects(image, confidence_threshold=0.1)
                print(f"Found {len(detections)} detections")
                
                # Visualize
                vis_image = demo.visualize_detections(image, detections, ["yellow_rect"])
                
                # Save result
                output_path = f"results_{os.path.basename(img_path)}"
                vis_image.save(output_path)
                print(f"Saved visualization: {output_path}")
        else:
            print("No sample images found")
    else:
        print("Sample data directory not found")


def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(description="Few-Shot Detection Pipeline")
    parser.add_argument('--step', type=str, 
                       choices=['data', 'stage1', 'stage2', 'demo', 'test', 'all'],
                       default='all',
                       help='Which step to run')
    
    args = parser.parse_args()
    
    print("Few-Shot Object Detection Pipeline")
    print("Using DINOv2 backbone with two-stage training")
    print("-" * 50)
    
    if args.step in ['data', 'all']:
        create_sample_dataset()
    
    if args.step in ['stage1', 'all']:
        run_stage1_training()
    
    if args.step in ['stage2', 'all']:
        run_stage2_finetuning()
    
    if args.step in ['test', 'all']:
        test_model_inference()
    
    if args.step in ['demo']:
        run_demo()
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED!")
    print("="*50)
    print("To run the interactive demo, use:")
    print("python run_pipeline.py --step demo")


if __name__ == "__main__":
    main()