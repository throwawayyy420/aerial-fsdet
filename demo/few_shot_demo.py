"""
Interactive Demo for Few-Shot Object Detection

This script provides an interactive demonstration of the two-stage few-shot
object detection system using DINOv2 backbone. It allows users to:

1. Load a pretrained model (or use a demo model)
2. Provide support examples for novel classes
3. Test detection on query images
4. Visualize results with bounding boxes and confidence scores
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse
import os
import json
from typing import List, Dict, Tuple, Optional
import gradio as gr

# Import model components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import DINOv2Backbone, FeatureExtractor, get_image_transforms
from models.rpn import RPN, ROIPooling
from models.detection_head import DetectionHead
from training.stage1_base_training import FewShotDetector
from training.stage2_few_shot_finetuning import extract_roi_features


class FewShotDetectionDemo:
    """
    Interactive demo class for few-shot object detection.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize the demo with a pretrained model.
        
        Args:
            model_path: Path to pretrained model weights
            device: Device to run inference on
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load or create model
        self.model = self._load_model(model_path)
        self.model.eval()
        self.model.detection_head.set_few_shot_mode(True)
        
        # Image preprocessing
        self.transform = get_image_transforms(is_training=False)
        
        # Color palette for visualization
        self.colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
            '#00FFFF', '#800000', '#008000', '#000080', '#808000',
            '#800080', '#008080', '#C0C0C0', '#808080', '#FFA500'
        ]
        
        print("Demo initialized successfully!")
    
    def _load_model(self, model_path: Optional[str]) -> FewShotDetector:
        """Load the detection model."""
        # Create model architecture
        model = FewShotDetector(
            num_classes=20,  # This will be overridden in few-shot mode
            backbone_model="dinov2_vitb14",
            feature_dim=256
        ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            # Load pretrained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        else:
            print("Using randomly initialized model for demonstration")
            print("For best results, provide a path to a trained model")
        
        return model
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess an image for model input."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        
        # Store original size
        original_size = image.size
        
        # Apply transforms
        tensor_image = self.transform(image).unsqueeze(0).to(self.device)
        
        return tensor_image, original_size
    
    def detect_objects_with_boxes(self, image: Image.Image, 
                                 bounding_boxes: List[List[float]]) -> torch.Tensor:
        """
        Extract features from specified bounding boxes in an image.
        
        Args:
            image: Input image
            bounding_boxes: List of [x1, y1, x2, y2] coordinates
            
        Returns:
            Feature tensor for the bounding boxes
        """
        # Preprocess image
        tensor_image, original_size = self.preprocess_image(image)
        
        # Convert boxes to tensor
        boxes = torch.tensor(bounding_boxes, dtype=torch.float32, device=self.device)
        
        # Scale boxes to input image size (518x518)
        input_size = 518
        scale_x = input_size / original_size[0]
        scale_y = input_size / original_size[1]
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        # Extract features
        features = extract_roi_features(self.model, tensor_image, boxes)
        
        return features
    
    def update_prototypes(self, support_images_and_boxes: List[Tuple[Image.Image, List[float], int]]):
        """
        Update model prototypes with support examples.
        
        Args:
            support_images_and_boxes: List of (image, bbox, class_label) tuples
        """
        all_features = []
        all_labels = []
        
        for image, bbox, label in support_images_and_boxes:
            features = self.detect_objects_with_boxes(image, [bbox])
            all_features.append(features)
            all_labels.append(label)
        
        if all_features:
            support_features = torch.cat(all_features, dim=0)
            support_labels = torch.tensor(all_labels, dtype=torch.long, device=self.device)
            
            # Update prototypes
            self.model.detection_head.few_shot_classifier.update_prototypes(
                support_features, support_labels
            )
            
            print(f"Updated prototypes with {len(all_features)} support examples")
    
    def detect_objects(self, image: Image.Image, 
                      confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in an image using the current prototypes.
        
        Args:
            image: Query image
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of detection dictionaries
        """
        # Preprocess image
        tensor_image, original_size = self.preprocess_image(image)
        
        with torch.no_grad():
            # Get model predictions
            output = self.model(tensor_image)
            
            # Extract proposals and features
            proposals = output["proposals"][0]  # First (only) image in batch
            roi_features = output["roi_features"]
            
            if len(proposals) == 0:
                return []
            
            # Get few-shot predictions
            similarities = self.model.detection_head.few_shot_classifier(roi_features)
            
            if len(similarities) == 0:
                return []
            
            # Convert similarities to probabilities
            probs = F.softmax(similarities, dim=1)
            max_probs, predicted_classes = torch.max(probs, dim=1)
            
            # Filter by confidence
            confident_mask = max_probs >= confidence_threshold
            
            if confident_mask.sum() == 0:
                return []
            
            # Get confident detections
            confident_proposals = proposals[confident_mask]
            confident_probs = max_probs[confident_mask]
            confident_classes = predicted_classes[confident_mask]
            
            # Scale boxes back to original image size
            input_size = 518
            scale_x = original_size[0] / input_size
            scale_y = original_size[1] / input_size
            
            confident_proposals[:, [0, 2]] *= scale_x
            confident_proposals[:, [1, 3]] *= scale_y
            
            # Create detection results
            detections = []
            for i in range(len(confident_proposals)):
                detection = {
                    'bbox': confident_proposals[i].cpu().numpy().tolist(),
                    'class': confident_classes[i].item(),
                    'confidence': confident_probs[i].item()
                }
                detections.append(detection)
            
            return detections
    
    def visualize_detections(self, image: Image.Image, 
                           detections: List[Dict],
                           class_names: Optional[List[str]] = None) -> Image.Image:
        """
        Visualize detection results on an image.
        
        Args:
            image: Original image
            detections: List of detection dictionaries
            class_names: Optional list of class names
            
        Returns:
            Image with visualized detections
        """
        # Create a copy of the image
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Try to load a font (fallback to default if not available)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            class_id = detection['class']
            confidence = detection['confidence']
            
            # Get color for this class
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            draw.rectangle(bbox, outline=color, width=3)
            
            # Create label
            if class_names and class_id < len(class_names):
                label = f"{class_names[class_id]}: {confidence:.2f}"
            else:
                label = f"Class {class_id}: {confidence:.2f}"
            
            # Draw label background
            text_bbox = draw.textbbox((bbox[0], bbox[1] - 25), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            
            # Draw label text
            draw.text((bbox[0], bbox[1] - 25), label, fill='white', font=font)
        
        return vis_image


def create_gradio_interface():
    """Create a Gradio interface with click-and-drag bounding box selection."""
    
    # Initialize demo
    demo = FewShotDetectionDemo()
    
    # Global state to store class information and current bounding box
    class_info = {"names": [], "support_examples": [], "current_bbox": None, "current_image": None}
    
    def draw_bbox_on_image(image, bbox):
        """Draw bounding box on image for visualization."""
        if image is None or bbox is None:
            return image
        
        # Create a copy of the image
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Draw bounding box
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        
        # Add corner indicators
        corner_size = 5
        draw.rectangle([x1-corner_size, y1-corner_size, x1+corner_size, y1+corner_size], fill='red')
        draw.rectangle([x2-corner_size, y2-corner_size, x2+corner_size, y2+corner_size], fill='red')
        
        return img_copy
    
    def handle_image_click(image, evt: gr.SelectData):
        """Handle clicks on the support image to define bounding box."""
        if image is None:
            return image, "Please upload an image first"
        
        # Store the image
        class_info["current_image"] = image
        
        # Get click coordinates
        x, y = evt.index[0], evt.index[1]
        
        if class_info["current_bbox"] is None:
            # First click - start of bounding box
            class_info["current_bbox"] = [x, y, x, y]
            status = f"Bounding box started at ({x}, {y}). Click again to set the opposite corner."
        else:
            # Second click - complete the bounding box
            x1, y1 = class_info["current_bbox"][:2]
            
            # Ensure proper box coordinates (top-left to bottom-right)
            bbox = [
                min(x1, x), min(y1, y),  # top-left
                max(x1, x), max(y1, y)   # bottom-right
            ]
            
            class_info["current_bbox"] = bbox
            
            # Draw bbox on image
            image_with_bbox = draw_bbox_on_image(image, bbox)
            
            status = f"Bounding box completed: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})"
            
            return image_with_bbox, status
        
        return image, status
    
    def add_support_example_with_bbox(class_name):
        """Add a support example using the drawn bounding box."""
        if class_info["current_image"] is None:
            return "Please upload and click on an image first", ""
        
        if class_info["current_bbox"] is None:
            return "Please draw a bounding box by clicking two points on the image", ""
        
        if not class_name.strip():
            return "Please provide a class name", ""
        
        try:
            bbox_coords = class_info["current_bbox"]
            image = class_info["current_image"]
            
            # Add class if new
            if class_name not in class_info["names"]:
                class_info["names"].append(class_name)
            
            class_id = class_info["names"].index(class_name)
            
            # Store support example
            class_info["support_examples"].append((image, bbox_coords, class_id))
            
            # Update prototypes
            demo.update_prototypes(class_info["support_examples"])
            
            # Reset bbox for next example
            class_info["current_bbox"] = None
            
            status = f"Added support example for '{class_name}'. Total examples: {len(class_info['support_examples'])}"
            classes_str = "Current classes: " + ", ".join(class_info["names"])
            
            return status, classes_str
            
        except Exception as e:
            return f"Error: {str(e)}", ""
    
    def clear_bbox():
        """Clear the current bounding box."""
        class_info["current_bbox"] = None
        if class_info["current_image"] is not None:
            return class_info["current_image"], "Bounding box cleared. Click two points to draw a new one."
        return None, "Bounding box cleared."
    
    def detect_and_visualize(image, confidence_threshold):
        """Detect objects and return visualization."""
        if image is None:
            return None, "Please provide an image"
        
        if len(class_info["support_examples"]) == 0:
            return None, "Please add support examples first"
        
        try:
            # Detect objects
            detections = demo.detect_objects(image, confidence_threshold)
            
            # Visualize results
            vis_image = demo.visualize_detections(image, detections, class_info["names"])
            
            # Create results summary
            if detections:
                results_text = f"Found {len(detections)} objects:\\n"
                for i, det in enumerate(detections):
                    class_name = class_info["names"][det['class']] if det['class'] < len(class_info["names"]) else f"Class {det['class']}"
                    results_text += f"{i+1}. {class_name}: {det['confidence']:.3f}\\n"
            else:
                results_text = "No objects detected above confidence threshold"
            
            return vis_image, results_text
            
        except Exception as e:
            return None, f"Error during detection: {str(e)}"
    
    def reset_demo():
        """Reset the demo state."""
        class_info["names"] = []
        class_info["support_examples"] = []
        class_info["current_bbox"] = None
        class_info["current_image"] = None
        return "Demo reset successfully", "", None, ""
    
    # Create Gradio interface
    with gr.Blocks(title="Few-Shot Object Detection Demo") as interface:
        gr.Markdown("# Few-Shot Object Detection with DINOv2")
        gr.Markdown("This demo showcases two-stage few-shot object detection. First, provide support examples for novel classes, then test detection on query images.")
        
        with gr.Tab("Add Support Examples"):
            gr.Markdown("### Step 1: Add Support Examples")
            gr.Markdown("📋 **Instructions:**")
            gr.Markdown("1. Upload an image")
            gr.Markdown("2. Click two points on the image to draw a bounding box (top-left and bottom-right corners)")
            gr.Markdown("3. Enter a class name")
            gr.Markdown("4. Click 'Add Support Example'")
            
            with gr.Row():
                with gr.Column():
                    support_image = gr.Image(type="pil", label="Support Image (Click to draw bounding box)")
                    
                    with gr.Row():
                        class_name_input = gr.Textbox(
                            label="Class Name",
                            placeholder="car",
                            info="Name for this object class"
                        )
                        clear_bbox_button = gr.Button("Clear Bounding Box", variant="secondary")
                    
                    add_button = gr.Button("Add Support Example", variant="primary")
                
                with gr.Column():
                    status_output = gr.Textbox(label="Status", interactive=False)
                    classes_output = gr.Textbox(label="Current Classes", interactive=False)
                    reset_button = gr.Button("Reset Demo", variant="secondary")
            
            # Handle image clicks for bounding box creation
            support_image.select(
                handle_image_click,
                inputs=[support_image],
                outputs=[support_image, status_output]
            )
            
            # Clear bounding box
            clear_bbox_button.click(
                clear_bbox,
                outputs=[support_image, status_output]
            )
            
            # Add support example
            add_button.click(
                add_support_example_with_bbox,
                inputs=[class_name_input],
                outputs=[status_output, classes_output]
            )
            
            reset_button.click(
                reset_demo,
                outputs=[status_output, classes_output, support_image, class_name_input]
            )
        
        with gr.Tab("Test Detection"):
            gr.Markdown("### Step 2: Test Few-Shot Detection")
            gr.Markdown("Upload a query image to test detection of the classes you defined.")
            
            with gr.Row():
                with gr.Column():
                    query_image = gr.Image(type="pil", label="Query Image")
                    confidence_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                        label="Confidence Threshold"
                    )
                    detect_button = gr.Button("Detect Objects", variant="primary")
                
                with gr.Column():
                    result_image = gr.Image(type="pil", label="Detection Results")
                    results_text = gr.Textbox(label="Detection Summary", interactive=False)
            
            detect_button.click(
                detect_and_visualize,
                inputs=[query_image, confidence_slider],
                outputs=[result_image, results_text]
            )
    
    return interface


def create_sample_data():
    """Create sample images for demonstration."""
    sample_dir = "demo/sample_data"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create simple colored rectangles as sample objects
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    shapes = ["car", "person", "bike", "truck"]
    
    for i, (color, shape) in enumerate(zip(colors, shapes)):
        # Create image with colored rectangle
        img = Image.new('RGB', (300, 300), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw colored rectangle
        x1, y1 = 50 + i * 10, 50 + i * 10
        x2, y2 = x1 + 100, y1 + 80
        draw.rectangle([x1, y1, x2, y2], fill=color, outline='black', width=2)
        
        # Save image
        img.save(f"{sample_dir}/{shape}_sample.png")
    
    # Create sample bounding box annotations
    annotations = {
        "car_sample.png": [60, 60, 160, 140],
        "person_sample.png": [70, 70, 170, 150],
        "bike_sample.png": [80, 80, 180, 160],
        "truck_sample.png": [90, 90, 190, 170]
    }
    
    with open(f"{sample_dir}/sample_boxes.json", 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Sample data created in {sample_dir}")


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="Few-Shot Detection Demo")
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pretrained model')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    parser.add_argument('--create_samples', action='store_true',
                       help='Create sample data for testing')
    parser.add_argument('--interface', type=str, default='gradio',
                       choices=['gradio', 'cli'],
                       help='Interface type')
    
    args = parser.parse_args()
    
    if args.create_samples:
        create_sample_data()
        return
    
    if args.interface == 'gradio':
        # Launch Gradio interface
        interface = create_gradio_interface()
        interface.launch(share=False, server_name="0.0.0.0", server_port=7860)
    
    else:
        # Simple CLI demo
        demo = FewShotDetectionDemo(args.model_path, args.device)
        print("CLI demo mode - implement your own interaction here")
        print("For interactive demo, use --interface gradio")


if __name__ == "__main__":
    main()