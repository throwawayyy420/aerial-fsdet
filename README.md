# Few-Shot Object Detection with DINOv2

This repository implements a two-stage few-shot object detection system using DINOv2 as the backbone, following the approach described in the provided architecture diagram. This repo is based on the ICML 2020 paper Frustratingly Simple Few-Shot Object Detection (arXiv:2003.06957).

## Architecture Overview

The system implements a two-stage training approach:

### Stage I: Base Training
- **Backbone**: DINOv2 vision transformer for feature extraction
- **RPN**: Region Proposal Network for generating object proposals
- **ROI Pooling**: Extract fixed-size features from variable-size regions
- **Detection Heads**: Box classifier and regressor for base classes
- **Training**: End-to-end training on abundant base class data

### Stage II: Few-Shot Fine-Tuning
- **Fixed Feature Extractor**: Frozen DINOv2 backbone and early layers
- **Trainable Heads**: Only classification and regression heads are updated
- **Prototypical Learning**: Novel classes represented by prototypes
- **Training**: Few-shot episodes with support and query examples

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ct_aerial
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
ct_aerial/
├── models/
│   ├── backbone.py              # DINOv2 backbone implementation
│   ├── rpn.py                   # Region Proposal Network
│   └── detection_head.py        # Classification and regression heads
├── training/
│   ├── stage1_base_training.py  # Stage I training script
│   └── stage2_few_shot_finetuning.py  # Stage II fine-tuning script
├── demo/
│   └── few_shot_demo.py         # Interactive demo interface
├── data/                        # Data directory (create your own)
├── utils/                       # Utility functions
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Quick Start

### Local Training

#### 1. Prepare Your Data

The system expects data in COCO format. Create the following directory structure:

```
data/
├── train/                       # Base class training images
├── novel/                       # Novel class images for few-shot learning
└── annotations/
    ├── train.json              # Base class annotations
    └── novel.json              # Novel class annotations
```

#### 2. Stage I: Base Training

Train the complete model on abundant base class data:

```bash
python training/stage1_base_training.py \
    --num_classes 20 \
    --train_data_dir data/train \
    --train_annotations data/annotations/train.json \
    --batch_size 4 \
    --epochs 50 \
    --learning_rate 1e-4 \
    --save_dir checkpoints/stage1
```

#### 3. Stage II: Few-Shot Fine-Tuning

Fine-tune the model for novel classes with few examples:

```bash
python training/stage2_few_shot_finetuning.py \
    --stage1_model_path checkpoints/stage1/stage1_final_model.pth \
    --novel_data_dir data/novel \
    --novel_annotations data/annotations/novel.json \
    --n_way 5 \
    --k_shot 5 \
    --n_query 15 \
    --learning_rate 1e-5 \
    --save_dir checkpoints/stage2
```

###  Enhanced Interactive Demo

Launch the enhanced demo with **click-and-drag bounding box selection**:

```bash
# Quick launch
python launch_demo.py

# Or through pipeline
python run_pipeline.py --step demo
```

Access at `http://localhost:7860` to:
- Add support examples by clicking on images
- Test detection on query images  
- Visualize results with confidence scores

## Training Parameters

### Stage I Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_classes` | Number of base classes | 20 |
| `--backbone_model` | DINOv2 model variant | dinov2_vitb14 |
| `--batch_size` | Batch size for training | 4 |
| `--epochs` | Number of training epochs | 50 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--weight_decay` | Weight decay | 1e-4 |

### Stage II Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n_way` | Number of classes per episode | 5 |
| `--k_shot` | Support examples per class | 5 |
| `--n_query` | Query examples per class | 15 |
| `--learning_rate` | Fine-tuning learning rate | 1e-5 |

## Model Components

### DINOv2 Backbone (`models/backbone.py`)
- Pretrained DINOv2 vision transformer
- Feature extraction with spatial preservation
- Configurable model variants (ViT-S/B/L/G)

### Region Proposal Network (`models/rpn.py`)
- Anchor generation at multiple scales
- Objectness scoring and box regression
- Non-maximum suppression for proposal filtering

### Detection Heads (`models/detection_head.py`)
- Box classification with cross-entropy loss
- Bounding box regression with smooth L1 loss
- Few-shot classifier using prototypical networks

## Data Format

The system expects COCO-format annotations:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height]
    }
  ]
}
```

## Creating Sample Data

To test the system without real data, create sample images:

```bash
python demo/few_shot_demo.py --create_samples
```

This creates simple colored rectangles in `demo/sample_data/` for testing.

## Advanced Usage

### Custom Backbone

To use a different DINOv2 variant:

```python
from models.backbone import DINOv2Backbone

backbone = DINOv2Backbone(model_name="dinov2_vitl14")  # Large variant
```

### Custom Loss Functions

Implement custom losses in `models/detection_head.py`:

```python
def custom_few_shot_loss(predictions, targets, alpha=0.5):
    cls_loss = F.cross_entropy(predictions["cls_scores"], targets["labels"])
    # Add your custom loss terms
    return cls_loss
```

### Evaluation

Evaluate few-shot performance:

```python
from training.stage2_few_shot_finetuning import evaluate_few_shot

accuracy = evaluate_few_shot(model, test_dataloader, device)
print(f"Few-shot accuracy: {accuracy:.4f}")
```