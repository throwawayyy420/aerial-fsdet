"""
Package initialization for ct_aerial few-shot object detection.
"""

__version__ = "0.1.0"
__author__ = "GitHub Copilot"
__description__ = "Few-shot object detection using DINOv2 backbone with two-stage training"

# Import main components
from .models.backbone import DINOv2Backbone, FeatureExtractor
from .models.rpn import RPN, ROIPooling
from .models.detection_head import DetectionHead, FewShotClassifier
from .training.stage1_base_training import FewShotDetector

__all__ = [
    'DINOv2Backbone',
    'FeatureExtractor', 
    'RPN',
    'ROIPooling',
    'DetectionHead',
    'FewShotClassifier',
    'FewShotDetector'
]