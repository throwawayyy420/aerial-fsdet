#!/usr/bin/env python3
"""
Comprehensive test for patch grid handling in DINOv2 backbone.
Tests various problematic patch counts to ensure robust handling.
"""

import torch
import torch.nn as nn
import sys
import os
import math

# Add the project root to Python path
sys.path.insert(0, '/home/ubuntu/ct_aerial')

class MockDINOv2(nn.Module):
    """Mock DINOv2 model for testing various patch scenarios."""
    
    def __init__(self, feature_dim=768, num_patches=1368):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_patches = num_patches
    
    def forward_features(self, x):
        """Simulate DINOv2 forward_features output."""
        batch_size = x.shape[0]
        return torch.randn(batch_size, self.num_patches, self.feature_dim)

# Patch torch.hub.load
original_hub_load = torch.hub.load
current_mock = None

def mock_hub_load(repo, model_name, *args, **kwargs):
    if repo == 'facebookresearch/dinov2':
        return current_mock
    return original_hub_load(repo, model_name, *args, **kwargs)

torch.hub.load = mock_hub_load

try:
    from models.backbone import DINOv2Backbone
    print("✓ Successfully imported backbone")
    
    # Test cases: various problematic patch counts
    test_cases = [
        (1368, "Common problematic case: 1368 patches"),
        (1369, "Expected case: 37×37 = 1369 patches"), 
        (1296, "Perfect square: 36×36 = 1296 patches"),
        (1200, "Rectangular: 40×30 = 1200 patches"),
        (1367, "Prime-like number: 1367 patches"),
        (1000, "Round number: 1000 patches"),
        (576, "Perfect square: 24×24 = 576 patches")
    ]
    
    print("\n🧪 Testing various patch grid scenarios:")
    print("=" * 60)
    
    for num_patches, description in test_cases:
        print(f"\n📊 {description}")
        print("-" * 40)
        
        # Set up mock for this test
        current_mock = MockDINOv2(num_patches=num_patches)
        
        try:
            # Create backbone
            backbone = DINOv2Backbone(model_name="dinov2_vitb14", pretrained=True)
            backbone.eval()
            
            # Test input
            x = torch.randn(1, 3, 518, 518)
            
            # Forward pass
            with torch.no_grad():
                output = backbone(x)
            
            feat_shape = output['features'].shape
            spatial_dims = output['spatial_dims']
            
            print(f"✓ Success: {feat_shape} -> {spatial_dims[0]}×{spatial_dims[1]} grid")
            
            # Check if the grid makes sense
            grid_patches = spatial_dims[0] * spatial_dims[1]
            if grid_patches == num_patches:
                print(f"  Perfect match: {grid_patches} patches")
            else:
                diff = abs(grid_patches - num_patches)
                print(f"  Adjusted: {num_patches} → {grid_patches} patches (diff: {diff})")
            
            # Check aspect ratio
            aspect_ratio = max(spatial_dims) / min(spatial_dims)
            if aspect_ratio <= 2.0:
                print(f"  Good aspect ratio: {aspect_ratio:.2f}")
            else:
                print(f"  ⚠️  High aspect ratio: {aspect_ratio:.2f}")
                
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Comprehensive patch grid testing completed!")
    print("The backbone should now handle various DINOv2 output scenarios robustly.")
    
except Exception as e:
    print(f"✗ Import or setup error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)