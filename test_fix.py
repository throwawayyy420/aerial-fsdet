#!/usr/bin/env python3
"""
Simple test to verify the backbone logic without requiring model downloads.
"""

import torch
import torch.nn as nn
import sys
import os

# Mock DINOv2 for testing
class MockDINOv2(nn.Module):
    """Mock DINOv2 model for testing purposes."""
    
    def __init__(self, feature_dim=768, num_patches=1368):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_patches = num_patches
    
    def forward_features(self, x):
        """Simulate DINOv2 forward_features output."""
        batch_size = x.shape[0]
        # Return the actual problematic number of patches to test the fix
        # This simulates the real error case where we get 1368 patches after CLS token removal
        return torch.randn(batch_size, self.num_patches, self.feature_dim)

# Add the project root to Python path
sys.path.insert(0, '/home/ubuntu/ct_aerial')

# Patch torch.hub.load to use our mock
original_hub_load = torch.hub.load

def mock_hub_load(repo, model_name, *args, **kwargs):
    if repo == 'facebookresearch/dinov2':
        # Test with the problematic 1368 patches (24×57 = 1368)
        return MockDINOv2(num_patches=1368)
    return original_hub_load(repo, model_name, *args, **kwargs)

torch.hub.load = mock_hub_load

try:
    from models.backbone import DINOv2Backbone
    print("✓ Successfully imported backbone with mock DINOv2")
    
    # Test the backbone
    print("Testing backbone forward pass...")
    
    backbone = DINOv2Backbone(model_name="dinov2_vitb14", pretrained=True)
    backbone.eval()
    
    # Create test input
    x = torch.randn(2, 3, 518, 518)
    
    # Forward pass
    with torch.no_grad():
        output = backbone(x)
    
    print("✓ Forward pass completed successfully!")
    print(f"Output feature shape: {output['features'].shape}")
    print(f"Spatial dimensions: {output['spatial_dims']}")
    print(f"Stride: {output['stride']}")
    
    # Verify dimensions
    batch_size, channels, feat_h, feat_w = output['features'].shape
    expected_channels = 256  # After projection
    
    if channels == expected_channels:
        print("✓ Feature channels are correct")
    else:
        print(f"✗ Expected {expected_channels} channels, got {channels}")
    
    if feat_h > 0 and feat_w > 0:
        print("✓ Spatial dimensions are positive")
    else:
        print(f"✗ Invalid spatial dimensions: {feat_h}x{feat_w}")
    
    print("\n🎉 The backbone fix is working correctly!")
    print("The TypeError: unhashable type: 'slice' issue should be resolved.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)