#!/usr/bin/env python3
"""
Test script to verify the DINOv2 backbone implementation works correctly.
This helps debug issues and verify the model can run successfully.
"""

import torch
import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/ubuntu/ct_aerial')

try:
    from models.backbone import DINOv2Backbone, get_image_transforms
    print("✓ Successfully imported backbone modules")
except Exception as e:
    print(f"✗ Failed to import backbone modules: {e}")
    sys.exit(1)

def test_backbone():
    """Test the DINOv2 backbone functionality."""
    print("\n🔧 Testing DINOv2 Backbone...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Create model
        print("Creating DINOv2 backbone...")
        backbone = DINOv2Backbone(model_name="dinov2_vitb14", pretrained=True)
        backbone = backbone.to(device)
        backbone.eval()
        print("✓ Model created successfully")
        
        # Create test input
        batch_size = 2
        height, width = 518, 518
        x = torch.randn(batch_size, 3, height, width).to(device)
        print(f"✓ Created test input: {x.shape}")
        
        # Forward pass
        print("Running forward pass...")
        with torch.no_grad():
            output = backbone(x)
            
        print("✓ Forward pass completed!")
        print(f"Output keys: {list(output.keys())}")
        print(f"Feature shape: {output['features'].shape}")
        print(f"Spatial dimensions: {output['spatial_dims']}")
        print(f"Stride: {output['stride']}")
        
        # Verify output dimensions
        expected_channels = 256  # After projection
        feat_h, feat_w = output['spatial_dims']
        
        if output['features'].shape == (batch_size, expected_channels, feat_h, feat_w):
            print("✓ Output dimensions are correct")
        else:
            print(f"✗ Unexpected output shape: {output['features'].shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error during backbone test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transforms():
    """Test image transforms."""
    print("\n🖼️ Testing Image Transforms...")
    
    try:
        from PIL import Image
        
        # Test transforms
        transform_train = get_image_transforms(is_training=True)
        transform_test = get_image_transforms(is_training=False)
        
        # Create dummy image
        dummy_img = Image.new('RGB', (256, 256), color='red')
        
        # Apply transforms
        img_train = transform_train(dummy_img)
        img_test = transform_test(dummy_img)
        
        print(f"✓ Training transform output: {img_train.shape}")
        print(f"✓ Test transform output: {img_test.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during transform test: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting DINOv2 Backbone Tests")
    print("=" * 50)
    
    success = True
    
    # Test transforms first (lighter test)
    if not test_transforms():
        success = False
    
    # Test backbone
    if not test_backbone():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! The backbone is working correctly.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()