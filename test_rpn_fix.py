#!/usr/bin/env python3
"""
Test the RPN clip_boxes fix to ensure no more inplace operations.
"""

import torch
import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/ubuntu/ct_aerial')

def test_rpn_clip_boxes():
    """Test that the RPN clip_boxes method doesn't cause gradient issues."""
    print("🧪 Testing RPN Clip Boxes Fix")
    print("=" * 40)
    
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Import RPN
    from models.rpn import RPN
    
    # Create RPN
    rpn = RPN()
    
    # Create test boxes that require gradients
    boxes = torch.tensor([
        [-10, -10, 100, 100],
        [50, 50, 600, 400],
        [200, 200, 250, 250]
    ], dtype=torch.float32, requires_grad=True)
    
    image_size = (512, 512)
    
    print(f"Input boxes shape: {boxes.shape}")
    print(f"Input boxes require_grad: {boxes.requires_grad}")
    
    try:
        # Test the clip operation
        clipped_boxes = rpn._clip_boxes(boxes, image_size)
        
        print(f"Clipped boxes shape: {clipped_boxes.shape}")
        print(f"Clipped boxes require_grad: {clipped_boxes.requires_grad}")
        
        # Test backward pass
        loss = clipped_boxes.sum()
        loss.backward()
        
        print("✓ Clip boxes operation completed successfully!")
        print("✓ Gradients computed without errors!")
        print(f"✓ Input gradient shape: {boxes.grad.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Clip boxes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_rpn():
    """Test full RPN forward pass."""
    print("\n🔄 Testing Full RPN Forward Pass")
    print("=" * 40)
    
    from models.rpn import RPN
    
    rpn = RPN()
    
    # Create test features
    features = torch.randn(2, 256, 32, 32, requires_grad=True)
    image_size = (512, 512)
    stride = 16
    
    try:
        output = rpn(features, image_size, stride)
        
        print("✓ RPN forward pass completed!")
        print(f"✓ Proposals generated: {[len(p) for p in output['proposals']]}")
        
        # Test backward through proposals
        if len(output['proposals'][0]) > 0:
            total_loss = output['objectness_raw'].sum() + output['bbox_deltas'].sum()
            total_loss.backward()
            print("✓ RPN gradients computed successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Full RPN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_rpn_clip_boxes()
    success2 = test_full_rpn()
    
    print("\n" + "=" * 40)
    if success1 and success2:
        print("🎉 RPN INPLACE OPERATION FIX SUCCESSFUL!")
        print("The ClampBackward1 error has been resolved.")
    else:
        print("❌ RPN ISSUES REMAIN")
        print("Further debugging needed.")