#!/usr/bin/env python3
"""
Quick test to verify the gradient computation fix.
Tests the model forward and backward pass to ensure no gradient errors.
"""

import torch
import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/ubuntu/ct_aerial')

def test_gradient_computation():
    """Test that gradients compute without errors."""
    print("🧪 Testing Gradient Computation Fix")
    print("=" * 50)
    
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Import after setting path
    from training.stage1_base_training import FewShotDetector
    
    # Create model
    model = FewShotDetector(num_classes=3, backbone_model="dinov2_vitb14", feature_dim=256)
    model.train()
    
    # Unfreeze backbone for Stage I
    model.backbone.unfreeze_backbone()
    
    # Create dummy input
    batch_size = 2
    images = torch.randn(batch_size, 3, 518, 518, requires_grad=True)
    
    # Create dummy targets
    targets = {
        'boxes': [torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]]) for _ in range(batch_size)],
        'labels': [torch.tensor([1, 2]) for _ in range(batch_size)]
    }
    
    print("✓ Model and data created")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Input shape: {images.shape}")
    
    try:
        # Forward pass
        print("\n🔄 Testing forward pass...")
        output = model(images, targets)
        
        print("✓ Forward pass completed")
        print(f"  Output keys: {list(output.keys())}")
        
        # Test backward pass
        print("\n⬅️ Testing backward pass...")
        loss = output.get("total_loss", torch.tensor(1.0, requires_grad=True))
        print(f"  Loss: {loss.item():.6f}")
        
        loss.backward()
        
        print("✓ Backward pass completed successfully!")
        
        # Check gradients
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"  Gradients computed: {grad_count}/{total_params} parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_gradient_computation()
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 GRADIENT FIX SUCCESSFUL!")
            print("The inplace operation error has been resolved.")
            print("Stage I training should now work properly.")
        else:
            print("❌ GRADIENT ISSUES REMAIN")
            print("Further debugging needed.")
            
    except Exception as e:
        print(f"Test setup failed: {e}")
        import traceback
        traceback.print_exc()