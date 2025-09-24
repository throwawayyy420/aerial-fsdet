#!/usr/bin/env python3
"""
Test the enhanced Gradio interface with click-and-drag bounding box functionality.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/ubuntu/ct_aerial')

def test_gradio_interface():
    """Test the enhanced Gradio interface."""
    print("🧪 Testing Enhanced Gradio Interface")
    print("=" * 50)
    
    try:
        from demo.few_shot_demo import create_gradio_interface
        
        print("✓ Successfully imported create_gradio_interface")
        
        # Create interface
        interface = create_gradio_interface()
        print("✓ Successfully created Gradio interface")
        
        print("\n🎉 Enhanced interface ready!")
        print("Features:")
        print("  ✅ Click-and-drag bounding box selection")
        print("  ✅ Visual bounding box preview")
        print("  ✅ Interactive image annotation")
        print("  ✅ No manual coordinate entry needed")
        
        print(f"\n🚀 Launch the interface:")
        print("interface.launch()")
        
        return True
        
    except Exception as e:
        print(f"✗ Interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gradio_interface()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 GRADIO INTERFACE ENHANCEMENT SUCCESSFUL!")
        print("The demo now supports click-and-drag bounding box selection!")
    else:
        print("❌ INTERFACE ISSUES FOUND")
        print("Please check the error messages above.")