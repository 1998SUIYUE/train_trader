#!/usr/bin/env python3
"""
Quick CUDA Test - Minimal test to verify CUDA is working
"""

def main():
    print("=== Quick CUDA Test ===")
    
    try:
        import torch
        print(f"✓ PyTorch imported successfully: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            
            # Test basic GPU operation
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            y = x * 2
            result = y.cpu().numpy()
            print(f"✓ Basic GPU operation test: {result}")
            
            # Show GPU info
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"✓ GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
            
            print("\n🎉 CUDA test PASSED! Your environment is ready.")
            return True
            
        else:
            print("❌ CUDA is not available")
            print("   Please check CUDA installation and PyTorch version")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please install PyTorch with CUDA support")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nNext step: Run 'python main_cuda_simple.py' for training test")
    else:
        print("\nPlease fix the issues above before proceeding")