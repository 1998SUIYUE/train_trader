#!/usr/bin/env python3
"""
Check Requirements - Verify all required packages are installed
"""

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True, "OK"
    except ImportError as e:
        return False, str(e)

def main():
    print("=== Requirements Check ===")
    print()
    
    # Required packages
    packages = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("tqdm", "tqdm"),
        ("scipy", "scipy"),
        ("sklearn", "sklearn"),
        ("psutil", "psutil"),
    ]
    
    # Optional packages
    optional_packages = [
        ("nvidia-ml-py3", "pynvml"),
        ("gpustat", "gpustat"),
    ]
    
    print("Required Packages:")
    print("-" * 40)
    all_required_ok = True
    
    for package_name, import_name in packages:
        success, message = check_package(package_name, import_name)
        status = "✓" if success else "✗"
        print(f"{status} {package_name:15} : {message}")
        if not success:
            all_required_ok = False
    
    print()
    print("Optional Packages:")
    print("-" * 40)
    
    for package_name, import_name in optional_packages:
        success, message = check_package(package_name, import_name)
        status = "✓" if success else "✗"
        print(f"{status} {package_name:15} : {message}")
    
    print()
    print("CUDA Check:")
    print("-" * 40)
    
    try:
        import torch
        print(f"✓ PyTorch version    : {torch.__version__}")
        print(f"✓ CUDA available     : {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA version       : {torch.version.cuda}")
            print(f"✓ cuDNN version      : {torch.backends.cudnn.version()}")
            print(f"✓ GPU count          : {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"✓ GPU {i}             : {props.name}")
                print(f"  Memory             : {props.total_memory / 1e9:.1f} GB")
        else:
            print("⚠ CUDA not available - will use CPU")
    
    except Exception as e:
        print(f"✗ CUDA check failed  : {e}")
        all_required_ok = False
    
    print()
    print("=" * 50)
    
    if all_required_ok:
        print("✅ All required packages are installed!")
        print()
        print("Next steps:")
        print("1. Run: python quick_test.py")
        print("2. Run: python main_cuda_simple.py")
    else:
        print("❌ Some required packages are missing!")
        print()
        print("To install missing packages:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("pip install numpy pandas matplotlib seaborn tqdm scipy scikit-learn psutil")

if __name__ == "__main__":
    main()