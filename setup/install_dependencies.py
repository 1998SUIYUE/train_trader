#!/usr/bin/env python3
"""
依赖安装脚本
自动检测环境并安装合适的依赖包
"""

import subprocess
import sys
import platform
import torch

def run_command(command):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_cuda_availability():
    """检查CUDA是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def detect_cuda_version():
    """检测CUDA版本"""
    success, stdout, stderr = run_command("nvidia-smi")
    if success and "CUDA Version" in stdout:
        # 从nvidia-smi输出中提取CUDA版本
        for line in stdout.split('\n'):
            if "CUDA Version" in line:
                version = line.split("CUDA Version: ")[1].split()[0]
                return version
    return None

def install_pytorch_cuda():
    """安装支持CUDA的PyTorch"""
    print("🔍 检测CUDA环境...")
    
    cuda_version = detect_cuda_version()
    if cuda_version:
        print(f"✅ 检测到CUDA版本: {cuda_version}")
        
        # 根据CUDA版本选择合适的PyTorch
        if cuda_version.startswith("12."):
            index_url = "https://download.pytorch.org/whl/cu121"
            print("📦 安装CUDA 12.1版本的PyTorch...")
        elif cuda_version.startswith("11.8"):
            index_url = "https://download.pytorch.org/whl/cu118"
            print("📦 安装CUDA 11.8版本的PyTorch...")
        else:
            index_url = "https://download.pytorch.org/whl/cu118"
            print("📦 使用默认CUDA 11.8版本的PyTorch...")
        
        # 安装PyTorch
        command = f"pip install torch torchvision torchaudio --index-url {index_url}"
        print(f"执行命令: {command}")
        
        success, stdout, stderr = run_command(command)
        if success:
            print("✅ PyTorch安装成功")
            return True
        else:
            print(f"❌ PyTorch安装失败: {stderr}")
            return False
    else:
        print("⚠️  未检测到CUDA，安装CPU版本的PyTorch...")
        success, stdout, stderr = run_command("pip install torch torchvision torchaudio")
        if success:
            print("✅ PyTorch (CPU版本) 安装成功")
            return True
        else:
            print(f"❌ PyTorch安装失败: {stderr}")
            return False

def install_other_dependencies():
    """安装其他依赖"""
    print("📦 安装其他依赖包...")
    
    dependencies = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0"
    ]
    
    for dep in dependencies:
        print(f"安装 {dep}...")
        success, stdout, stderr = run_command(f"pip install {dep}")
        if success:
            print(f"✅ {dep} 安装成功")
        else:
            print(f"❌ {dep} 安装失败: {stderr}")
            return False
    
    return True

def verify_installation():
    """验证安装"""
    print("\n🔍 验证安装...")
    
    # 检查PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} 安装成功")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA可用，设备: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA不可用，将使用CPU")
    except ImportError:
        print("❌ PyTorch导入失败")
        return False
    
    # 检查其他包
    packages = ['numpy', 'pandas', 'matplotlib', 'tqdm', 'scipy', 'sklearn']
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} 可用")
        except ImportError:
            print(f"❌ {package} 导入失败")
            return False
    
    return True

def main():
    """主安装函数"""
    print("🚀 开始安装训练环境依赖")
    print("=" * 50)
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ 需要Python 3.8或更高版本")
        return
    
    print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    print(f"✅ 操作系统: {platform.system()} {platform.release()}")
    
    # 升级pip
    print("\n📦 升级pip...")
    run_command("python -m pip install --upgrade pip")
    
    # 安装PyTorch
    print("\n" + "=" * 50)
    if not install_pytorch_cuda():
        print("❌ PyTorch安装失败，请手动安装")
        return
    
    # 安装其他依赖
    print("\n" + "=" * 50)
    if not install_other_dependencies():
        print("❌ 依赖安装失败")
        return
    
    # 验证安装
    print("\n" + "=" * 50)
    if verify_installation():
        print("\n🎉 所有依赖安装成功！")
        print("\n下一步:")
        print("1. 测试环境: python setup/test_rtx4060.py")
        print("2. 开始训练: python core/main_cuda.py")
    else:
        print("\n❌ 安装验证失败，请检查错误信息")

if __name__ == "__main__":
    main()