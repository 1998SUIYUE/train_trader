#!/bin/bash

echo "========================================"
echo "RTX 4060 训练环境一键安装脚本"
echo "========================================"

# 检查Python
echo ""
echo "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python 3.8或更高版本"
    exit 1
fi

python3 --version

# 升级pip
echo ""
echo "升级pip..."
python3 -m pip install --upgrade pip

# 安装PyTorch (CUDA版本)
echo ""
echo "安装CUDA版本的PyTorch..."
echo "正在安装适用于RTX 4060的PyTorch..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
echo ""
echo "安装其他依赖包..."
pip3 install numpy>=1.21.0
pip3 install pandas>=1.3.0
pip3 install matplotlib>=3.5.0
pip3 install seaborn>=0.11.0
pip3 install tqdm>=4.62.0
pip3 install scipy>=1.7.0
pip3 install scikit-learn>=1.0.0

# 验证安装
echo ""
echo "验证安装..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

echo ""
echo "========================================"
echo "安装完成！"
echo "========================================"
echo ""
echo "下一步:"
echo "1. 测试环境: python3 setup/test_rtx4060.py"
echo "2. 开始训练: python3 core/main_cuda.py"
echo ""