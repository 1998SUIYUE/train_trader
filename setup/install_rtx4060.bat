@echo off
echo ========================================
echo RTX 4060 训练环境一键安装脚本
echo ========================================

echo.
echo 检查Python环境...
python --version
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8或更高版本
    pause
    exit /b 1
)

echo.
echo 升级pip...
python -m pip install --upgrade pip

echo.
echo 安装CUDA版本的PyTorch...
echo 正在安装适用于RTX 4060的PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo 安装其他依赖包...
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install tqdm>=4.62.0
pip install scipy>=1.7.0
pip install scikit-learn>=1.0.0

echo.
echo 验证安装...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 下一步:
echo 1. 测试环境: python setup/test_rtx4060.py
echo 2. 开始训练: python core/main_cuda.py
echo.
pause