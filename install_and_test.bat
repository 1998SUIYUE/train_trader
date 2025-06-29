@echo off
echo ========================================
echo CUDA 12.9 Installation and Test Script
echo ========================================
echo.

echo This script will:
echo 1. Install PyTorch with CUDA support
echo 2. Install required dependencies
echo 3. Run basic tests
echo.

set /p continue="Do you want to continue? (y/n): "
if /i not "%continue%"=="y" goto :end

echo.
echo Step 1: Installing PyTorch with CUDA 12.1 support...
echo -----------------------------------------------------
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if %errorlevel% neq 0 (
    echo ERROR: PyTorch installation failed
    goto :end
)

echo.
echo Step 2: Installing other dependencies...
echo ----------------------------------------
pip install numpy pandas matplotlib seaborn tqdm scipy scikit-learn psutil

echo.
echo Step 3: Running quick test...
echo -----------------------------
python quick_test.py

echo.
echo Step 4: Running simple training test...
echo ---------------------------------------
python main_cuda_simple.py

echo.
echo ========================================
echo Installation and testing completed!
echo ========================================
echo.
echo If all tests passed, you can now run:
echo   python core/main_cuda.py
echo.

:end
pause