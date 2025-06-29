@echo off
echo Testing CUDA Environment...
echo.

echo === Basic PyTorch Test ===
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo.

echo === Running Simple CUDA Test ===
python test_simple_cuda.py
echo.

echo === Running Simple Training Demo ===
echo Do you want to run the training demo? (y/n)
set /p choice=
if /i "%choice%"=="y" (
    python main_cuda_simple.py
) else (
    echo Training demo skipped.
)

echo.
echo Test completed. Press any key to exit...
pause >nul