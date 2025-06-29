@echo off
echo ========================================
echo CUDA Environment Test Suite
echo ========================================
echo.

echo Step 1: Basic CUDA Test
echo ------------------------
python test_simple_cuda.py
echo.

echo Step 2: Simple Training Test
echo ----------------------------
python main_cuda_simple.py
echo.

echo Step 3: Full Environment Test (Optional)
echo ----------------------------------------
echo Run this if you want comprehensive testing:
echo python test_cuda_environment.py
echo.

echo Step 4: Demo Training (Optional)
echo --------------------------------
echo Run this for interactive demo:
echo python demo_cuda_training.py
echo.

echo Step 5: Full Training (When ready)
echo ----------------------------------
echo Run this for full training:
echo python core/main_cuda.py
echo.

pause