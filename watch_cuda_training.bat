@echo off
chcp 65001 >nul
echo 🚀 CUDA训练进度监控工具
echo.
echo 选择监控模式:
echo 1. 查看当前进度 (一次性)
echo 2. 实时监控 (持续更新)
echo 3. 查看最近20代
echo 4. 查看训练历史图表
echo.
set /p choice="请选择 (1-4): "

if "%choice%"=="1" (
    python view_cuda_progress.py
) else if "%choice%"=="2" (
    python view_cuda_progress.py --watch
) else if "%choice%"=="3" (
    python view_cuda_progress.py --tail 20
) else if "%choice%"=="4" (
    python view_cuda_progress.py --tail 50
) else (
    echo 无效选择，默认显示当前进度
    python view_cuda_progress.py
)

echo.
pause