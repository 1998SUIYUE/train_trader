@echo off
chcp 65001 >nul
echo 🚀 CUDA训练监控启动器
echo ================================

echo.
echo 选择监控模式:
echo 1. 动态图表监控 (推荐)
echo 2. 简单文本监控
echo 3. 快速监控
echo 4. 查看现有日志
echo.

set /p choice="请选择 (1-4): "

if "%choice%"=="1" (
    echo.
    echo 🎯 启动动态图表监控...
    python real_time_training_dashboard.py --auto
) else if "%choice%"=="2" (
    echo.
    echo 📝 启动文本监控...
    python real_time_training_dashboard.py --auto --text-mode
) else if "%choice%"=="3" (
    echo.
    echo ⚡ 启动快速监控...
    python quick_monitor.py
) else if "%choice%"=="4" (
    echo.
    echo 📋 查看现有日志...
    python tools/view_training_log.py --auto --plot
) else (
    echo.
    echo ❌ 无效选择，启动默认监控...
    python quick_monitor.py
)

echo.
pause