@echo off
chcp 65001 >nul
title CUDA训练进度监控

echo 🚀 启动CUDA训练进度监控...
echo ================================

REM 尝试启动最佳的监控工具
if exist "tools\watch_training_progress.py" (
    echo 🎯 使用完整版监控工具...
    python tools\watch_training_progress.py
) else if exist "view_cuda_progress.py" (
    echo 🎯 使用CUDA进度查看器...
    python view_cuda_progress.py --watch
) else if exist "quick_monitor.py" (
    echo 🎯 使用快速监控器...
    python quick_monitor.py
) else (
    echo ❌ 未找到监控工具
    echo 请确保以下文件之一存在:
    echo   - tools\watch_training_progress.py
    echo   - view_cuda_progress.py
    echo   - quick_monitor.py
    pause
    exit /b 1
)

echo.
echo 👋 监控已结束
pause