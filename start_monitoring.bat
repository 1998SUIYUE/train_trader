@echo off
chcp 65001 >nul
title CUDA训练进度监控

echo.
echo ========================================
echo    CUDA训练进度监控工具
echo ========================================
echo.
echo 请选择监控模式:
echo.
echo [1] 查看当前训练状态
echo [2] 实时监控训练进度
echo [3] 查看最近20代历史
echo [4] 查看完整训练历史
echo [5] 退出
echo.

:menu
set /p choice="请输入选择 (1-5): "

if "%choice%"=="1" (
    echo.
    echo 正在查看当前训练状态...
    python cuda_progress_monitor.py
    echo.
    pause
    goto menu
) else if "%choice%"=="2" (
    echo.
    echo 开始实时监控...
    echo 按 Ctrl+C 停止监控
    python cuda_progress_monitor.py --watch
    echo.
    pause
    goto menu
) else if "%choice%"=="3" (
    echo.
    echo 查看最近20代历史...
    python cuda_progress_monitor.py --tail 20
    echo.
    pause
    goto menu
) else if "%choice%"=="4" (
    echo.
    echo 查看完整训练历史...
    python cuda_progress_monitor.py
    echo.
    pause
    goto menu
) else if "%choice%"=="5" (
    echo 再见!
    exit /b 0
) else (
    echo 无效选择，请重新输入
    goto menu
)