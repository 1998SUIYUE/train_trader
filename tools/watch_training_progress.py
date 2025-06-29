#!/usr/bin/env python3
"""
实时训练进度监控工具
监控正在进行的训练过程，实时显示进度和统计信息
"""

import json
import time
import argparse
from pathlib import Path
import os
import sys

# 尝试导入可选依赖
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_latest_data(log_file):
    """加载最新的训练数据"""
    data = []
    try:
        if not Path(log_file).exists():
            return data
            
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return data
    except Exception as e:
        print(f"加载日志文件失败: {e}")
        return []

def display_current_status(data):
    """显示当前训练状态"""
    if not data:
        print("📊 等待训练数据...")
        return
    
    latest = data[-1]
    
    # 基本信息
    print("=" * 80)
    print("🚀 CUDA遗传算法训练实时监控")
    print("=" * 80)
    
    # 当前状态
    print(f"📈 当前代数: {latest.get('generation', 0)}")
    print(f"🏆 最佳适应度: {latest.get('best_fitness', 0):.6f}")
    print(f"📊 平均适应度: {latest.get('avg_fitness', 0):.6f}")
    print(f"📉 标准差: {latest.get('std_fitness', 0):.6f}")
    print(f"⏱️  本代用时: {latest.get('generation_time', 0):.2f}秒")
    print(f"🔄 无改进代数: {latest.get('no_improvement_count', 0)}")
    
    # 内存使用情况
    if 'gpu_memory_allocated' in latest:
        print(f"🖥️  GPU内存: {latest['gpu_memory_allocated']:.2f}GB")
    if 'system_memory_gb' in latest:
        print(f"💾 系统内存: {latest['system_memory_gb']:.2f}GB")
    
    # 训练统计
    if len(data) > 1:
        total_time = sum(d.get('generation_time', 0) for d in data)
        avg_time = total_time / len(data)
        best_ever = max(d.get('best_fitness', 0) for d in data)
        
        print(f"\n📊 训练统计:")
        print(f"   总代数: {len(data)}")
        print(f"   历史最佳: {best_ever:.6f}")
        print(f"   总训练时间: {total_time/3600:.2f}小时")
        print(f"   平均每代: {avg_time:.2f}秒")
        
        # 改进趋势
        recent_10 = data[-10:] if len(data) >= 10 else data
        recent_best = [d.get('best_fitness', 0) for d in recent_10]
        if len(recent_best) > 1:
            trend = "📈 上升" if recent_best[-1] > recent_best[0] else "📉 下降"
            print(f"   最近趋势: {trend}")
    
    print("=" * 80)

def display_progress_chart(data, max_points=50):
    """显示简单的ASCII进度图表"""
    if len(data) < 2:
        return
    
    print("\n📈 适应度趋势图 (最近50代):")
    
    # 获取最近的数据点
    recent_data = data[-max_points:] if len(data) > max_points else data
    fitness_values = [d.get('best_fitness', 0) for d in recent_data]
    
    if not fitness_values:
        return
    
    # 归一化到0-20的范围用于ASCII图表
    min_val = min(fitness_values)
    max_val = max(fitness_values)
    
    if max_val == min_val:
        normalized = [10] * len(fitness_values)
    else:
        normalized = [int((val - min_val) / (max_val - min_val) * 20) for val in fitness_values]
    
    # 绘制ASCII图表
    for row in range(20, -1, -1):
        line = f"{max_val - (max_val - min_val) * (20 - row) / 20:8.4f} |"
        for val in normalized:
            if val >= row:
                line += "█"
            else:
                line += " "
        print(line)
    
    # 底部标尺
    print(" " * 10 + "+" + "-" * len(normalized))
    print(f"         最小值: {min_val:.4f}, 最大值: {max_val:.4f}")

def watch_training(log_file, refresh_interval=2.0, show_chart=True):
    """监控训练进度"""
    print(f"🔍 开始监控训练日志: {log_file}")
    print(f"🔄 刷新间隔: {refresh_interval}秒")
    print("按 Ctrl+C 停止监控\n")
    
    last_size = 0
    
    try:
        while True:
            # 检查文件是否有更新
            if Path(log_file).exists():
                current_size = Path(log_file).stat().st_size
                if current_size != last_size:
                    # 文件有更新，重新加载数据
                    data = load_latest_data(log_file)
                    
                    clear_screen()
                    display_current_status(data)
                    
                    if show_chart and PLOTTING_AVAILABLE:
                        display_progress_chart(data)
                    
                    last_size = current_size
                    print(f"\n⏰ 最后更新: {time.strftime('%H:%M:%S')}")
                    print("按 Ctrl+C 停止监控")
                else:
                    # 文件没有更新，显示等待状态
                    print(f"\r⏳ 等待新数据... {time.strftime('%H:%M:%S')}", end="", flush=True)
            else:
                print(f"\r📁 等待日志文件创建... {time.strftime('%H:%M:%S')}", end="", flush=True)
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")

def main():
    parser = argparse.ArgumentParser(description='实时训练进度监控工具')
    parser.add_argument('log_file', nargs='?', help='日志文件路径')
    parser.add_argument('--interval', '-i', type=float, default=2.0, help='刷新间隔(秒)')
    parser.add_argument('--no-chart', action='store_true', help='不显示ASCII图表')
    parser.add_argument('--auto', action='store_true', help='自动查找最新的训练日志文件')
    
    args = parser.parse_args()
    
    # 如果没有指定文件且启用了auto模式，自动查找最新日志
    if not args.log_file and args.auto:
        # 查找可能的日志文件位置
        possible_paths = [
            Path("../results/training_history_cuda.jsonl"),
            Path("results/training_history_cuda.jsonl"),
            Path("training_history_cuda.jsonl"),
            Path("../results/training_history.jsonl"),
            Path("results/training_history.jsonl"),
            Path("training_history.jsonl")
        ]
        
        log_file = None
        for path in possible_paths:
            if path.exists():
                log_file = path
                break
        
        if log_file:
            args.log_file = str(log_file)
            print(f"🔍 自动发现日志文件: {args.log_file}")
        else:
            print("❌ 未找到训练日志文件")
            print("请指定日志文件路径，或确保在正确的目录中运行")
            print("用法: python watch_training_progress.py [日志文件路径]")
            print("或者: python watch_training_progress.py --auto")
            return
    elif not args.log_file:
        # 如果没有指定文件也没有auto模式，显示帮助
        print("实时训练进度监控工具")
        print("用法:")
        print("  python watch_training_progress.py <日志文件路径>")
        print("  python watch_training_progress.py --auto  # 自动查找最新日志")
        print("")
        print("选项:")
        print("  --interval, -i  刷新间隔(秒，默认2.0)")
        print("  --no-chart      不显示ASCII图表")
        print("  --auto          自动查找最新的训练日志文件")
        return
    
    # 开始监控
    watch_training(
        args.log_file, 
        refresh_interval=args.interval,
        show_chart=not args.no_chart
    )

if __name__ == "__main__":
    main()