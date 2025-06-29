#!/usr/bin/env python3
"""
CUDA训练进度查看器
实时显示CUDA训练的进度和统计信息
"""

import json
import time
import os
from pathlib import Path
import argparse

def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_training_data(log_file):
    """加载训练数据"""
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

def display_progress(data):
    """显示训练进度"""
    if not data:
        print("📊 没有训练数据")
        return
    
    latest = data[-1]
    
    print("=" * 80)
    print("🚀 CUDA遗传算法训练进度监控")
    print("=" * 80)
    
    # 基本信息
    print(f"📈 当前代数: {latest.get('generation', 0)}")
    print(f"🏆 最佳适应度: {latest.get('best_fitness', 0):.6f}")
    print(f"📊 平均适应度: {latest.get('mean_fitness', 0):.6f}")
    print(f"📉 标准差: {latest.get('std_fitness', 0):.6f}")
    print(f"⏱️  本代用时: {latest.get('generation_time', 0):.2f}秒")
    
    # 交易指标
    if 'mean_sharpe_ratio' in latest:
        print(f"📈 夏普比率: {latest['mean_sharpe_ratio']:.6f}")
    if 'mean_sortino_ratio' in latest:
        print(f"📊 索提诺比率: {latest['mean_sortino_ratio']:.6f}")
    if 'mean_max_drawdown' in latest:
        print(f"📉 最大回撤: {latest['mean_max_drawdown']:.6f}")
    if 'mean_overall_return' in latest:
        print(f"💰 总回报: {latest['mean_overall_return']:.6f}")
    
    # 系统信息
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
        
        # 最近趋势
        recent_10 = data[-10:] if len(data) >= 10 else data
        recent_best = [d.get('best_fitness', 0) for d in recent_10]
        if len(recent_best) > 1:
            trend = "📈 上升" if recent_best[-1] > recent_best[0] else "📉 下降"
            print(f"   最近趋势: {trend}")
    
    print("=" * 80)

def display_fitness_chart(data, max_points=50):
    """显示适应度趋势图"""
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

def watch_training(log_file, refresh_interval=3.0, show_chart=True):
    """实时监控训练进度"""
    print(f"🔍 开始监控CUDA训练日志: {log_file}")
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
                    data = load_training_data(log_file)
                    
                    clear_screen()
                    display_progress(data)
                    
                    if show_chart:
                        display_fitness_chart(data)
                    
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
    parser = argparse.ArgumentParser(description='CUDA训练进度查看器')
    parser.add_argument('log_file', nargs='?', help='日志文件路径')
    parser.add_argument('--watch', '-w', action='store_true', help='实时监控模式')
    parser.add_argument('--interval', '-i', type=float, default=3.0, help='刷新间隔(秒)')
    parser.add_argument('--no-chart', action='store_true', help='不显示图表')
    parser.add_argument('--tail', type=int, help='只显示最后N条记录')
    
    args = parser.parse_args()
    
    # 如果没有指定文件，自动查找
    if not args.log_file:
        possible_paths = [
            Path("results/training_history_cuda.jsonl"),
            Path("results/training_history.jsonl"),
            Path("training_history_cuda.jsonl"),
            Path("training_history.jsonl"),
            Path("../results/training_history_cuda.jsonl"),
            Path("../results/training_history.jsonl")
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
            print("请确保训练已经开始并生成了日志文件")
            return
    
    if not Path(args.log_file).exists():
        print(f"❌ 文件不存在: {args.log_file}")
        print("请确保训练已经开始并生成了日志文件")
        return
    
    # 加载数据
    data = load_training_data(args.log_file)
    
    if args.tail:
        data = data[-args.tail:]
    
    if args.watch:
        # 实时监控模式
        watch_training(args.log_file, args.interval, not args.no_chart)
    else:
        # 一次性显示模式
        display_progress(data)
        if not args.no_chart:
            display_fitness_chart(data)

if __name__ == "__main__":
    main()