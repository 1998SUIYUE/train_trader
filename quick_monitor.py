#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动的训练进度监控器
简单易用的实时监控工具
"""

import json
import time
import os
from pathlib import Path
import sys

def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')

def find_log_file():
    """自动查找训练日志文件"""
    possible_paths = [
        "results/training_history.jsonl",
        "results/training_history_cuda.jsonl", 
        "training_history.jsonl",
        "../results/training_history.jsonl",
        "../results/training_history_cuda.jsonl"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    return None

def load_latest_data(log_file):
    """加载最新的训练数据"""
    try:
        if not Path(log_file).exists():
            return None
            
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            return None
            
        # 获取最后一行数据
        latest_line = lines[-1].strip()
        if latest_line:
            return json.loads(latest_line)
            
        return None
    except Exception as e:
        print(f"读取数据出错: {e}")
        return None

def create_progress_bar(current, total, width=50):
    """创建进度条"""
    if total == 0:
        return "[" + "?" * width + "]"
    
    progress = min(current / total, 1.0)
    filled = int(width * progress)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {progress*100:.1f}%"

def display_training_info(data, total_records=0):
    """显示训练信息"""
    clear_screen()
    
    print("🚀" + "=" * 78 + "🚀")
    print("                    CUDA训练实时监控面板")
    print("🚀" + "=" * 78 + "🚀")
    
    if not data:
        print("⏳ 等待训练数据...")
        return
    
    # 基本信息
    current_gen = data.get('generation', 0)
    best_fitness = data.get('best_fitness', 0)
    mean_fitness = data.get('mean_fitness', 0)
    std_fitness = data.get('std_fitness', 0)
    gen_time = data.get('generation_time', 0)
    
    print(f"📊 当前代数: {current_gen:>8}")
    print(f"🏆 最佳适应度: {best_fitness:>12.6f}")
    print(f"📈 平均适应度: {mean_fitness:>12.6f}")
    print(f"📉 标准差: {std_fitness:>16.6f}")
    print(f"⏱️  本代用时: {gen_time:>12.2f} 秒")
    
    # 交易指标
    print("\n💰 交易指标:")
    if 'mean_sharpe_ratio' in data:
        print(f"   📈 夏普比率: {data['mean_sharpe_ratio']:>10.4f}")
    if 'mean_sortino_ratio' in data:
        print(f"   📊 索提诺比率: {data['mean_sortino_ratio']:>8.4f}")
    if 'mean_max_drawdown' in data:
        print(f"   📉 最大回撤: {data['mean_max_drawdown']:>10.4f}")
    if 'mean_overall_return' in data:
        print(f"   💵 总回报率: {data['mean_overall_return']:>10.4f}")
    
    # 系统信息
    print("\n🖥️  系统信息:")
    if 'system_memory_gb' in data:
        print(f"   💾 系统内存: {data['system_memory_gb']:>10.2f} GB")
    if 'gpu_memory_used_gb' in data:
        print(f"   🎮 GPU内存: {data['gpu_memory_used_gb']:>11.2f} GB")
    
    # 训练进度（如果有总代数信息）
    if 'total_generations' in data and data['total_generations'] > 0:
        total_gens = data['total_generations']
        progress_bar = create_progress_bar(current_gen, total_gens)
        print(f"\n🎯 训练进度: {progress_bar}")
        print(f"   ({current_gen}/{total_gens} 代)")
    
    # 统计信息
    if total_records > 1:
        print(f"\n📊 统计信息:")
        print(f"   📝 总记录数: {total_records}")
        
        # 估算剩余时间
        if 'total_generations' in data and data['total_generations'] > 0:
            remaining_gens = data['total_generations'] - current_gen
            if remaining_gens > 0 and gen_time > 0:
                estimated_time = remaining_gens * gen_time
                hours = int(estimated_time // 3600)
                minutes = int((estimated_time % 3600) // 60)
                print(f"   ⏰ 预计剩余: {hours}小时{minutes}分钟")
    
    print("\n" + "=" * 80)
    print(f"🕐 最后更新: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("💡 按 Ctrl+C 停止监控")
    print("=" * 80)

def create_simple_chart(data_history, width=60):
    """创建简单的ASCII图表"""
    if len(data_history) < 2:
        return
    
    print(f"\n📈 适应度趋势图 (最近{len(data_history)}代):")
    
    fitness_values = [d.get('best_fitness', 0) for d in data_history]
    
    if not fitness_values:
        return
    
    # 归一化到0-10的范围
    min_val = min(fitness_values)
    max_val = max(fitness_values)
    
    if max_val == min_val:
        normalized = [5] * len(fitness_values)
    else:
        normalized = [int((val - min_val) / (max_val - min_val) * 10) for val in fitness_values]
    
    # 绘制ASCII图表
    for row in range(10, -1, -1):
        line = f"{max_val - (max_val - min_val) * (10 - row) / 10:8.4f} |"
        for val in normalized:
            if val >= row:
                line += "█"
            else:
                line += " "
        print(line)
    
    # 底部标尺
    print(" " * 10 + "+" + "-" * len(normalized))
    print(f"         范围: {min_val:.4f} ~ {max_val:.4f}")

def monitor_training():
    """主监控函数"""
    # 查找日志文件
    log_file = find_log_file()
    
    if not log_file:
        print("❌ 未找到训练日志文件")
        print("请确保训练已经开始，或者在正确的目录中运行此脚本")
        print("\n可能的日志文件位置:")
        print("  - results/training_history.jsonl")
        print("  - results/training_history_cuda.jsonl")
        return
    
    print(f"🔍 找到日志文件: {log_file}")
    print("🚀 启动实时监控...")
    time.sleep(2)
    
    last_size = 0
    data_history = []
    
    try:
        while True:
            if Path(log_file).exists():
                current_size = Path(log_file).stat().st_size
                
                if current_size != last_size:
                    # 文件有更新
                    data = load_latest_data(log_file)
                    if data:
                        # 保存历史数据用于图表
                        data_history.append(data)
                        if len(data_history) > 50:  # 只保留最近50条
                            data_history.pop(0)
                        
                        # 显示信息
                        display_training_info(data, len(data_history))
                        
                        # 显示简单图表
                        if len(data_history) >= 5:
                            create_simple_chart(data_history[-20:])  # 显示最近20代
                    
                    last_size = current_size
                else:
                    # 文件没有更新，显示等待状态
                    print(f"\r⏳ 等待新数据... {time.strftime('%H:%M:%S')}", end="", flush=True)
            else:
                print(f"\r📁 等待日志文件... {time.strftime('%H:%M:%S')}", end="", flush=True)
            
            time.sleep(3)  # 每3秒检查一次
            
    except KeyboardInterrupt:
        clear_screen()
        print("👋 监控已停止")
        print("感谢使用CUDA训练监控器！")

if __name__ == "__main__":
    print("🚀 CUDA训练快速监控器")
    print("=" * 50)
    monitor_training()