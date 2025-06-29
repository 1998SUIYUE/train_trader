#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版实时训练图表
确保最大兼容性的动态图表显示
"""

import json
import time
import os
import sys
from pathlib import Path

def find_log_file():
    """查找训练日志文件"""
    possible_paths = [
        "results/training_history_cuda.jsonl",
        "results/training_history.jsonl", 
        "training_history_cuda.jsonl",
        "training_history.jsonl",
        "../results/training_history_cuda.jsonl",
        "../results/training_history.jsonl"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return Path(path)
    return None

def load_data(log_file):
    """安全地加载数据"""
    data = []
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'latin1']
    
    for encoding in encodings:
        try:
            with open(log_file, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"读取文件时出错 ({encoding}): {e}")
            continue
    
    return data

def create_ascii_chart(values, width=60, height=15):
    """创建ASCII图表"""
    if len(values) < 2:
        return ["没有足够的数据"]
    
    # 取最后width个数据点
    data = values[-width:] if len(values) > width else values
    
    min_val = min(data)
    max_val = max(data)
    
    if max_val == min_val:
        return [f"数值恒定: {min_val:.6f}"]
    
    # 归一化到0-height范围
    normalized = []
    for val in data:
        norm_val = int((val - min_val) / (max_val - min_val) * (height - 1))
        normalized.append(norm_val)
    
    # 创建图表
    chart = []
    for row in range(height - 1, -1, -1):
        line = f"{max_val - (max_val - min_val) * (height - 1 - row) / (height - 1):8.4f} |"
        for val in normalized:
            if val >= row:
                line += "█"
            else:
                line += " "
        chart.append(line)
    
    # 添加底部标尺
    chart.append(" " * 10 + "+" + "-" * len(normalized))
    chart.append(f"         范围: {min_val:.4f} ~ {max_val:.4f}")
    
    return chart

def display_info(data):
    """显示训练信息"""
    if not data:
        print("⏳ 等待训练数据...")
        return
    
    latest = data[-1]
    
    # 清屏
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🚀" + "=" * 78 + "🚀")
    print("                    实时训练进度监控")
    print("🚀" + "=" * 78 + "🚀")
    
    # 基本信息
    print(f"📊 当前代数: {latest.get('generation', 0):>12}")
    print(f"🏆 最佳适应度: {latest.get('best_fitness', 0):>16.6f}")
    print(f"📈 平均适应度: {latest.get('avg_fitness', latest.get('mean_fitness', 0)):>16.6f}")
    print(f"📉 标准差: {latest.get('std_fitness', 0):>20.6f}")
    print(f"⏱️  本代用时: {latest.get('generation_time', 0):>16.2f} 秒")
    
    # 交易指标
    if 'mean_sharpe_ratio' in latest:
        print(f"📈 夏普比率: {latest['mean_sharpe_ratio']:>18.6f}")
    if 'mean_sortino_ratio' in latest:
        print(f"📊 索提诺比率: {latest['mean_sortino_ratio']:>16.6f}")
    
    # 系统信息
    if 'system_memory_gb' in latest:
        print(f"💾 系统内存: {latest['system_memory_gb']:>18.2f} GB")
    if 'gpu_memory_allocated' in latest:
        print(f"🎮 GPU内存: {latest['gpu_memory_allocated']:>19.2f} GB")
    
    # 统计信息
    if len(data) > 1:
        total_time = sum(d.get('generation_time', 0) for d in data)
        best_ever = max(d.get('best_fitness', -float('inf')) for d in data)
        avg_time = total_time / len(data)
        
        print(f"\n📊 训练统计:")
        print(f"   总代数: {len(data)}")
        print(f"   历史最佳: {best_ever:.6f}")
        print(f"   总训练时间: {total_time/3600:.2f} 小时")
        print(f"   平均每代: {avg_time:.2f} 秒")
    
    # 显示适应度趋势图
    if len(data) >= 5:
        print(f"\n📈 适应度趋势图 (最近{min(len(data), 60)}代):")
        fitness_values = [d.get('best_fitness', 0) for d in data]
        chart = create_ascii_chart(fitness_values)
        for line in chart:
            print(line)
    
    print("\n" + "=" * 80)
    print(f"🕐 最后更新: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("💡 按 Ctrl+C 停止监控")
    print("=" * 80)

def main():
    """主函数"""
    print("🚀 简化版实时训练图表")
    print("=" * 50)
    
    # 查找日志文件
    log_file = find_log_file()
    if not log_file:
        print("❌ 未找到训练日志文件")
        print("请确保训练已经开始并生成了日志文件")
        print("\n可能的日志文件位置:")
        print("  - results/training_history.jsonl")
        print("  - results/training_history_cuda.jsonl")
        return
    
    print(f"🔍 找到日志文件: {log_file}")
    print("🚀 开始实时监控...\n")
    
    last_size = 0
    
    try:
        while True:
            if log_file.exists():
                current_size = log_file.stat().st_size
                
                if current_size != last_size:
                    # 文件有更新，重新加载数据
                    data = load_data(log_file)
                    display_info(data)
                    last_size = current_size
                else:
                    # 显示等待状态
                    print(f"\r⏳ 等待新数据... {time.strftime('%H:%M:%S')}", end="", flush=True)
            else:
                print(f"\r📁 等待日志文件... {time.strftime('%H:%M:%S')}", end="", flush=True)
            
            time.sleep(3)  # 每3秒检查一次
            
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")
        print("感谢使用训练监控器！")

if __name__ == "__main__":
    main()