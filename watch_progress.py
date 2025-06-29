#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时监控CUDA训练进度
"""

import json
import time
import os
from pathlib import Path

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_data(log_file):
    data = []
    try:
        if Path(log_file).exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
    except:
        pass
    return data

def display_progress(data):
    if not data:
        print("等待训练数据...")
        return
    
    latest = data[-1]
    
    print("=" * 70)
    print("🚀 CUDA训练实时监控")
    print("=" * 70)
    print(f"当前代数: {latest.get('generation', 0)}")
    print(f"最佳适应度: {latest.get('best_fitness', 0):.6f}")
    print(f"平均适应度: {latest.get('mean_fitness', 0):.6f}")
    print(f"本代用时: {latest.get('generation_time', 0):.2f}秒")
    
    if 'mean_sharpe_ratio' in latest:
        print(f"夏普比率: {latest['mean_sharpe_ratio']:.6f}")
    if 'system_memory_gb' in latest:
        print(f"系统内存: {latest['system_memory_gb']:.2f}GB")
    
    if len(data) > 1:
        total_time = sum(d.get('generation_time', 0) for d in data)
        print(f"总训练时间: {total_time/3600:.2f}小时")
        print(f"总代数: {len(data)}")
    
    print("=" * 70)
    print(f"最后更新: {time.strftime('%H:%M:%S')}")
    print("按 Ctrl+C 停止监控")

def watch():
    log_file = "results/training_history.jsonl"
    print(f"监控文件: {log_file}")
    print("刷新间隔: 3秒")
    print()
    
    last_size = 0
    
    try:
        while True:
            if Path(log_file).exists():
                current_size = Path(log_file).stat().st_size
                if current_size != last_size:
                    data = load_data(log_file)
                    clear_screen()
                    display_progress(data)
                    last_size = current_size
                else:
                    print(f"\r等待新数据... {time.strftime('%H:%M:%S')}", end="", flush=True)
            else:
                print(f"\r等待日志文件... {time.strftime('%H:%M:%S')}", end="", flush=True)
            
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n\n监控已停止")

if __name__ == "__main__":
    watch()