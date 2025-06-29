#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的CUDA训练进度查看器
"""

import json
import sys
from pathlib import Path

def show_progress():
    log_file = "results/training_history.jsonl"
    
    if not Path(log_file).exists():
        print("未找到训练日志文件:", log_file)
        return
    
    # 读取最后几行数据
    data = []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print("读取日志失败:", e)
        return
    
    if not data:
        print("没有训练数据")
        return
    
    latest = data[-1]
    
    print("=" * 60)
    print("CUDA训练进度")
    print("=" * 60)
    print(f"当前代数: {latest.get('generation', 0)}")
    print(f"最佳适应度: {latest.get('best_fitness', 0):.6f}")
    print(f"平均适应度: {latest.get('mean_fitness', 0):.6f}")
    print(f"本代用时: {latest.get('generation_time', 0):.2f}秒")
    
    if 'mean_sharpe_ratio' in latest:
        print(f"夏普比率: {latest['mean_sharpe_ratio']:.6f}")
    if 'mean_sortino_ratio' in latest:
        print(f"索提诺比率: {latest['mean_sortino_ratio']:.6f}")
    if 'system_memory_gb' in latest:
        print(f"系统内存: {latest['system_memory_gb']:.2f}GB")
    
    # 显示最近几代的趋势
    if len(data) >= 5:
        recent_5 = data[-5:]
        print("\n最近5代适应度:")
        for i, d in enumerate(recent_5):
            gen = d.get('generation', 0)
            fitness = d.get('best_fitness', 0)
            print(f"  代数 {gen}: {fitness:.6f}")
    
    print("=" * 60)
    print(f"总共训练了 {len(data)} 代")
    
    if len(data) > 1:
        best_ever = max(d.get('best_fitness', 0) for d in data)
        print(f"历史最佳: {best_ever:.6f}")

if __name__ == "__main__":
    show_progress()