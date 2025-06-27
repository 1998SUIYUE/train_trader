#!/usr/bin/env python3
"""
训练日志查看工具
用于查看和分析实时训练日志文件
"""

import json
import argparse
from pathlib import Path
import time
import os
import sys

# 尝试导入可选依赖
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

def load_training_log(log_file):
    """加载训练日志文件"""
    data = []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"加载日志文件失败: {e}")
        return []

def print_summary(data):
    """打印训练摘要"""
    if not data:
        print("没有数据")
        return
    
    print("="*60)
    print("                训练摘要")
    print("="*60)
    print(f"总代数: {len(data)}")
    print(f"最佳适应度: {max(d['best_fitness'] for d in data):.4f}")
    print(f"最终适应度: {data[-1]['best_fitness']:.4f}")
    print(f"平均每代时间: {sum(d['generation_time'] for d in data) / len(data):.2f}秒")
    
    if 'mean_sharpe_ratio' in data[-1]:
        print(f"最终夏普比率: {data[-1]['mean_sharpe_ratio']:.4f}")
    if 'mean_sortino_ratio' in data[-1]:
        print(f"最终索提诺比率: {data[-1]['mean_sortino_ratio']:.4f}")
    
    print("="*60)

def plot_training_progress(data, save_path=None):
    """绘制训练进度图"""
    if not PLOTTING_AVAILABLE:
        print("错误: 绘图功能需要matplotlib和pandas库")
        print("请安装依赖: pip install matplotlib pandas")
        return
        
    if not data:
        print("没有数据可绘制")
        return
    
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('训练进度分析', fontsize=16)
    
    # 适应度曲线
    axes[0, 0].plot(df['generation'], df['best_fitness'], 'b-', label='最佳适应度')
    axes[0, 0].plot(df['generation'], df['mean_fitness'], 'r--', label='平均适应度')
    axes[0, 0].set_xlabel('代数')
    axes[0, 0].set_ylabel('适应度')
    axes[0, 0].set_title('适应度进化曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 训练时间
    axes[0, 1].plot(df['generation'], df['generation_time'], 'g-')
    axes[0, 1].set_xlabel('代数')
    axes[0, 1].set_ylabel('时间(秒)')
    axes[0, 1].set_title('每代训练时间')
    axes[0, 1].grid(True)
    
    # 夏普比率（如果有）
    if 'mean_sharpe_ratio' in df.columns:
        axes[1, 0].plot(df['generation'], df['mean_sharpe_ratio'], 'purple')
        axes[1, 0].set_xlabel('代数')
        axes[1, 0].set_ylabel('夏普比率')
        axes[1, 0].set_title('平均夏普比率')
        axes[1, 0].grid(True)
    
    # 内存使用（如果有）
    if 'system_memory_gb' in df.columns:
        axes[1, 1].plot(df['generation'], df['system_memory_gb'], 'orange')
        axes[1, 1].set_xlabel('代数')
        axes[1, 1].set_ylabel('内存(GB)')
        axes[1, 1].set_title('系统内存使用')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='训练日志查看工具')
    parser.add_argument('log_file', nargs='?', help='日志文件路径')
    parser.add_argument('--plot', action='store_true', help='绘制训练进度图')
    parser.add_argument('--save-plot', help='保存图表到指定路径')
    parser.add_argument('--tail', type=int, help='只显示最后N条记录')
    parser.add_argument('--auto', action='store_true', help='自动查找最新的训练日志文件')
    
    args = parser.parse_args()
    
    # 如果没有指定文件且启用了auto模式，自动查找最新日志
    if not args.log_file and args.auto:
        # 查找可能的日志文件位置
        possible_paths = [
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
            print("用法: python view_training_log.py [日志文件路径]")
            print("或者: python view_training_log.py --auto")
            return
    elif not args.log_file:
        # 如果没有指定文件也没有auto模式，显示帮助
        print("训练日志查看工具")
        print("用法:")
        print("  python view_training_log.py <日志文件路径>")
        print("  python view_training_log.py --auto  # 自动查找最新日志")
        print("")
        print("选项:")
        print("  --plot          绘制训练进度图")
        print("  --save-plot     保存图表到指定路径")
        print("  --tail N        只显示最后N条记录")
        print("  --auto          自动查找最新的训练日志文件")
        return
    
    if not Path(args.log_file).exists():
        print(f"❌ 文件不存在: {args.log_file}")
        return
    
    # 加载数据
    data = load_training_log(args.log_file)
    
    if args.tail:
        data = data[-args.tail:]
    
    # 打印摘要
    print_summary(data)
    
    # 绘制图表
    if args.plot or args.save_plot:
        try:
            plot_training_progress(data, args.save_plot)
        except ImportError:
            print("需要安装matplotlib和pandas: pip install matplotlib pandas")

if __name__ == "__main__":
    main()