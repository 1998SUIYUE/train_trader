#!/usr/bin/env python3
"""
训练进度监控演示脚本
展示如何查看和监控训练进度
"""

import sys
import os
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("=" * 80)
    print("🚀 CUDA遗传算法训练进度监控指南")
    print("=" * 80)
    
    print("\n📊 训练进度监控功能:")
    print("1. 实时显示每代训练进度")
    print("2. 显示适应度变化趋势")
    print("3. 监控GPU和系统内存使用")
    print("4. 显示训练时间统计")
    print("5. 自动保存训练日志")
    print("6. 支持图表可视化")
    
    print("\n🔧 可用工具:")
    
    print("\n1. 实时监控正在进行的训练:")
    print("   python tools/watch_training_progress.py --auto")
    print("   python tools/watch_training_progress.py results/training_history_cuda.jsonl")
    
    print("\n2. 查看历史训练日志:")
    print("   python tools/view_training_log.py --auto")
    print("   python tools/view_training_log.py results/training_history_cuda.jsonl")
    
    print("\n3. 生成训练进度图表:")
    print("   python tools/view_training_log.py results/training_history_cuda.jsonl --plot")
    print("   python tools/view_training_log.py results/training_history_cuda.jsonl --save-plot progress.png")
    
    print("\n4. 查看最近的训练记录:")
    print("   python tools/view_training_log.py results/training_history_cuda.jsonl --tail 50")
    
    print("\n📁 日志文件位置:")
    print("   - CUDA训练日志: results/training_history_cuda.jsonl")
    print("   - 普通训练日志: results/training_history.jsonl")
    print("   - 检查点文件: results/checkpoints/")
    print("   - 最佳个体: results/best_individual_*.npy")
    
    print("\n🎯 训练进度信息包含:")
    print("   - 当前代数和目标代数")
    print("   - 最佳适应度和平均适应度")
    print("   - 每代训练时间")
    print("   - 无改进代数计数")
    print("   - GPU内存使用情况")
    print("   - 系统内存使用情况")
    print("   - 训练趋势分析")
    
    print("\n💡 使用建议:")
    print("1. 开始训练前，在另一个终端运行实时监控:")
    print("   python tools/watch_training_progress.py --auto")
    
    print("\n2. 训练完成后，查看完整的训练历史:")
    print("   python tools/view_training_log.py --auto --plot")
    
    print("\n3. 如果训练时间很长，可以定期检查进度:")
    print("   python tools/view_training_log.py --auto --tail 20")
    
    print("\n🚀 现在开始训练:")
    print("   python core/main_cuda.py")
    
    print("\n" + "=" * 80)
    
    # 检查是否有现有的日志文件
    possible_logs = [
        Path("results/training_history_cuda.jsonl"),
        Path("results/training_history.jsonl")
    ]
    
    existing_logs = [log for log in possible_logs if log.exists()]
    
    if existing_logs:
        print("\n📋 发现现有训练日志:")
        for log in existing_logs:
            size = log.stat().st_size / 1024  # KB
            print(f"   {log} ({size:.1f} KB)")
        
        print("\n🔍 查看现有日志:")
        for log in existing_logs:
            print(f"   python tools/view_training_log.py {log}")
    else:
        print("\n📝 未发现现有训练日志，开始新的训练将自动创建日志文件")

if __name__ == "__main__":
    main()