#!/usr/bin/env python3
"""
CUDA训练进度监控演示
展示如何使用各种监控工具
"""

import os
import sys
from pathlib import Path

def main():
    print("=" * 80)
    print("🚀 CUDA训练进度监控演示")
    print("=" * 80)
    
    # 检查训练日志文件
    log_file = Path("results/training_history.jsonl")
    
    if log_file.exists():
        size_kb = log_file.stat().st_size / 1024
        print(f"✅ 发现训练日志文件: {log_file}")
        print(f"📁 文件大小: {size_kb:.1f} KB")
        
        # 快速统计
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"📊 训练记录数: {len(lines)} 条")
        except:
            print("📊 无法读取训练记录")
    else:
        print("❌ 未找到训练日志文件")
        print("💡 请先运行训练程序: python core/main_cuda.py")
        return
    
    print("\n" + "=" * 80)
    print("📋 可用的监控工具:")
    print("=" * 80)
    
    print("\n1. 🖥️  主要监控工具 (推荐)")
    print("   命令: python cuda_progress_monitor.py")
    print("   功能: 完整的训练状态显示，包含图表和详细统计")
    
    print("\n2. ⚡ 实时监控模式")
    print("   命令: python cuda_progress_monitor.py --watch")
    print("   功能: 自动刷新显示最新训练进度")
    
    print("\n3. 📊 查看最近记录")
    print("   命令: python cuda_progress_monitor.py --tail 20")
    print("   功能: 显示最近20代的训练历史")
    
    print("\n4. 🎛️  批处理启动器 (Windows)")
    print("   命令: start_monitoring.bat")
    print("   功能: 菜单式界面，适合不熟悉命令行的用户")
    
    print("\n5. 🔧 简化版工具")
    print("   命令: python show_progress.py")
    print("   功能: 快速查看当前状态")
    
    print("\n" + "=" * 80)
    print("💡 使用建议:")
    print("=" * 80)
    
    print("\n🎯 开始新训练时:")
    print("1. 在一个终端启动训练:")
    print("   python core/main_cuda.py")
    print("\n2. 在另一个终端启动实时监控:")
    print("   python cuda_progress_monitor.py --watch")
    
    print("\n📈 检查训练进度:")
    print("1. 快速查看: python show_progress.py")
    print("2. 详细分析: python cuda_progress_monitor.py")
    print("3. 历史趋势: python cuda_progress_monitor.py --tail 50")
    
    print("\n🔍 监控信息包含:")
    print("- 当前代数和最佳适应度")
    print("- 平均适应度和标准差")
    print("- 每代训练时间")
    print("- 交易性能指标 (夏普比率、索提诺比率、最大回撤)")
    print("- GPU和系统内存使用情况")
    print("- 适应度趋势ASCII图表")
    print("- 最近训练历史")
    
    print("\n" + "=" * 80)
    print("🚀 现在就开始监控!")
    print("=" * 80)
    
    # 提供交互式选择
    print("\n选择要执行的操作:")
    print("1. 查看当前训练状态")
    print("2. 开始实时监控")
    print("3. 查看最近20代")
    print("4. 退出")
    
    try:
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            print("\n正在查看当前训练状态...")
            os.system("python cuda_progress_monitor.py")
        elif choice == "2":
            print("\n开始实时监控 (按 Ctrl+C 停止)...")
            os.system("python cuda_progress_monitor.py --watch")
        elif choice == "3":
            print("\n查看最近20代...")
            os.system("python cuda_progress_monitor.py --tail 20")
        elif choice == "4":
            print("👋 再见!")
        else:
            print("无效选择")
    except KeyboardInterrupt:
        print("\n👋 再见!")

if __name__ == "__main__":
    main()