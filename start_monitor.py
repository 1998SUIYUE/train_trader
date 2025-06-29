#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练监控启动器
提供多种监控模式的统一入口
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查依赖库"""
    missing_deps = []
    
    try:
        import matplotlib
        import pandas
        plotting_available = True
    except ImportError:
        plotting_available = False
        missing_deps.extend(['matplotlib', 'pandas'])
    
    return plotting_available, missing_deps

def find_log_files():
    """查找可用的日志文件"""
    possible_paths = [
        Path("results/training_history.jsonl"),
        Path("results/training_history_cuda.jsonl"),
        Path("training_history.jsonl"),
        Path("../results/training_history.jsonl"),
        Path("../results/training_history_cuda.jsonl")
    ]
    
    existing_files = []
    for path in possible_paths:
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            existing_files.append((str(path), size))
    
    return existing_files

def show_menu():
    """显示主菜单"""
    print("🚀" + "=" * 60 + "🚀")
    print("                CUDA训练监控启动器")
    print("🚀" + "=" * 60 + "🚀")
    
    # 检查依赖
    plotting_available, missing_deps = check_dependencies()
    
    if not plotting_available:
        print("⚠️  注意: 图形库不可用，某些功能将受限")
        print(f"   缺少依赖: {', '.join(missing_deps)}")
        print(f"   安装命令: pip install {' '.join(missing_deps)}")
        print()
    
    # 检查日志文件
    log_files = find_log_files()
    if log_files:
        print("📁 发现的日志文件:")
        for file_path, size in log_files:
            print(f"   📄 {file_path} ({size:.1f} KB)")
        print()
    else:
        print("⚠️  未发现训练日志文件")
        print("   请确保训练已经开始并生成了日志文件")
        print()
    
    print("🎯 可用的监控模式:")
    print()
    
    if plotting_available:
        print("1. 🎨 动态图表监控 (推荐)")
        print("   - 实时更新的多图表面板")
        print("   - 适应度曲线、训练时间、内存使用等")
        print("   - 支持历史趋势分析")
        print()
    
    print("2. 📝 文本监控模式")
    print("   - 简洁的文本界面")
    print("   - 实时更新训练状态")
    print("   - 低资源占用")
    print()
    
    print("3. ⚡ 快速监控")
    print("   - 最简单的监控方式")
    print("   - 自动查找日志文件")
    print("   - 包含ASCII图表")
    print()
    
    print("4. 📊 查看历史日志")
    print("   - 分析完整的训练历史")
    print("   - 生成训练报告")
    if plotting_available:
        print("   - 保存图表到文件")
    print()
    
    print("5. 🔧 高级选项")
    print("   - 自定义监控参数")
    print("   - 指定日志文件")
    print()
    
    print("0. 🚪 退出")
    print()

def run_command(cmd):
    """运行命令"""
    try:
        if isinstance(cmd, list):
            subprocess.run(cmd, check=True)
        else:
            subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 命令执行失败: {e}")
    except KeyboardInterrupt:
        print("\n👋 用户中断")

def advanced_options():
    """高级选项菜单"""
    print("\n🔧 高级选项:")
    print("1. 指定日志文件路径")
    print("2. 自定义更新间隔")
    print("3. 设置最大数据点数")
    print("4. 返回主菜单")
    
    choice = input("\n请选择 (1-4): ").strip()
    
    if choice == "1":
        log_path = input("请输入日志文件路径: ").strip()
        if Path(log_path).exists():
            print(f"🎯 启动监控: {log_path}")
            run_command(f"python real_time_training_dashboard.py \"{log_path}\"")
        else:
            print(f"❌ 文件不存在: {log_path}")
    
    elif choice == "2":
        try:
            interval = float(input("请输入更新间隔(秒): ").strip())
            interval_ms = int(interval * 1000)
            print(f"🎯 启动监控，更新间隔: {interval}秒")
            run_command(f"python real_time_training_dashboard.py --auto --interval {interval_ms}")
        except ValueError:
            print("❌ 无效的数值")
    
    elif choice == "3":
        try:
            max_points = int(input("请输入最大数据点数: ").strip())
            print(f"🎯 启动监控，最大数据点: {max_points}")
            run_command(f"python real_time_training_dashboard.py --auto --max-points {max_points}")
        except ValueError:
            print("❌ 无效的数值")
    
    elif choice == "4":
        return
    
    else:
        print("❌ 无效选择")

def main():
    """主函数"""
    while True:
        show_menu()
        
        choice = input("请选择监控模式 (0-5): ").strip()
        
        if choice == "0":
            print("👋 再见！")
            break
        
        elif choice == "1":
            plotting_available, _ = check_dependencies()
            if plotting_available:
                print("\n🎨 启动动态图表监控...")
                run_command("python real_time_training_dashboard.py --auto")
            else:
                print("\n❌ 图形库不可用，请安装matplotlib和pandas")
                print("安装命令: pip install matplotlib pandas")
        
        elif choice == "2":
            print("\n📝 启动文本监控模式...")
            run_command("python real_time_training_dashboard.py --auto --text-mode")
        
        elif choice == "3":
            print("\n⚡ 启动快速监控...")
            run_command("python quick_monitor.py")
        
        elif choice == "4":
            print("\n📊 查看历史日志...")
            plotting_available, _ = check_dependencies()
            if plotting_available:
                run_command("python tools/view_training_log.py --auto --plot")
            else:
                run_command("python tools/view_training_log.py --auto")
        
        elif choice == "5":
            advanced_options()
        
        else:
            print("❌ 无效选择，请重新输入")
        
        print("\n" + "=" * 60)
        input("按回车键继续...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 程序已退出")
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")
        input("按回车键退出...")