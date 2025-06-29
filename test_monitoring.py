#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控工具测试脚本
生成模拟的训练数据来测试监控工具
"""

import json
import time
import random
import os
from pathlib import Path
import threading

def create_test_data():
    """生成模拟的训练数据"""
    # 确保results目录存在
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    log_file = results_dir / "training_history.jsonl"
    
    print(f"🧪 开始生成测试数据到: {log_file}")
    print("📊 模拟训练进度...")
    
    # 模拟训练参数
    total_generations = 100
    base_fitness = 0.1
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            for generation in range(1, total_generations + 1):
                # 模拟适应度逐渐提升，但有随机波动
                improvement = generation * 0.001 + random.uniform(-0.0005, 0.001)
                best_fitness = base_fitness + improvement + random.uniform(-0.0001, 0.0001)
                mean_fitness = best_fitness - random.uniform(0.001, 0.005)
                std_fitness = random.uniform(0.0005, 0.002)
                
                # 模拟训练时间（有一些随机性）
                generation_time = random.uniform(8, 15)
                
                # 模拟交易指标
                sharpe_ratio = best_fitness * 10 + random.uniform(-0.5, 0.5)
                sortino_ratio = sharpe_ratio * 1.2 + random.uniform(-0.2, 0.2)
                max_drawdown = -random.uniform(0.01, 0.05)
                overall_return = best_fitness * 5 + random.uniform(-0.01, 0.02)
                
                # 模拟系统资源
                system_memory = random.uniform(6.5, 8.5)
                gpu_memory = random.uniform(2.8, 4.2)
                
                # 创建数据记录
                data_point = {
                    "generation": generation,
                    "best_fitness": best_fitness,
                    "mean_fitness": mean_fitness,
                    "std_fitness": std_fitness,
                    "generation_time": generation_time,
                    "mean_sharpe_ratio": sharpe_ratio,
                    "mean_sortino_ratio": sortino_ratio,
                    "mean_max_drawdown": max_drawdown,
                    "mean_overall_return": overall_return,
                    "system_memory_gb": system_memory,
                    "gpu_memory_used_gb": gpu_memory,
                    "total_generations": total_generations,
                    "timestamp": time.time()
                }
                
                # 写入数据
                f.write(json.dumps(data_point, ensure_ascii=False) + '\n')
                f.flush()  # 确保数据立即写入
                
                # 显示进度
                if generation % 10 == 0:
                    print(f"📈 已生成 {generation}/{total_generations} 代数据")
                
                # 模拟训练间隔
                time.sleep(2)  # 每2秒生成一条数据
                
    except KeyboardInterrupt:
        print(f"\n⏹️  测试数据生成已停止 (已生成 {generation} 代)")
    
    print(f"✅ 测试数据生成完成: {log_file}")

def monitor_test():
    """启动监控测试"""
    print("🚀 启动监控工具测试")
    print("=" * 50)
    
    # 检查监控工具是否存在
    monitoring_tools = [
        ("real_time_training_dashboard.py", "动态图表监控"),
        ("quick_monitor.py", "快速监控"),
        ("start_monitor.py", "监控启动器")
    ]
    
    available_tools = []
    for tool_file, tool_name in monitoring_tools:
        if Path(tool_file).exists():
            available_tools.append((tool_file, tool_name))
            print(f"✅ {tool_name}: {tool_file}")
        else:
            print(f"❌ {tool_name}: {tool_file} (未找到)")
    
    if not available_tools:
        print("❌ 未找到任何监控工具")
        return
    
    print(f"\n📊 发现 {len(available_tools)} 个监控工具")
    
    # 检查依赖
    try:
        import matplotlib
        import pandas
        print("✅ 图形库可用 (matplotlib, pandas)")
        plotting_available = True
    except ImportError:
        print("⚠️  图形库不可用，某些功能将受限")
        plotting_available = False
    
    print("\n🎯 测试建议:")
    print("1. 在一个终端运行: python test_monitoring.py --generate")
    print("2. 在另一个终端运行监控工具:")
    
    for tool_file, tool_name in available_tools:
        if "dashboard" in tool_file and plotting_available:
            print(f"   python {tool_file} --auto")
        elif "quick" in tool_file:
            print(f"   python {tool_file}")
        elif "start" in tool_file:
            print(f"   python {tool_file}")
    
    print("\n💡 或者使用自动测试:")
    print("   python test_monitoring.py --auto-test")

def auto_test():
    """自动测试监控工具"""
    print("🤖 启动自动测试...")
    
    # 在后台启动数据生成
    data_thread = threading.Thread(target=create_test_data, daemon=True)
    data_thread.start()
    
    print("⏳ 等待数据生成...")
    time.sleep(5)  # 等待一些初始数据
    
    # 检查是否有快速监控工具
    if Path("quick_monitor.py").exists():
        print("🚀 启动快速监控工具...")
        import subprocess
        try:
            subprocess.run(["python", "quick_monitor.py"], timeout=30)
        except subprocess.TimeoutExpired:
            print("⏰ 测试超时，监控工具运行正常")
        except KeyboardInterrupt:
            print("👋 用户中断测试")
    else:
        print("❌ 未找到快速监控工具")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='监控工具测试脚本')
    parser.add_argument('--generate', action='store_true', help='生成测试数据')
    parser.add_argument('--auto-test', action='store_true', help='自动测试监控工具')
    parser.add_argument('--clean', action='store_true', help='清理测试数据')
    
    args = parser.parse_args()
    
    if args.clean:
        # 清理测试数据
        test_files = [
            Path("results/training_history.jsonl"),
            Path("results/training_history_cuda.jsonl")
        ]
        
        for file_path in test_files:
            if file_path.exists():
                file_path.unlink()
                print(f"🗑️  已删除: {file_path}")
        
        print("✅ 测试数据清理完成")
    
    elif args.generate:
        create_test_data()
    
    elif args.auto_test:
        auto_test()
    
    else:
        monitor_test()

if __name__ == "__main__":
    main()