#!/usr/bin/env python3
"""
测试增强版日志记录功能
Test Enhanced Logging Functionality
"""

import json
import time
from pathlib import Path
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_test_log():
    """Create test log file"""
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    log_file = results_dir / "enhanced_training_history.jsonl"
    
    print(f"🧪 Creating test log file: {log_file}")
    
    # 模拟训练数据
    for generation in range(1, 21):
        test_data = {
            'generation': generation,
            'best_fitness': 0.5 + generation * 0.01 + (generation % 3) * 0.001,
            'avg_fitness': 0.3 + generation * 0.008 + (generation % 2) * 0.001,
            'std_fitness': 0.1 + (generation % 5) * 0.01,
            'generation_time': 2.0 + (generation % 4) * 0.5,
            'no_improvement_count': max(0, 10 - generation // 2),
            'mutation_rate': 0.01,
            'crossover_rate': 0.8,
            'elite_ratio': 0.05,
            
            # 增强版特有数据
            'data_ratio': min(1.0, 0.3 + generation * 0.035),
            'complexity_score': min(1.0, generation / 20),
            'annealing_strategy': 'progressive',
            'annealing_progress': generation / 20,
            
            # 多目标优化数据
            'pareto_front_size': 20 + generation // 2,
            'hypervolume': 0.1 + generation * 0.002,
            'pareto_ratio': 0.1 + generation * 0.001,
            
            # 交易性能数据
            'avg_sharpe_ratio': 1.0 + generation * 0.01,
            'avg_max_drawdown': 0.1 - generation * 0.001,
            'avg_total_return': 0.05 + generation * 0.002,
            'avg_win_rate': 0.6 + generation * 0.001,
            'avg_trade_frequency': 0.1 + generation * 0.001,
            'avg_volatility': 0.2 - generation * 0.001,
            'avg_profit_factor': 1.5 + generation * 0.01,
            
            # 系统性能数据
            'gpu_memory_allocated': 2.0 + (generation % 3) * 0.1,
            'gpu_memory_reserved': 3.0 + (generation % 3) * 0.1,
            'system_memory_gb': 8.0 + (generation % 4) * 0.2,
            
            # 种群多样性
            'population_diversity': 1.0 - generation * 0.02,
        }
        
        # 写入日志
        with open(log_file, 'a', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False)
            f.write('\n')
            f.flush()
        
        print(f"✅ 写入代数 {generation} 的数据")
        time.sleep(0.1)  # 模拟训练间隔
    
    print(f"🎉 测试日志文件创建完成: {log_file}")
    print(f"📊 总共写入 {generation} 条记录")
    
    # 验证文件内容
    print("\n🔍 验证文件内容:")
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"📝 文件行数: {len(lines)}")
            
            # 检查最后一行
            if lines:
                last_line = json.loads(lines[-1].strip())
                print(f"🎯 最后一代: {last_line.get('generation', 'N/A')}")
                print(f"📈 最终适应度: {last_line.get('best_fitness', 'N/A'):.6f}")
                
    except Exception as e:
        print(f"❌ 验证失败: {e}")
    
    return log_file

def test_monitor_compatibility():
    """测试监控器兼容性"""
    print("\n🧪 测试监控器兼容性...")
    
    try:
        # 测试快速监控器
        print("📊 测试快速监控器...")
        from quick_monitor import QuickEnhancedMonitor
        
        monitor = QuickEnhancedMonitor()
        if monitor.log_file:
            print(f"✅ 快速监控器找到日志文件: {monitor.log_file}")
            
            # 加载数据测试
            data = monitor.load_latest_data()
            print(f"📊 加载了 {len(data)} 条记录")
            
            if data:
                print("✅ 快速监控器数据加载正常")
            else:
                print("⚠️  快速监控器未加载到数据")
        else:
            print("❌ 快速监控器未找到日志文件")
            
    except Exception as e:
        print(f"❌ 快速监控器测试失败: {e}")
    
    try:
        # 测试完整监控器
        print("\n📊 测试完整监控器...")
        from enhanced_monitor import EnhancedTrainingMonitor
        
        monitor = EnhancedTrainingMonitor()
        if monitor.log_file:
            print(f"✅ 完整监控器找到日志文件: {monitor.log_file}")
            
            # 加载数据测试
            data = monitor.load_data()
            print(f"📊 加载了 {len(data)} 条记录")
            
            if data:
                print("✅ 完整监控器数据加载正常")
            else:
                print("⚠️  完整监控器未加载到数据")
        else:
            print("❌ 完整监控器未找到日志文件")
            
    except Exception as e:
        print(f"❌ 完整监控器测试失败: {e}")

def main():
    print("🧪 增强版日志记录功能测试")
    print("=" * 50)
    
    # 创建测试日志
    log_file = create_test_log()
    
    # 测试监控器
    test_monitor_compatibility()
    
    print("\n🎉 测试完成!")
    print(f"📁 日志文件位置: {log_file.absolute()}")
    print("\n💡 现在你可以运行监控器:")
    print("  - 快速监控: python quick_monitor.py")
    print("  - 完整监控: python enhanced_monitor.py")

if __name__ == "__main__":
    main()