"""
调试增强版CUDA训练卡住问题
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_save_generation_log():
    """测试日志保存功能"""
    print("=== 测试日志保存功能 ===")
    
    # 创建测试数据
    stats = {
        'generation': 1,
        'best_fitness': 0.123456,
        'avg_fitness': 0.098765,
        'std_fitness': 0.045678,
        'generation_time': 1.234,
        'no_improvement_count': 0,
        'mutation_rate': 0.01,
        'crossover_rate': 0.8,
        'elite_ratio': 0.05,
        'gpu_memory_allocated': 2.5,
        'gpu_memory_reserved': 3.0,
        'system_memory_gb': 8.5,
    }
    
    # 测试文件路径
    test_log_file = Path("test_debug_log.jsonl")
    
    try:
        # 确保目录存在
        test_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"尝试写入日志文件: {test_log_file}")
        
        # 写入文件
        with open(test_log_file, 'a', encoding='utf-8', buffering=1) as f:
            json.dump(stats, f, ensure_ascii=False)
            f.write('\n')
            f.flush()
            import os
            os.fsync(f.fileno())  # 强制写入磁盘
        
        print("✅ 日志写入成功")
        
        # 验证文件内容
        with open(test_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"文件内容: {content[:100]}...")
        
        # 清理测试文件
        test_log_file.unlink()
        print("✅ 测试文件已清理")
        
    except Exception as e:
        print(f"❌ 日志写入失败: {e}")
        import traceback
        traceback.print_exc()

def test_data_annealing():
    """测试数据退火功能"""
    print("\n=== 测试数据退火功能 ===")
    
    try:
        from data_annealing_scheduler import DataAnnealingScheduler, AnnealingConfig, AnnealingStrategy
        
        config = AnnealingConfig(
            strategy=AnnealingStrategy.PROGRESSIVE,
            total_generations=100,
            min_data_ratio=0.3,
            max_data_ratio=1.0,
            warmup_generations=10
        )
        
        scheduler = DataAnnealingScheduler(config)
        print("✅ 数据退火调度器创建成功")
        
        # 创建测试数据
        features = torch.randn(1000, 50)
        labels = torch.randn(1000) * 0.01
        
        print("开始测试退火进度获取...")
        start_time = time.time()
        
        # 测试几代
        for gen in [0, 5, 10, 20]:
            print(f"  测试代数 {gen}...")
            annealed_features, annealed_labels, info = scheduler.get_annealed_data(gen, features, labels)
            print(f"    数据比例: {info.get('data_ratio', 1.0):.3f}")
            
            # 测试获取进度信息
            progress = scheduler.get_annealing_progress()
            print(f"    进度信息: {progress}")
        
        elapsed = time.time() - start_time
        print(f"✅ 数据退火测试完成，耗时: {elapsed:.3f}秒")
        
    except Exception as e:
        print(f"❌ 数据退火测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_multi_objective():
    """测试多目标优化功能"""
    print("\n=== 测试多目标优化功能 ===")
    
    try:
        from multi_objective_optimizer import MultiObjectiveOptimizer, MultiObjectiveConfig, ObjectiveConfig, ObjectiveType
        
        objectives_config = [
            ObjectiveConfig("sharpe_ratio", ObjectiveType.MAXIMIZE, weight=0.3),
            ObjectiveConfig("max_drawdown", ObjectiveType.MINIMIZE, weight=0.2),
            ObjectiveConfig("total_return", ObjectiveType.MAXIMIZE, weight=0.25),
            ObjectiveConfig("win_rate", ObjectiveType.MAXIMIZE, weight=0.15),
            ObjectiveConfig("volatility", ObjectiveType.MINIMIZE, weight=0.1),
        ]
        
        config = MultiObjectiveConfig(
            objectives=objectives_config,
            pareto_front_size=50,
            enable_hypervolume=True
        )
        
        optimizer = MultiObjectiveOptimizer(config)
        print("✅ 多目标优化器创建成功")
        
        # 创建测试数据
        population_size = 100
        n_samples = 200
        
        signals = torch.sigmoid(torch.randn(population_size, n_samples))
        labels = torch.randn(n_samples) * 0.01
        buy_thresholds = torch.rand(population_size) * 0.3 + 0.5
        sell_thresholds = torch.rand(population_size) * 0.3 + 0.2
        stop_losses = torch.rand(population_size) * 0.06 + 0.02
        max_positions = torch.rand(population_size) * 0.5 + 0.5
        max_drawdowns = torch.rand(population_size) * 0.15 + 0.1
        trade_positions = torch.rand(population_size) * 0.8 + 0.2
        
        print("开始测试多目标评估...")
        start_time = time.time()
        
        # 评估所有目标
        objectives = optimizer.evaluate_all_objectives(
            signals, labels, buy_thresholds, sell_thresholds,
            stop_losses, max_positions, max_drawdowns, trade_positions
        )
        
        print(f"✅ 目标评估完成，目标数量: {len(objectives)}")
        
        # 测试获取优化总结
        print("开始测试优化总结...")
        summary = optimizer.get_optimization_summary(objectives)
        print(f"✅ 优化总结完成")
        print(f"    帕累托前沿大小: {summary['pareto_front_size']}")
        print(f"    帕累托比例: {summary['pareto_ratio']:.3f}")
        
        elapsed = time.time() - start_time
        print(f"✅ 多目标优化测试完成，耗时: {elapsed:.3f}秒")
        
    except Exception as e:
        print(f"❌ 多目标优化测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_enhanced_monitoring():
    """测试增强监控功能"""
    print("\n=== 测试增强监控功能 ===")
    
    try:
        from enhanced_monitoring import EnhancedMonitor, MonitoringConfig
        
        config = MonitoringConfig(
            save_interval=5,
            detailed_logging=True,
            track_diversity=True,
            track_convergence=True,
            export_format="json"
        )
        
        monitor = EnhancedMonitor(config)
        print("✅ 增强监控器创建成功")
        
        # 测试更新指标
        print("开始测试监控更新...")
        start_time = time.time()
        
        # 模拟统计数据
        stats = {
            'generation': 1,
            'best_fitness': 0.123,
            'avg_fitness': 0.098,
            'std_fitness': 0.045,
            'generation_time': 1.234,
            'no_improvement_count': 0,
        }
        
        multi_objective_stats = {
            'pareto_front_size': 20,
            'hypervolume': 0.5,
            'pareto_ratio': 0.2,
            'objective_stats': {
                'sharpe_ratio': {'mean': 0.5},
                'max_drawdown': {'mean': 0.1},
            }
        }
        
        annealing_info = {
            'data_ratio': 0.8,
            'complexity_score': 0.7,
            'strategy': 'progressive',
        }
        
        # 创建模拟种群（用于多样性计算）
        population = torch.randn(100, 50)
        
        metrics = monitor.update_metrics(
            1, stats, multi_objective_stats, annealing_info, population
        )
        
        print(f"✅ 监控更新完成")
        
        elapsed = time.time() - start_time
        print(f"✅ 增强监控测试完成，耗时: {elapsed:.3f}秒")
        
    except Exception as e:
        print(f"❌ 增强监控测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("开始调试增强版CUDA训练卡住问题...")
    
    # 测试各个组件
    test_save_generation_log()
    test_data_annealing()
    test_multi_objective()
    test_enhanced_monitoring()
    
    print("\n=== 调试测试完成 ===")

if __name__ == "__main__":
    main()