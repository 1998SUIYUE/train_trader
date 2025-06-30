"""
第一阶段实现测试脚本
Test script for Phase 1 implementation
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
from pathlib import Path

def test_data_annealing():
    """测试数据退火调度器"""
    print("=== 测试数据退火调度器 ===")
    
    try:
        from data_annealing_scheduler import DataAnnealingScheduler, AnnealingConfig, AnnealingStrategy
        
        # 创建测试配置
        config = AnnealingConfig(
            strategy=AnnealingStrategy.PROGRESSIVE,
            total_generations=100,
            min_data_ratio=0.3,
            warmup_generations=10
        )
        
        scheduler = DataAnnealingScheduler(config)
        
        # 创建测试数据
        features = torch.randn(1000, 100)
        labels = torch.randn(1000) * 0.01
        
        # 测试几个代数
        for gen in [0, 25, 50, 99]:
            annealed_features, annealed_labels, info = scheduler.get_annealed_data(
                gen, features, labels
            )
            print(f"  代数 {gen:2d}: 数据比例={info.get('data_ratio', 1.0):.3f}, "
                  f"样本数={annealed_features.shape[0]}")
        
        print("✅ 数据退火调度器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据退火调度器测试失败: {e}")
        return False

def test_multi_objective():
    """测试多目标优化器"""
    print("\n=== 测试多目标优化器 ===")
    
    try:
        from multi_objective_optimizer import MultiObjectiveOptimizer, MultiObjectiveConfig, ObjectiveConfig, ObjectiveType
        
        # 创建测试配置
        objectives_config = [
            ObjectiveConfig("sharpe_ratio", ObjectiveType.MAXIMIZE, weight=1.0),
            ObjectiveConfig("max_drawdown", ObjectiveType.MINIMIZE, weight=1.0),
            ObjectiveConfig("total_return", ObjectiveType.MAXIMIZE, weight=1.0),
        ]
        
        config = MultiObjectiveConfig(
            objectives=objectives_config,
            pareto_front_size=50,
            enable_hypervolume=True
        )
        
        optimizer = MultiObjectiveOptimizer(config)
        
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
        
        # 评估目标
        objectives = optimizer.evaluate_all_objectives(
            signals, labels, buy_thresholds, sell_thresholds,
            stop_losses, max_positions, max_drawdowns, trade_positions
        )
        
        # 计算帕累托前沿
        pareto_front, domination_counts = optimizer.calculate_pareto_front(objectives)
        
        print(f"  目标数量: {len(objectives)}")
        print(f"  帕累托前沿大小: {len(pareto_front)}")
        print(f"  帕累托比例: {len(pareto_front)/population_size:.3f}")
        
        print("✅ 多目标优化器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 多目标优化器测试失败: {e}")
        return False

def test_enhanced_monitoring():
    """测试增强监控系统"""
    print("\n=== 测试增强监控系统 ===")
    
    try:
        from enhanced_monitoring import EnhancedMonitor, MonitoringConfig
        
        # 创建测试配置
        config = MonitoringConfig(
            save_interval=5,
            detailed_logging=True,
            export_format="json"
        )
        
        monitor = EnhancedMonitor(config)
        monitor.start_monitoring(total_generations=20)
        
        # 模拟几代训练
        for generation in range(10):
            basic_stats = {
                'best_fitness': 0.5 + generation * 0.01,
                'avg_fitness': 0.3 + generation * 0.008,
                'std_fitness': 0.1,
                'generation_time': 2.0,
                'no_improvement_count': max(0, 5 - generation),
            }
            
            multi_objective_stats = {
                'pareto_front_size': 20 + generation,
                'hypervolume': 0.1 + generation * 0.002,
                'pareto_ratio': 0.1 + generation * 0.001,
                'objective_stats': {
                    'sharpe_ratio': {'mean': 1.0 + generation * 0.01},
                    'max_drawdown': {'mean': 0.1 - generation * 0.001},
                }
            }
            
            annealing_stats = {
                'data_ratio': min(1.0, 0.3 + generation * 0.07),
                'complexity_score': generation / 10,
                'strategy': 'progressive',
                'progress': generation / 10,
            }
            
            population = torch.randn(50, 1407)
            
            metrics = monitor.update_metrics(
                generation, basic_stats, multi_objective_stats, 
                annealing_stats, population
            )
        
        # 获取训练总结
        summary = monitor.get_training_summary()
        
        print(f"  监控代数: {len(monitor.metrics_history)}")
        print(f"  最佳适应度: {summary.get('best_fitness_ever', 0.0):.4f}")
        print(f"  收敛状态: {summary.get('convergence_achieved', False)}")
        
        print("✅ 增强监控系统测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 增强监控系统测试失败: {e}")
        return False

def test_enhanced_ga_config():
    """测试增强版遗传算法配置"""
    print("\n=== 测试增强版遗传算法配置 ===")
    
    try:
        from enhanced_cuda_ga import EnhancedGAConfig
        from data_annealing_scheduler import AnnealingStrategy
        
        # 创建测试配置
        config = EnhancedGAConfig(
            population_size=1000,
            max_generations=100,
            feature_dim=100,
            
            enable_data_annealing=True,
            annealing_strategy=AnnealingStrategy.PROGRESSIVE,
            min_data_ratio=0.3,
            
            enable_multi_objective=True,
            pareto_front_size=50,
            
            enable_enhanced_monitoring=True,
            detailed_logging=True,
        )
        
        print(f"  种群大小: {config.population_size}")
        print(f"  数据退火: {config.enable_data_annealing}")
        print(f"  多目标优化: {config.enable_multi_objective}")
        print(f"  增强监控: {config.enable_enhanced_monitoring}")
        print(f"  目标权重: {config.objective_weights}")
        
        print("✅ 增强版遗传算法配置测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 增强版遗传算法配置测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 第一阶段实现测试开始")
    print("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(test_data_annealing())
    test_results.append(test_multi_objective())
    test_results.append(test_enhanced_monitoring())
    test_results.append(test_enhanced_ga_config())
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("🏁 第一阶段实现测试总结")
    print("=" * 60)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"通过测试: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！第一阶段实现成功！")
        print("\n✨ 已实现的功能:")
        print("  ✅ 数据退火机制 - 逐步增加训练数据复杂度")
        print("  ✅ 多目标优化 - 帕累托前沿分析")
        print("  ✅ 增强监控系统 - 全方位性能追踪")
        print("  ✅ 增强版遗传算法配置 - 统一配置管理")
        
        print("\n🎯 下一步建议:")
        print("  1. 运行完整的增强版训练测试")
        print("  2. 开始实施第二阶段功能")
        print("  3. 进行性能对比分析")
        
    else:
        print("❌ 部分测试失败，需要修复问题")
        failed_count = total_tests - passed_tests
        print(f"失败测试数量: {failed_count}")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)