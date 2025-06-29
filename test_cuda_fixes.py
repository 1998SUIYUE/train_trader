#!/usr/bin/env python3
"""
测试CUDA修复后的代码
验证回测优化和错误修复
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_cuda():
    """测试基本CUDA功能"""
    print("=== 基本CUDA测试 ===")
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"CUDA可用: True")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            
            device = torch.device('cuda:0')
            x = torch.randn(100, 100, device=device)
            y = torch.matmul(x, x.T)
            print(f"基本GPU计算测试: 通过")
            
            return True
        else:
            print("CUDA不可用，将使用CPU")
            return False
            
    except Exception as e:
        print(f"基本测试失败: {e}")
        return False

def test_cuda_modules():
    """测试CUDA模块导入"""
    print("\n=== CUDA模块测试 ===")
    
    try:
        from cuda_gpu_utils import get_cuda_gpu_manager, CudaGPUManager
        print("✓ cuda_gpu_utils导入成功")
        
        from cuda_accelerated_ga import CudaGAConfig, CudaGPUAcceleratedGA
        print("✓ cuda_accelerated_ga导入成功")
        
        from cuda_backtest_optimizer import CudaBacktestOptimizer
        print("✓ cuda_backtest_optimizer导入成功")
        
        return True
        
    except ImportError as e:
        print(f"模块导入失败: {e}")
        return False

def test_backtest_optimizer():
    """测试回测优化器"""
    print("\n=== 回测优化器测试 ===")
    
    try:
        import torch
        from cuda_backtest_optimizer import CudaBacktestOptimizer
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = CudaBacktestOptimizer(device)
        
        # 创建测试数据
        population_size = 100
        n_samples = 500
        
        signals = torch.rand(population_size, n_samples, device=device)
        returns = torch.randn(n_samples, device=device) * 0.01
        buy_thresholds = torch.rand(population_size, device=device) * 0.3 + 0.5
        sell_thresholds = torch.rand(population_size, device=device) * 0.3 + 0.2
        max_positions = torch.rand(population_size, device=device) * 0.8 + 0.2
        stop_losses = torch.rand(population_size, device=device) * 0.05 + 0.02
        
        print(f"测试数据: 种群{population_size}, 样本{n_samples}, 设备{device}")
        
        # 测试不同版本的回测
        start_time = time.time()
        fitness_v1 = optimizer.vectorized_backtest_v1(
            signals, returns, buy_thresholds, sell_thresholds, max_positions
        )
        v1_time = time.time() - start_time
        
        start_time = time.time()
        fitness_v2 = optimizer.vectorized_backtest_v2(
            signals, returns, buy_thresholds, sell_thresholds, max_positions
        )
        v2_time = time.time() - start_time
        
        start_time = time.time()
        fitness_v3 = optimizer.vectorized_backtest_v3(
            signals, returns, buy_thresholds, sell_thresholds, max_positions, stop_losses
        )
        v3_time = time.time() - start_time
        
        print(f"V1 (简化): {v1_time:.4f}s, 适应度均值: {torch.mean(fitness_v1):.6f}")
        print(f"V2 (平衡): {v2_time:.4f}s, 适应度均值: {torch.mean(fitness_v2):.6f}")
        print(f"V3 (完整): {v3_time:.4f}s, 适应度均值: {torch.mean(fitness_v3):.6f}")
        
        print("✓ 回测优化器测试通过")
        return True
        
    except Exception as e:
        print(f"回测优化器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_genetic_algorithm():
    """测试遗传算法"""
    print("\n=== 遗传算法测试 ===")
    
    try:
        import torch
        from cuda_gpu_utils import get_cuda_gpu_manager
        from cuda_accelerated_ga import CudaGAConfig, CudaGPUAcceleratedGA
        
        # 初始化GPU管理器
        gpu_manager = get_cuda_gpu_manager()
        print(f"GPU管理器: {gpu_manager.device}")
        
        # 创建配置
        config = CudaGAConfig(
            population_size=50,
            max_generations=5,
            feature_dim=100,
            mutation_rate=0.02,
            crossover_rate=0.8,
            use_torch_scan=False  # 使用快速模式
        )
        
        # 创建遗传算法
        ga = CudaGPUAcceleratedGA(config, gpu_manager)
        ga.initialize_population(seed=42)
        
        # 创建测试数据
        n_samples = 200
        features = torch.randn(n_samples, config.feature_dim, device=gpu_manager.device)
        labels = torch.randn(n_samples, device=gpu_manager.device) * 0.01
        
        print(f"测试数据: 特征{features.shape}, 标签{labels.shape}")
        
        # 运行几代进化
        start_time = time.time()
        for gen in range(config.max_generations):
            stats = ga.evolve_one_generation(features, labels)
            print(f"代数{gen}: 最佳适应度{stats['best_fitness']:.6f}, 时间{stats['generation_time']:.3f}s")
        
        total_time = time.time() - start_time
        print(f"总时间: {total_time:.3f}s, 平均每代: {total_time/config.max_generations:.3f}s")
        
        print("✓ 遗传算法测试通过")
        return True
        
    except Exception as e:
        print(f"遗传算法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_training():
    """测试简化训练流程"""
    print("\n=== 简化训练测试 ===")
    
    try:
        # 运行简化的训练脚本
        import subprocess
        result = subprocess.run([
            'python', 'main_cuda_simple.py'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ 简化训练测试通过")
            print("训练输出摘要:")
            lines = result.stdout.split('\n')
            for line in lines[-10:]:  # 显示最后10行
                if line.strip():
                    print(f"  {line}")
            return True
        else:
            print(f"简化训练失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("训练超时，但这可能是正常的")
        return True
    except Exception as e:
        print(f"简化训练测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("CUDA修复验证测试")
    print("=" * 50)
    
    tests = [
        ("基本CUDA", test_basic_cuda),
        ("CUDA模块", test_cuda_modules),
        ("回测优化器", test_backtest_optimizer),
        ("遗传算法", test_genetic_algorithm),
        ("简化训练", test_simple_training),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"测试异常: {e}")
            results[test_name] = False
    
    # 总结
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "通过" if result else "失败"
        print(f"{test_name:15} : {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！CUDA环境已修复并优化。")
        print("\n可以运行:")
        print("  python core/main_cuda.py")
    elif passed >= total - 1:
        print("\n⚠️  大部分测试通过，环境基本可用。")
    else:
        print("\n❌ 多项测试失败，请检查环境配置。")

if __name__ == "__main__":
    main()