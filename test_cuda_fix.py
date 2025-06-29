#!/usr/bin/env python3
"""
测试CUDA遗传算法修复
验证参数传递是否正确
"""

import torch
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_cuda_backtest_optimizer():
    """测试CUDA回测优化器"""
    print("=== 测试CUDA回测优化器修复 ===")
    
    try:
        from cuda_backtest_optimizer import CudaBacktestOptimizer
        
        # 检查CUDA
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，跳过测试")
            return False
        
        device = torch.device('cuda:0')
        optimizer = CudaBacktestOptimizer(device)
        
        # 创建小规模测试数据
        population_size = 10
        n_samples = 100
        
        print(f"测试配置: 种群{population_size}, 样本{n_samples}")
        
        # 生成模拟数据
        signals = torch.rand(population_size, n_samples, device=device)
        returns = torch.randn(n_samples, device=device) * 0.01
        buy_thresholds = torch.rand(population_size, device=device) * 0.25 + 0.55
        sell_thresholds = torch.rand(population_size, device=device) * 0.25 + 0.2
        max_positions = torch.rand(population_size, device=device) * 0.5 + 0.5
        stop_losses = torch.rand(population_size, device=device) * 0.06 + 0.02
        max_drawdowns = torch.rand(population_size, device=device) * 0.15 + 0.1
        
        print("✅ 测试数据生成成功")
        
        # 测试v3方法
        try:
            fitness_v3 = optimizer.vectorized_backtest_v3(
                signals, returns, buy_thresholds, sell_thresholds, 
                max_positions, stop_losses, max_drawdowns
            )
            print(f"✅ vectorized_backtest_v3 调用成功")
            print(f"   适应度形状: {fitness_v3.shape}")
            print(f"   适应度范围: [{fitness_v3.min():.4f}, {fitness_v3.max():.4f}]")
        except Exception as e:
            print(f"❌ vectorized_backtest_v3 调用失败: {e}")
            return False
        
        # 测试v2方法
        try:
            fitness_v2 = optimizer.vectorized_backtest_v2(
                signals, returns, buy_thresholds, sell_thresholds, max_positions
            )
            print(f"✅ vectorized_backtest_v2 调用成功")
            print(f"   适应度形状: {fitness_v2.shape}")
            print(f"   适应度范围: [{fitness_v2.min():.4f}, {fitness_v2.max():.4f}]")
        except Exception as e:
            print(f"❌ vectorized_backtest_v2 调用失败: {e}")
            return False
        
        print("✅ CUDA回测优化器测试通过")
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_cuda_ga_integration():
    """测试CUDA遗传算法集成"""
    print("\n=== 测试CUDA遗传算法集成 ===")
    
    try:
        from cuda_accelerated_ga import CudaGAConfig, CudaGPUAcceleratedGA
        from cuda_gpu_utils import get_cuda_gpu_manager
        
        # 检查CUDA
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，跳过测试")
            return False
        
        # 创建小规模配置
        config = CudaGAConfig(
            population_size=10,
            max_generations=1,
            feature_dim=20,
            use_torch_scan=True  # 测试高精度模式
        )
        
        print(f"✅ 配置创建成功")
        
        # 初始化GPU管理器
        gpu_manager = get_cuda_gpu_manager()
        
        # 创建遗传算法实例
        ga = CudaGPUAcceleratedGA(config, gpu_manager)
        ga.initialize_population(seed=42)
        
        print(f"✅ 遗传算法初始化成功")
        
        # 创建测试数据
        n_samples = 100
        features = torch.randn(n_samples, config.feature_dim, device=gpu_manager.device)
        labels = torch.randn(n_samples, device=gpu_manager.device) * 0.01
        
        print(f"✅ 测试数据创建成功")
        
        # 测试适应度评估
        try:
            fitness_scores = ga.evaluate_fitness_batch(features, labels)
            print(f"✅ 适应度评估成功")
            print(f"   适应度形状: {fitness_scores.shape}")
            print(f"   适应度范围: [{fitness_scores.min():.4f}, {fitness_scores.max():.4f}]")
        except Exception as e:
            print(f"❌ 适应度评估失败: {e}")
            return False
        
        print("✅ CUDA遗传算法集成测试通过")
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🔧 CUDA遗传算法修复验证")
    print("=" * 50)
    
    test1_success = test_cuda_backtest_optimizer()
    test2_success = test_cuda_ga_integration()
    
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print(f"   - CUDA回测优化器: {'✅ 通过' if test1_success else '❌ 失败'}")
    print(f"   - CUDA遗传算法集成: {'✅ 通过' if test2_success else '❌ 失败'}")
    
    if test1_success and test2_success:
        print("\n🎉 所有测试通过！CUDA遗传算法修复成功。")
        print("\n💡 现在你可以正常运行训练了：")
        print("   python core/main_cuda.py")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
    
    return test1_success and test2_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)