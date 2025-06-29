"""
torch.scan性能对比测试
比较torch.scan优化版本与传统循环版本的性能差异
"""

import sys
import os
import time
import torch
import numpy as np
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpu_utils import get_windows_gpu_manager
from gpu_accelerated_ga import WindowsGPUAcceleratedGA, WindowsGAConfig

def create_test_data(n_samples: int = 10000, feature_dim: int = 1400):
    """创建测试数据"""
    print(f"创建测试数据: {n_samples}样本, {feature_dim}特征")
    
    # 创建模拟特征数据
    features = torch.randn(n_samples, feature_dim, dtype=torch.float32)
    
    # 创建模拟价格数据（随机游走）
    price_changes = torch.randn(n_samples) * 0.01
    prices = torch.cumsum(price_changes, dim=0) + 100.0
    
    return features, prices

def benchmark_backtest_methods(population_sizes=[100, 500], n_samples_list=[1000, 5000, 10000]):
    """对比不同回测方法的性能"""
    
    gpu_manager = get_windows_gpu_manager()
    print(f"使用设备: {gpu_manager.device}")
    
    results = []
    
    for pop_size in population_sizes:
        for n_samples in n_samples_list:
            print(f"\n=== 测试配置: 种群大小={pop_size}, 样本数={n_samples} ===")
            
            # 创建测试数据
            features, prices = create_test_data(n_samples)
            
            # 测试torch.scan版本
            print("\n--- 测试torch.scan版本 ---")
            config_scan = WindowsGAConfig(
                population_size=pop_size,
                feature_dim=features.shape[1],
                use_torch_scan=True
            )
            
            ga_scan = WindowsGPUAcceleratedGA(config_scan, gpu_manager)
            ga_scan.initialize_population(seed=42)
            
            start_time = time.time()
            try:
                fitness_scan, _, _, _, _ = ga_scan.batch_fitness_evaluation(features, prices)
                scan_time = time.time() - start_time
                scan_success = True
                print(f"torch.scan版本完成，用时: {scan_time:.3f}秒")
            except Exception as e:
                scan_time = float('inf')
                scan_success = False
                print(f"torch.scan版本失败: {e}")
            
            # 测试传统循环版本
            print("\n--- 测试传统循环版本 ---")
            config_legacy = WindowsGAConfig(
                population_size=pop_size,
                feature_dim=features.shape[1],
                use_torch_scan=False
            )
            
            ga_legacy = WindowsGPUAcceleratedGA(config_legacy, gpu_manager)
            ga_legacy.initialize_population(seed=42)
            
            start_time = time.time()
            fitness_legacy, _, _, _, _ = ga_legacy.batch_fitness_evaluation(features, prices)
            legacy_time = time.time() - start_time
            print(f"传统循环版本完成，用时: {legacy_time:.3f}秒")
            
            # 验证结果一致性
            if scan_success:
                fitness_diff = torch.abs(fitness_scan - fitness_legacy).max().item()
                print(f"结果差异: {fitness_diff:.6f}")
                results_match = fitness_diff < 1e-4
            else:
                results_match = False
            
            # 计算加速比
            if scan_success and legacy_time > 0:
                speedup = legacy_time / scan_time
                print(f"加速比: {speedup:.2f}x")
            else:
                speedup = 0
            
            # 记录结果
            result = {
                'population_size': pop_size,
                'n_samples': n_samples,
                'scan_time': scan_time if scan_success else None,
                'legacy_time': legacy_time,
                'speedup': speedup if scan_success else 0,
                'scan_success': scan_success,
                'results_match': results_match
            }
            results.append(result)
            
            # 清理GPU内存
            gpu_manager.clear_cache()
    
    return results

def print_benchmark_summary(results):
    """打印性能测试摘要"""
    print("\n" + "="*80)
    print("                    torch.scan性能对比测试结果")
    print("="*80)
    
    print(f"{'种群大小':<8} {'样本数':<8} {'scan时间':<10} {'循环时间':<10} {'加速比':<8} {'成功':<6} {'一致':<6}")
    print("-" * 80)
    
    total_speedup = []
    success_count = 0
    
    for r in results:
        scan_time_str = f"{r['scan_time']:.3f}s" if r['scan_success'] else "失败"
        legacy_time_str = f"{r['legacy_time']:.3f}s"
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] > 0 else "N/A"
        success_str = "✓" if r['scan_success'] else "✗"
        match_str = "✓" if r['results_match'] else "✗"
        
        print(f"{r['population_size']:<8} {r['n_samples']:<8} {scan_time_str:<10} {legacy_time_str:<10} "
              f"{speedup_str:<8} {success_str:<6} {match_str:<6}")
        
        if r['speedup'] > 0:
            total_speedup.append(r['speedup'])
            success_count += 1
    
    print("-" * 80)
    
    if total_speedup:
        avg_speedup = np.mean(total_speedup)
        max_speedup = np.max(total_speedup)
        min_speedup = np.min(total_speedup)
        
        print(f"成功测试: {success_count}/{len(results)}")
        print(f"平均加速比: {avg_speedup:.2f}x")
        print(f"最大加速比: {max_speedup:.2f}x")
        print(f"最小加速比: {min_speedup:.2f}x")
    else:
        print("所有torch.scan测试都失败了")
    
    print("="*80)

def main():
    """主函数"""
    print("=== torch.scan性能对比测试 ===")
    
    # 检查GPU可用性
    gpu_manager = get_windows_gpu_manager()
    if gpu_manager.device.type == 'cpu':
        print("警告: 未检测到GPU，测试将在CPU上运行")
    
    # 运行性能测试
    print("\n开始性能对比测试...")
    results = benchmark_backtest_methods(
        population_sizes=[50, 200],  # 较小的测试规模
        n_samples_list=[1000, 5000]
    )
    
    # 打印结果摘要
    print_benchmark_summary(results)
    
    # 保存详细结果
    import json
    results_file = Path("torch_scan_benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n详细结果已保存到: {results_file}")

if __name__ == "__main__":
    main()