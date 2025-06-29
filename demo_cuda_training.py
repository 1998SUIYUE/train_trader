#!/usr/bin/env python3
"""
CUDA版遗传算法交易员训练演示脚本
展示如何在CUDA 12.9环境下运行训练
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent / 'src'))

def create_demo_data():
    """创建演示用的交易数据"""
    print("创建演示交易数据...")
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 生成5000个交易日的数据
    n_days = 5000
    base_price = 2000.0
    
    # 模拟价格随机游走
    daily_returns = np.random.normal(0.0005, 0.02, n_days)  # 平均日收益0.05%，波动率2%
    
    # 添加一些趋势和周期性
    trend = np.linspace(0, 0.5, n_days)  # 长期上升趋势
    cycle = 0.1 * np.sin(2 * np.pi * np.arange(n_days) / 252)  # 年度周期
    
    # 计算累积价格
    price_changes = daily_returns + trend/n_days + cycle/n_days
    close_prices = base_price * np.exp(np.cumsum(price_changes))
    
    # 生成OHLC数据
    opens = np.roll(close_prices, 1)
    opens[0] = base_price
    
    # 生成高低价（基于开盘和收盘价）
    highs = np.maximum(opens, close_prices) * (1 + np.random.exponential(0.005, n_days))
    lows = np.minimum(opens, close_prices) * (1 - np.random.exponential(0.005, n_days))
    
    # 创建DataFrame
    data = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices
    })
    
    # 保存数据
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    data_file = data_dir / 'demo_trading_data.csv'
    data.to_csv(data_file, index=False)
    
    print(f"演示数据已保存: {data_file}")
    print(f"数据形状: {data.shape}")
    print(f"价格范围: {close_prices.min():.2f} - {close_prices.max():.2f}")
    
    return data_file


def demo_cuda_environment():
    """演示CUDA环境检查"""
    print("\n=== CUDA环境检查 ===")
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
        else:
            print("⚠️  CUDA不可用，将使用CPU")
            
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    return True


def demo_cuda_modules():
    """演示CUDA模块功能"""
    print("\n=== CUDA模块测试 ===")
    
    try:
        from cuda_gpu_utils import get_cuda_gpu_manager, check_cuda_compatibility
        from cuda_accelerated_ga import CudaGAConfig, CudaGPUAcceleratedGA
        from data_processor import GPUDataProcessor
        
        print("✅ 所有CUDA模块导入成功")
        
        # 检查CUDA兼容性
        cuda_info = check_cuda_compatibility()
        print(f"CUDA兼容性检查完成: {cuda_info['cuda_available']}")
        
        # 初始化GPU管理器
        gpu_manager = get_cuda_gpu_manager()
        print(f"GPU管理器初始化成功: {gpu_manager.device}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 模块测试失败: {e}")
        return False


def demo_quick_training():
    """演示快速训练"""
    print("\n=== 快速训练演示 ===")
    
    try:
        from cuda_gpu_utils import get_cuda_gpu_manager
        from cuda_accelerated_ga import CudaGAConfig, CudaGPUAcceleratedGA
        from data_processor import GPUDataProcessor
        
        # 创建演示数据
        data_file = create_demo_data()
        
        # 初始化GPU管理器
        gpu_manager = get_cuda_gpu_manager()
        print(f"使用设备: {gpu_manager.device}")
        
        # 初始化数据处理器
        processor = GPUDataProcessor(
            gpu_manager=gpu_manager,
            window_size=100,  # 较小的窗口用于演示
            normalization_method='relative'
        )
        
        # 加载和处理数据
        print("加载和处理数据...")
        features, labels = processor.load_and_process_data(str(data_file))
        print(f"特征形状: {features.shape}, 标签形状: {labels.shape}")
        
        # 创建遗传算法配置（小规模用于演示）
        config = CudaGAConfig(
            population_size=50,      # 小种群
            max_generations=10,      # 少代数
            feature_dim=features.shape[1],
            mutation_rate=0.02,
            crossover_rate=0.8,
            elite_ratio=0.2,
            batch_size=500,
            use_torch_scan=True
        )
        
        print(f"遗传算法配置: 种群{config.population_size}, 代数{config.max_generations}")
        
        # 初始化遗传算法
        ga = CudaGPUAcceleratedGA(config, gpu_manager)
        ga.initialize_population(seed=42)
        
        # 开始训练
        print("开始快速训练演示...")
        start_time = time.time()
        
        results = ga.evolve(features, labels)
        
        training_time = time.time() - start_time
        
        # 显示结果
        print("\n=== 训练结果 ===")
        print(f"最佳适应度: {results['best_fitness']:.6f}")
        print(f"训练代数: {results['final_generation']}")
        print(f"训练时间: {training_time:.2f}秒")
        print(f"平均每代时间: {training_time/results['final_generation']:.3f}秒")
        
        # 显示GPU内存使用
        if gpu_manager.device.type == 'cuda':
            gpu_alloc, gpu_total, sys_used, sys_total = gpu_manager.get_memory_usage()
            print(f"GPU内存使用: {gpu_alloc:.2f}GB / {gpu_total:.2f}GB")
        
        print("✅ 快速训练演示完成！")
        return True
        
    except Exception as e:
        print(f"❌ 训练演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_performance_comparison():
    """演示CPU vs GPU性能对比"""
    print("\n=== 性能对比演示 ===")
    
    try:
        import torch
        
        # 创建测试数据
        size = 2000
        x = torch.randn(size, size)
        y = torch.randn(size, size)
        
        # CPU测试
        print("CPU矩阵乘法测试...")
        start_time = time.time()
        z_cpu = torch.matmul(x, y)
        cpu_time = time.time() - start_time
        print(f"CPU时间: {cpu_time:.4f}秒")
        
        # GPU测试
        if torch.cuda.is_available():
            print("GPU矩阵乘法测试...")
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            
            # 预热
            _ = torch.matmul(x_gpu, y_gpu)
            torch.cuda.synchronize()
            
            start_time = time.time()
            z_gpu = torch.matmul(x_gpu, y_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            print(f"GPU时间: {gpu_time:.4f}秒")
            print(f"加速比: {cpu_time/gpu_time:.2f}x")
            
            # 验证结果一致性
            if torch.allclose(z_cpu, z_gpu.cpu(), rtol=1e-4):
                print("✅ 计算结果一致")
            else:
                print("⚠️  计算结果存在差异")
        else:
            print("⚠️  CUDA不可用，跳过GPU测试")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能对比失败: {e}")
        return False


def main():
    """主演示函数"""
    print("🚀 CUDA版遗传算法交易员训练演示")
    print("=" * 50)
    
    # 检查环境
    if not demo_cuda_environment():
        print("❌ 环境检查失败，请先安装必要的依赖")
        return
    
    # 测试模块
    if not demo_cuda_modules():
        print("❌ 模块测试失败，请检查安装")
        return
    
    # 性能对比
    demo_performance_comparison()
    
    # 询问是否运行训练演示
    print("\n" + "=" * 50)
    print("是否运行快速训练演示？")
    print("这将创建演示数据并运行一个小规模的训练过程")
    response = input("输入 'y' 继续，其他键跳过: ").strip().lower()
    
    if response == 'y':
        if demo_quick_training():
            print("\n🎉 演示完成！")
            print("\n接下来您可以：")
            print("1. 运行完整训练: python core/main_cuda.py")
            print("2. 修改配置参数以适应您的硬件")
            print("3. 使用您自己的交易数据")
        else:
            print("\n❌ 训练演示失败")
    else:
        print("\n演示结束")
    
    print("\n有用的命令：")
    print("  nvidia-smi                    # 查看GPU状态")
    print("  python test_cuda_environment.py  # 完整环境测试")
    print("  python core/main_cuda.py     # 运行完整训练")


if __name__ == "__main__":
    main()