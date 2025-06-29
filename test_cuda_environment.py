#!/usr/bin/env python3
"""
CUDA环境测试脚本
验证CUDA 12.9环境下的PyTorch和GPU加速功能
"""

import sys
import time
import numpy as np
from pathlib import Path

def test_basic_imports():
    """测试基本库导入"""
    print("=== 基本库导入测试 ===")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy导入失败: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas导入失败: {e}")
        return False
    
    return True


def test_cuda_availability():
    """测试CUDA可用性"""
    print("\n=== CUDA可用性测试 ===")
    
    import torch
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA编译支持: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        print("可能的原因：")
        print("  1. 没有安装CUDA")
        print("  2. PyTorch版本不支持当前CUDA版本")
        print("  3. NVIDIA驱动程序版本过低")
        return False
    
    print(f"✅ CUDA版本: {torch.version.cuda}")
    print(f"✅ cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"✅ GPU数量: {torch.cuda.device_count()}")
    
    # 显示每个GPU的详细信息
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  - 显存: {props.total_memory / 1e9:.1f} GB")
        print(f"  - 计算能力: {props.major}.{props.minor}")
        print(f"  - 多处理器数量: {props.multi_processor_count}")
    
    return True


def test_gpu_computation():
    """测试GPU计算性能"""
    print("\n=== GPU计算性能测试 ===")
    
    import torch
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过GPU计算测试")
        return False
    
    device = torch.device('cuda:0')
    print(f"使用设备: {device}")
    
    # 测试矩阵乘法
    sizes = [1000, 2000, 4000]
    
    for size in sizes:
        print(f"\n测试矩阵大小: {size}x{size}")
        
        # CPU测试
        x_cpu = torch.randn(size, size)
        y_cpu = torch.randn(size, size)
        
        start_time = time.time()
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        print(f"  CPU时间: {cpu_time:.4f}秒")
        
        # GPU测试
        x_gpu = x_cpu.to(device)
        y_gpu = y_cpu.to(device)
        
        # 预热GPU
        _ = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()
        
        start_time = time.time()
        z_gpu = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"  GPU时间: {gpu_time:.4f}秒")
        
        speedup = cpu_time / gpu_time
        print(f"  加速比: {speedup:.2f}x")
        
        # 验证结果一致性
        z_gpu_cpu = z_gpu.cpu()
        if torch.allclose(z_cpu, z_gpu_cpu, rtol=1e-4):
            print(f"  ✅ 计算结果一致")
        else:
            print(f"  ❌ 计算结果不一致")
            return False
    
    return True


def test_memory_management():
    """测试GPU内存管理"""
    print("\n=== GPU内存管理测试 ===")
    
    import torch
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过内存管理测试")
        return False
    
    device = torch.device('cuda:0')
    
    # 显示初始内存状态
    allocated = torch.cuda.memory_allocated(device) / 1e9
    cached = torch.cuda.memory_reserved(device) / 1e9
    print(f"初始内存 - 已分配: {allocated:.2f}GB, 已缓存: {cached:.2f}GB")
    
    # 分配大量内存
    print("分配大量GPU内存...")
    tensors = []
    try:
        for i in range(10):
            tensor = torch.randn(1000, 1000, device=device)
            tensors.append(tensor)
            
            allocated = torch.cuda.memory_allocated(device) / 1e9
            print(f"  分配第{i+1}个张量后: {allocated:.2f}GB")
    
    except RuntimeError as e:
        print(f"❌ 内存分配失败: {e}")
        return False
    
    # 清理内存
    print("清理GPU内存...")
    del tensors
    torch.cuda.empty_cache()
    
    allocated = torch.cuda.memory_allocated(device) / 1e9
    cached = torch.cuda.memory_reserved(device) / 1e9
    print(f"清理后内存 - 已分配: {allocated:.2f}GB, 已缓存: {cached:.2f}GB")
    
    print("✅ 内存管理测试通过")
    return True


def test_custom_modules():
    """测试自定义模块"""
    print("\n=== 自定义模块测试 ===")
    
    # 添加src目录到路径
    sys.path.append(str(Path(__file__).parent / 'src'))
    
    try:
        from cuda_gpu_utils import get_cuda_gpu_manager, check_cuda_compatibility
        print("✅ cuda_gpu_utils 导入成功")
        
        # 测试GPU管理器
        gpu_manager = get_cuda_gpu_manager()
        print(f"✅ GPU管理器初始化成功: {gpu_manager.device}")
        
        # 测试数据转移
        test_data = np.random.randn(100, 50).astype(np.float32)
        gpu_tensor = gpu_manager.to_gpu(test_data)
        cpu_result = gpu_manager.to_cpu(gpu_tensor)
        
        if np.allclose(test_data, cpu_result):
            print("✅ 数据转移测试通过")
        else:
            print("❌ 数据转移测试失败")
            return False
        
    except ImportError as e:
        print(f"❌ 自定义模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 自定义模块测试失败: {e}")
        return False
    
    try:
        from cuda_accelerated_ga import CudaGAConfig, CudaGPUAcceleratedGA
        print("✅ cuda_accelerated_ga 导入成功")
        
        # 创建测试配置
        config = CudaGAConfig(
            population_size=50,
            max_generations=2,
            feature_dim=20
        )
        
        # 初始化遗传算法
        ga = CudaGPUAcceleratedGA(config, gpu_manager)
        ga.initialize_population(seed=42)
        print("✅ CUDA遗传算法初始化成功")
        
    except ImportError as e:
        print(f"❌ CUDA遗传算法导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ CUDA遗传算法测试失败: {e}")
        return False
    
    return True


def test_torch_scan():
    """测试torch.scan功能"""
    print("\n=== torch.scan功能测试 ===")
    
    import torch
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过torch.scan测试")
        return False
    
    try:
        # 测试torch.func.scan是否可用
        if hasattr(torch, 'func') and hasattr(torch.func, 'scan'):
            print("✅ torch.func.scan 可用")
            
            # 简单的scan测试
            def scan_fn(carry, x):
                return carry + x, carry + x
            
            init = torch.tensor(0.0, device='cuda')
            xs = torch.randn(10, device='cuda')
            
            final_carry, ys = torch.func.scan(scan_fn, init, xs)
            print("✅ torch.scan 基本功能测试通过")
            
        else:
            print("⚠️  torch.func.scan 不可用，将使用传统方法")
            print("   这不会影响训练，但可能会降低性能")
            
    except Exception as e:
        print(f"⚠️  torch.scan 测试失败: {e}")
        print("   将使用传统回测方法")
    
    return True


def main():
    """主测试函数"""
    print("CUDA 12.9 环境测试开始")
    print("=" * 50)
    
    tests = [
        ("基本库导入", test_basic_imports),
        ("CUDA可用性", test_cuda_availability),
        ("GPU计算性能", test_gpu_computation),
        ("GPU内存管理", test_memory_management),
        ("torch.scan功能", test_torch_scan),
        ("自定义模块", test_custom_modules),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results[test_name] = False
    
    # 总结
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！您的CUDA环境已准备就绪。")
        print("\n现在可以运行:")
        print("  python core/main_cuda.py")
    elif passed >= total - 2:
        print("\n⚠️  大部分测试通过，环境基本可用。")
        print("   某些高级功能可能不可用，但不影响基本训练。")
    else:
        print("\n❌ 多项测试失败，请检查环境配置。")
        print("\n建议:")
        print("  1. 重新运行安装脚本: powershell setup/install_cuda129.ps1")
        print("  2. 检查CUDA和PyTorch版本兼容性")
        print("  3. 更新NVIDIA驱动程序")


if __name__ == "__main__":
    main()