#!/usr/bin/env python3
"""
RTX 4060 环境测试脚本
验证CUDA环境和显卡性能
"""

import torch
import time
import numpy as np

def test_cuda_environment():
    """测试CUDA环境"""
    print("🔍 检查CUDA环境...")
    print("-" * 50)
    
    # 基本CUDA检查
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        print("请检查：")
        print("  1. NVIDIA驱动是否安装")
        print("  2. CUDA Toolkit是否安装")
        print("  3. PyTorch是否支持CUDA")
        return False
    
    print("✅ CUDA可用")
    
    # 设备信息
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    device_props = torch.cuda.get_device_properties(current_device)
    
    print(f"设备数量: {device_count}")
    print(f"当前设备: {current_device}")
    print(f"设备名称: {device_name}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"显存总量: {device_props.total_memory / 1024**3:.1f} GB")
    print(f"多处理器数量: {device_props.multi_processor_count}")
    
    # 检查是否为RTX 4060
    if "4060" in device_name:
        print("🎯 检测到RTX 4060显卡，配置优化建议：")
        print("  - 推荐种群大小: 1000")
        print("  - 推荐批处理大小: 512")
        print("  - 预期显存使用: 4-6GB")
    
    return True

def test_memory_allocation():
    """测试显存分配"""
    print("\n🧪 测试显存分配...")
    print("-" * 50)
    
    device = torch.device('cuda')
    
    # 测试不同大小的张量分配
    test_sizes = [
        (1000, 1400),    # 小型种群
        (2000, 1400),    # 中型种群
        (3000, 1400),    # 大型种群
    ]
    
    for pop_size, gene_length in test_sizes:
        try:
            # 分配张量
            tensor = torch.randn(pop_size, gene_length, device=device)
            
            # 检查显存使用
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            
            print(f"种群大小 {pop_size}x{gene_length}: ✅")
            print(f"  显存使用: {allocated:.2f}GB (分配) / {cached:.2f}GB (缓存)")
            
            # 清理
            del tensor
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"种群大小 {pop_size}x{gene_length}: ❌ 显存不足")
                break
            else:
                print(f"种群大小 {pop_size}x{gene_length}: ❌ 错误: {e}")

def test_computation_speed():
    """测试计算速度"""
    print("\n⚡ 测试计算速度...")
    print("-" * 50)
    
    device = torch.device('cuda')
    
    # 模拟遗传算法的矩阵运算
    pop_size = 1000
    feature_dim = 1400
    n_samples = 1000
    
    print(f"测试配置: 种群{pop_size}, 特征{feature_dim}, 样本{n_samples}")
    
    # 创建测试数据
    population = torch.randn(pop_size, feature_dim, device=device)
    features = torch.randn(n_samples, feature_dim, device=device)
    
    # 测试矩阵乘法速度（适应度评估的核心运算）
    num_tests = 10
    times = []
    
    for i in range(num_tests):
        torch.cuda.synchronize()  # 确保GPU操作完成
        start_time = time.time()
        
        # 模拟适应度评估
        scores = torch.mm(population, features.T)  # (pop_size, n_samples)
        fitness = torch.mean(scores, dim=1)        # (pop_size,)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"平均计算时间: {avg_time:.4f} ± {std_time:.4f} 秒")
    print(f"预估每代训练时间: {avg_time * 2:.2f} 秒")  # 考虑其他操作的开销
    
    # 性能评级
    if avg_time < 0.1:
        print("🚀 性能评级: 优秀")
    elif avg_time < 0.2:
        print("✅ 性能评级: 良好")
    elif avg_time < 0.5:
        print("⚠️  性能评级: 一般")
    else:
        print("❌ 性能评级: 较慢")

def test_mixed_precision():
    """测试混合精度支持"""
    print("\n🔬 测试混合精度支持...")
    print("-" * 50)
    
    device = torch.device('cuda')
    
    try:
        # 测试FP16支持
        with torch.cuda.amp.autocast():
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
        
        print("✅ 支持混合精度训练 (FP16)")
        print("  可以使用混合精度减少显存使用")
        
    except Exception as e:
        print(f"❌ 混合精度测试失败: {e}")

def recommend_settings():
    """推荐配置设置"""
    print("\n📋 RTX 4060 推荐配置...")
    print("-" * 50)
    
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if "4060" in device_name and total_memory >= 7:  # RTX 4060 8GB
        print("🎯 RTX 4060 优化配置:")
        print("""
TRAINING_CONFIG = {
    # RTX 4060 优化参数
    "population_size": 1000,        # 充分利用8GB显存
    "generations": 100,             # 推荐训练代数
    "mutation_rate": 0.01,
    "crossover_rate": 0.8,
    "elite_ratio": 0.1,
    
    # 性能优化
    "batch_size": 512,              # 平衡速度和显存
    "checkpoint_interval": 20,      # 定期保存
    "memory_cleanup_interval": 10,  # 定期清理显存
}
        """)
    else:
        print("⚠️  未检测到RTX 4060，使用通用配置:")
        print("""
TRAINING_CONFIG = {
    "population_size": 500,         # 保守配置
    "generations": 50,
    "mutation_rate": 0.01,
    "crossover_rate": 0.8,
    "elite_ratio": 0.1,
}
        """)

def main():
    """主测试函数"""
    print("🔧 RTX 4060 环境测试")
    print("=" * 60)
    
    # 测试CUDA环境
    if not test_cuda_environment():
        return
    
    # 测试显存分配
    test_memory_allocation()
    
    # 测试计算速度
    test_computation_speed()
    
    # 测试混合精度
    test_mixed_precision()
    
    # 推荐配置
    recommend_settings()
    
    print("\n" + "=" * 60)
    print("🎉 环境测试完成！")
    print("如果所有测试通过，可以开始运行训练：")
    print("  cd core")
    print("  python main_cuda.py")

if __name__ == "__main__":
    main()