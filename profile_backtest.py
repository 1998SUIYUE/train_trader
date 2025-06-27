
import torch
import cProfile
import pstats
import io
import sys
from pathlib import Path

# 将 src 目录添加到Python路径中，以解决模块导入问题
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# 现在可以直接从模块名导入
from gpu_accelerated_ga import WindowsGPUAcceleratedGA, WindowsGAConfig
from gpu_utils import get_windows_gpu_manager

def profile_backtest():
    """
    对 batch_fitness_evaluation 函数进行性能分析。
    """
    print("=== 开始性能分析 ===")

    # 1. 设置一个最小化的训练环境
    config = WindowsGAConfig(
        population_size=100,  # 使用较小的种群进行快速分析
        gene_length=1400,
        feature_dim=1400
    )
    
    # 使用与您项目中相同的 GPU 管理器
    gpu_manager = get_windows_gpu_manager()
    device = gpu_manager.device
    print(f"使用设备: {device}")

    ga = WindowsGPUAcceleratedGA(config, gpu_manager)
    ga.initialize_population(seed=42)

    # 2. 创建模拟数据
    n_samples = 2000  # 模拟较长的时间序列
    features = torch.randn(n_samples, config.feature_dim, device=device)
    prices = torch.cumsum(torch.randn(n_samples, device=device) * 0.01, dim=0) + 100
    
    print(f"种群大小: {config.population_size}, 样本数: {n_samples}")

    # 3. 准备 cProfile
    pr = cProfile.Profile()
    pr.enable()

    # 4. 隔离并运行目标函数
    ga.batch_fitness_evaluation(features, prices)

    pr.disable()

    # 5. 输出分析报告
    s = io.StringIO()
    # sortby='cumulative' 可以帮助我们找到总耗时最长的函数
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    print("\n=== 性能分析报告 (按累计耗时排序) ===")
    print(s.getvalue())
    
    # 打印前10个最耗时的函数
    print("\n=== 前10个最耗时的函数 (按函数自身耗时) ===")
    ps.sort_stats('tottime').print_stats(10)


if __name__ == "__main__":
    profile_backtest()
