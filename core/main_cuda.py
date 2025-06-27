"""
CUDA版遗传算法交易员训练主程序
适用于NVIDIA RTX 4060等CUDA兼容显卡
"""

import time
from pathlib import Path
import json
import torch
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# 检查CUDA可用性
def check_cuda_availability():
    """检查CUDA环境"""
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，请检查：")
        print("  1. 是否安装了NVIDIA驱动")
        print("  2. 是否安装了CUDA Toolkit")
        print("  3. 是否安装了支持CUDA的PyTorch版本")
        print("  安装命令: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    
    print("✅ CUDA环境检查通过")
    print(f"  设备数量: {device_count}")
    print(f"  当前设备: {current_device}")
    print(f"  设备名称: {device_name}")
    print(f"  CUDA版本: {torch.version.cuda}")
    print(f"  显存容量: {torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.1f} GB")
    
    return True

# 简化的GPU管理器（适用于CUDA）
class CudaGPUManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def to_gpu(self, tensor):
        """将张量移动到GPU"""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        elif isinstance(tensor, np.ndarray):
            return torch.from_numpy(tensor).to(self.device)
        else:
            return torch.tensor(tensor).to(self.device)
    
    def to_cpu(self, tensor):
        """将张量移动到CPU"""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor
    
    def clear_cache(self):
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_usage(self):
        """获取显存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3      # GB
            return allocated, cached
        return 0.0, 0.0

# 简化的数据处理器（适用于CUDA）
class CudaDataProcessor:
    def __init__(self, window_size=350, normalization_method='rolling', gpu_manager=None):
        self.window_size = window_size
        self.normalization_method = normalization_method
        self.gpu_manager = gpu_manager or CudaGPUManager()
    
    def load_and_process_data(self, data_file):
        """加载和处理数据"""
        print(f"正在加载数据文件: {data_file}")
        
        # 这里应该实现实际的数据加载逻辑
        # 为了演示，我们创建模拟数据
        n_samples = 1000
        feature_dim = 1400
        
        # 创建模拟特征数据
        features = np.random.randn(n_samples, feature_dim).astype(np.float32)
        
        # 创建模拟标签数据（价格）
        labels = np.cumsum(np.random.randn(n_samples) * 0.01) + 100
        labels = labels.astype(np.float32)
        
        # 转换为GPU张量
        features_tensor = self.gpu_manager.to_gpu(features)
        labels_tensor = self.gpu_manager.to_gpu(labels)
        
        print(f"数据加载完成: 特征 {features_tensor.shape}, 标签 {labels_tensor.shape}")
        return features_tensor, labels_tensor

# 简化的遗传算法配置
class CudaGAConfig:
    def __init__(self, population_size=500, max_generations=100, mutation_rate=0.01, 
                 crossover_rate=0.8, elite_ratio=0.1, feature_dim=1400):
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.feature_dim = feature_dim
        self.gene_length = feature_dim + 5  # 特征权重 + 5个风险参数

# 简化的CUDA遗传算法
class CudaGeneticAlgorithm:
    def __init__(self, config, gpu_manager):
        self.config = config
        self.gpu_manager = gpu_manager
        self.device = gpu_manager.device
        self.population = None
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.generation = 0
    
    def initialize_population(self, seed=None):
        """初始化种群"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.population = torch.randn(
            self.config.population_size, 
            self.config.gene_length,
            device=self.device,
            dtype=torch.float32
        ) * 0.1
        
        print(f"种群初始化完成: {self.population.shape}")
        return self.population
    
    def evaluate_fitness(self, features, prices):
        """简化的适应度评估"""
        # 提取权重
        weights = self.population[:, :self.config.feature_dim]
        
        # 计算决策分数
        scores = torch.mm(weights, features.T)
        
        # 简化的适应度计算（这里应该实现实际的交易策略评估）
        fitness = torch.mean(scores, dim=1) + torch.randn(self.config.population_size, device=self.device) * 0.01
        
        return fitness
    
    def evolve_one_generation(self, features, prices):
        """进化一代"""
        start_time = time.time()
        
        # 适应度评估
        fitness = self.evaluate_fitness(features, prices)
        
        # 更新最佳个体
        best_idx = torch.argmax(fitness)
        current_best_fitness = fitness[best_idx].item()
        
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_individual = self.population[best_idx].clone()
        
        # 简化的选择、交叉、变异
        # 精英保留
        elite_count = max(1, int(self.config.population_size * self.config.elite_ratio))
        elite_indices = torch.topk(fitness, elite_count).indices
        
        # 创建新种群
        new_population = torch.zeros_like(self.population)
        new_population[:elite_count] = self.population[elite_indices]
        
        # 随机交叉和变异
        for i in range(elite_count, self.config.population_size):
            # 随机选择两个父代
            parent1_idx = torch.randint(0, self.config.population_size, (1,)).item()
            parent2_idx = torch.randint(0, self.config.population_size, (1,)).item()
            
            # 交叉
            if torch.rand(1) < self.config.crossover_rate:
                mask = torch.rand(self.config.gene_length, device=self.device) < 0.5
                child = torch.where(mask, self.population[parent1_idx], self.population[parent2_idx])
            else:
                child = self.population[parent1_idx].clone()
            
            # 变异
            mutation_mask = torch.rand(self.config.gene_length, device=self.device) < self.config.mutation_rate
            mutation = torch.randn(self.config.gene_length, device=self.device) * 0.01
            child[mutation_mask] += mutation[mutation_mask]
            
            new_population[i] = child
        
        self.population = new_population
        self.generation += 1
        
        # 统计信息
        gen_time = time.time() - start_time
        allocated_memory, cached_memory = self.gpu_manager.get_memory_usage()
        
        stats = {
            'generation': self.generation,
            'best_fitness': current_best_fitness,
            'mean_fitness': torch.mean(fitness).item(),
            'std_fitness': torch.std(fitness).item(),
            'generation_time': gen_time,
            'system_memory_gb': allocated_memory,
            'mean_sharpe_ratio': current_best_fitness * 0.8,  # 模拟数据
            'mean_sortino_ratio': current_best_fitness * 1.2   # 模拟数据
        }
        
        return stats
    
    def evolve(self, features, prices, save_checkpoints=False, checkpoint_dir=None,
               checkpoint_interval=50, continuous_training=False,
               save_generation_results=False, generation_log_file=None,
               generation_log_interval=1, auto_save_best=False,
               output_dir=None):
        """执行进化过程"""
        
        def save_generation_log(stats_data):
            """保存每代结果到日志文件"""
            if save_generation_results and generation_log_file:
                try:
                    stats_data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(generation_log_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(stats_data, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f"警告：保存每代结果失败: {e}")
        
        total_start_time = time.time()
        fitness_history = []
        
        print("开始CUDA加速进化...")
        
        for gen in range(self.config.max_generations):
            stats = self.evolve_one_generation(features, prices)
            fitness_history.append(stats)
            
            print(f"第 {stats['generation']} 代: "
                  f"最佳适应度={stats['best_fitness']:.4f}, "
                  f"平均适应度={stats['mean_fitness']:.4f}, "
                  f"用时={stats['generation_time']:.2f}秒, "
                  f"显存={stats['system_memory_gb']:.2f}GB")
            
            # 保存每代结果
            if save_generation_results and stats['generation'] % generation_log_interval == 0:
                save_generation_log(stats)
            
            # 清理显存
            if gen % 10 == 0:
                self.gpu_manager.clear_cache()
        
        total_time = time.time() - total_start_time
        
        return {
            'best_individual': self.gpu_manager.to_cpu(self.best_individual),
            'best_fitness': self.best_fitness,
            'fitness_history': fitness_history,
            'total_time': total_time,
            'final_generation': self.generation
        }

def main():
    """主函数 - RTX 4060优化版本"""
    
    # ==============================================================================
    # ======================= RTX 4060 训练参数配置 ============================
    # ==============================================================================
    TRAINING_CONFIG = {
        # --- 数据参数 ---
        "data_directory": "../data",
        "window_size": 350,
        "normalization": "rolling",

        # --- 遗传算法参数（RTX 4060优化）---
        "population_size": 1000,     # RTX 4060可以支持更大的种群
        "generations": 100,          # 训练代数
        "mutation_rate": 0.01,
        "crossover_rate": 0.8,
        "elite_ratio": 0.1,

        # --- 输出参数 ---
        "save_checkpoints": True,
        "checkpoint_interval": 20,
        "results_dir": "../results",
        "save_generation_results": True,
        "generation_log_interval": 1,
        "auto_save_best": True,
    }
    # ==============================================================================
    
    print("=== RTX 4060 CUDA加速遗传算法交易员训练 ===")
    
    # 检查CUDA环境
    if not check_cuda_availability():
        print("❌ CUDA环境不可用，程序退出")
        return
    
    # 设置路径
    output_dir = Path(TRAINING_CONFIG["results_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    data_dir = Path(TRAINING_CONFIG["data_directory"])
    
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("\n--- 训练参数 ---")
    for key, value in TRAINING_CONFIG.items():
        print(f"{key}: {value}")
    print("--------------------\n")
    
    try:
        # 初始化GPU管理器
        gpu_manager = CudaGPUManager()
        print(f"✅ 使用设备: {gpu_manager.device}")
        
        # 数据处理
        processor = CudaDataProcessor(
            window_size=TRAINING_CONFIG["window_size"],
            normalization_method=TRAINING_CONFIG["normalization"],
            gpu_manager=gpu_manager
        )
        
        # 模拟数据文件路径（实际使用时应该从data_dir中选择）
        data_file = data_dir / "sample_data.csv"
        train_features, train_labels = processor.load_and_process_data(data_file)
        
        # 配置遗传算法
        ga_config = CudaGAConfig(
            population_size=TRAINING_CONFIG["population_size"],
            max_generations=TRAINING_CONFIG["generations"],
            mutation_rate=TRAINING_CONFIG["mutation_rate"],
            crossover_rate=TRAINING_CONFIG["crossover_rate"],
            elite_ratio=TRAINING_CONFIG["elite_ratio"],
            feature_dim=train_features.shape[1]
        )
        
        # 初始化遗传算法
        ga = CudaGeneticAlgorithm(ga_config, gpu_manager)
        ga.initialize_population(seed=int(time.time()))
        
        # 设置日志文件
        generation_log_file = output_dir / "training_history.jsonl"
        print(f"📝 训练日志将写入: {generation_log_file}")
        
        # 开始训练
        print("🚀 开始CUDA加速训练...")
        results = ga.evolve(
            train_features,
            train_labels,
            save_checkpoints=TRAINING_CONFIG["save_checkpoints"],
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=TRAINING_CONFIG["checkpoint_interval"],
            save_generation_results=TRAINING_CONFIG["save_generation_results"],
            generation_log_file=generation_log_file,
            generation_log_interval=TRAINING_CONFIG["generation_log_interval"],
            auto_save_best=TRAINING_CONFIG["auto_save_best"],
            output_dir=output_dir
        )
        
        # 保存最终结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        best_individual_path = output_dir / f"best_individual_{timestamp}.npy"
        np.save(best_individual_path, results['best_individual'])
        
        # 输出最终报告
        print("="*60)
        print("           RTX 4060 CUDA训练完成")
        print("="*60)
        print(f"  - 最佳适应度: {results['best_fitness']:.4f}")
        print(f"  - 总训练时间: {results['total_time']:.2f}秒")
        print(f"  - 最终代数:   {results['final_generation']}")
        print(f"  - 最佳个体:   {best_individual_path}")
        print(f"  - 训练日志:   {generation_log_file}")
        print("="*60)
        
        # 显存使用情况
        allocated, cached = gpu_manager.get_memory_usage()
        print(f"  - 显存使用:   {allocated:.2f}GB / {cached:.2f}GB (已分配/已缓存)")
        
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        raise
    finally:
        # 清理显存
        if 'gpu_manager' in locals():
            gpu_manager.clear_cache()

if __name__ == "__main__":
    main()