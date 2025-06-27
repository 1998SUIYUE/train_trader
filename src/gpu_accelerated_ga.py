"""
Windows版GPU加速遗传算法模块
使用DirectML后端实现AMD GPU加速
"""

import torch

import numpy as np
from typing import Tuple, Optional, Dict, Any
import time
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm


from gpu_utils import WindowsGPUManager, get_windows_gpu_manager

@dataclass
class WindowsGAConfig:
    """Windows遗传算法配置"""
    population_size: int = 500  # Windows上建议较小的种群
    gene_length: int = 1405  # 1400特征权重 + 5风险参数
    feature_dim: int = 1400
    mutation_rate: float = 0.01
    crossover_rate: float = 0.8
    elite_ratio: float = 0.1
    tournament_size: int = 5
    max_generations: int = 500  # Windows上建议较少代数
    early_stop_patience: int = 30
    
    # Windows GPU优化参数
    batch_size: int = 500
    use_mixed_precision: bool = False  # DirectML混合精度支持有限
    memory_efficient: bool = True


class WindowsGPUAcceleratedGA:
    """Windows GPU加速的遗传算法"""
    
    def __init__(self, config: WindowsGAConfig, gpu_manager: Optional[WindowsGPUManager] = None):
        """
        初始化Windows GPU加速遗传算法
        
        Args:
            config: 遗传算法配置
            gpu_manager: GPU管理器，None表示使用全局管理器
        """
        self.config = config
        self.gpu_manager = gpu_manager or get_windows_gpu_manager()
        self.device = self.gpu_manager.device
        
        # 初始化种群
        self.population = None
        self.fitness_scores = None
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.generation = 0
        
        # 性能统计
        self.stats = {
            'generation_times': [],
            'fitness_times': [],
            'genetic_op_times': [],
            'memory_usage': []
        }
        
        self._initialize_memory_pools()
        print("--- WindowsGPUAcceleratedGA配置 ---")
        print(f"种群大小: {config.population_size}")
        print(f"基因长度: {config.gene_length}")
        print(f"特征维度: {config.feature_dim}")
        print(f"变异率: {config.mutation_rate}")
        print(f"交叉率: {config.crossover_rate}")
        print(f"精英比例: {config.elite_ratio}")
        print(f"锦标赛大小: {config.tournament_size}")
        print(f"设备: {self.device}")
        print("----------------------------------")
    
    def _initialize_memory_pools(self):
        """初始化内存池"""
        if self.device.type == 'privateuseone':  # DirectML设备
            # 预分配主要数据结构的内存
            self.gpu_manager.create_memory_pool(
                'population', 
                (self.config.population_size, self.config.gene_length),
                torch.float32
            )
            
            self.gpu_manager.create_memory_pool(
                'fitness_scores',
                (self.config.population_size,),
                torch.float32
            )
            
            print("Windows GPU内存池初始化完成")
    
    def initialize_population(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        初始化种群
        
        Args:
            seed: 随机种子
            
        Returns:
            初始化的种群张量
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 在GPU上直接生成随机种群
        population = torch.randn(
            self.config.population_size, 
            self.config.gene_length,
            device=self.device,
            dtype=torch.float32
        )
        
        # 初始化权重部分 (前1400维)
        population[:, :self.config.feature_dim] *= 0.1  # 小的初始权重
        
        # 初始化决策阈值 (1400-1402维)
        population[:, self.config.feature_dim:self.config.feature_dim+2] = torch.rand(
            self.config.population_size, 2, device=self.device
        ) * 0.1 + 0.01  # 阈值范围 [0.01, 0.11]
        
        # 初始化风险参数 (1402-1405维)
        # 止损比例 [0.02, 0.08]
        population[:, self.config.feature_dim+2] = torch.rand(
            self.config.population_size, device=self.device
        ) * 0.06 + 0.02
        
        # 最大仓位比例 [0.2, 0.8]
        population[:, self.config.feature_dim+3] = torch.rand(
            self.config.population_size, device=self.device
        ) * 0.6 + 0.2
        
        # 最大回撤限制 [0.05, 0.15]
        population[:, self.config.feature_dim+4] = torch.rand(
            self.config.population_size, device=self.device
        ) * 0.1 + 0.05
        
        self.population = population
        print(f"种群初始化完成: {population.shape}")
        return population
    
    def batch_fitness_evaluation(self, features: torch.Tensor, prices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量适应度评估 (Windows GPU加速)
        
        Args:
            features: 特征矩阵 (n_samples, feature_dim)
            prices: 价格序列 (n_samples,)
            
        Returns:
            一个元组，包含 (适应度分数, 夏普比率, 索提诺比率)
        """
        start_time = time.time()
        
        # 确保数据在GPU上
        if features.device != self.device:
            features = features.to(self.device)
        if prices.device != self.device:
            prices = prices.to(self.device)
        
        # 提取基因组件
        weights = self.population[:, :self.config.feature_dim]  # (pop_size, 1400)
        buy_threshold = self.population[:, self.config.feature_dim]  # (pop_size,)
        sell_threshold = self.population[:, self.config.feature_dim + 1]  # (pop_size,)
        stop_loss = self.population[:, self.config.feature_dim + 2]  # (pop_size,)
        max_position = self.population[:, self.config.feature_dim + 3]  # (pop_size,)
        max_drawdown = self.population[:, self.config.feature_dim + 4]  # (pop_size,)
        
        # 批量计算决策分数 (GPU矩阵乘法)
        # scores: (pop_size, n_samples)
        scores = torch.mm(weights, features.T)
        
        # 向量化交易信号生成
        buy_signals = scores > buy_threshold.unsqueeze(1)
        sell_signals = scores < -sell_threshold.unsqueeze(1)
        
        # 批量回测计算 (Windows优化版本)
        fitness_scores_tuple = self._vectorized_backtest_windows(
            buy_signals, sell_signals, prices, 
            stop_loss, max_position, max_drawdown, self.generation
        )
        
        self.fitness_scores = fitness_scores_tuple[0]
        self.stats['fitness_times'].append(time.time() - start_time)
        
        return fitness_scores_tuple
    
    def _vectorized_backtest_windows(self, buy_signals: torch.Tensor, sell_signals: torch.Tensor, 
                                   prices: torch.Tensor, stop_loss: torch.Tensor,
                                   max_position: torch.Tensor, max_drawdown: torch.Tensor, generation: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Windows优化的向量化回测计算 (内存优化版)
        
        Args:
            buy_signals: 买入信号 (pop_size, n_samples)
            sell_signals: 卖出信号 (pop_size, n_samples)
            prices: 价格序列 (n_samples,)
            stop_loss: 止损比例 (pop_size,)
            max_position: 最大仓位 (pop_size,)
            max_drawdown: 最大回撤 (pop_size,)
            
        Returns:
            一个元组，包含 (适应度分数, 夏普比率, 索提诺比率)
        """
        pop_size, n_samples = buy_signals.shape
        
        # 初始化状态
        positions = torch.zeros(pop_size, device=self.device)
        equity = torch.ones(pop_size, device=self.device)
        peak_equity = torch.ones(pop_size, device=self.device)
        
        # 内存优化：不保存完整的收益序列，而是计算统计数据
        sum_returns = torch.zeros(pop_size, device=self.device)
        sum_sq_returns = torch.zeros(pop_size, device=self.device)
        downside_sum_sq_returns = torch.zeros(pop_size, device=self.device) # 用于索提诺比率
        trade_counts = torch.zeros(pop_size, device=self.device)

        # 预计算价格变化率
        price_changes = (prices[1:] - prices[:-1]) / prices[:-1]
        price_changes = torch.cat([torch.zeros(1, device=self.device), price_changes]) # 补全第一个为0

        # Windows优化：分块处理以减少内存使用
        chunk_size = min(1000, n_samples)
        
        # 为回测数据块添加内层tqdm进度条
        backtest_progress = tqdm(range(1, n_samples, chunk_size), desc=f"第 {generation} 代 回测", leave=True)

        for chunk_start in backtest_progress:
            chunk_end = min(chunk_start + chunk_size, n_samples)
            
            for t in range(chunk_start, chunk_end):
                price_change = price_changes[t]
                
                # 计算当前收益
                period_return = positions * price_change
                equity += period_return
                
                # 更新统计数据
                sum_returns += period_return
                sum_sq_returns += period_return.pow(2)
                downside_sum_sq_returns += torch.where(period_return < 0, period_return.pow(2), torch.zeros_like(period_return))

                # 更新历史最高净值
                peak_equity = torch.maximum(peak_equity, equity)
                
                # 计算当前回撤
                current_drawdown = (peak_equity - equity) / peak_equity
                
                # 风险控制：强制平仓
                force_close = current_drawdown > max_drawdown
                positions = torch.where(force_close, torch.zeros_like(positions), positions)
                
                # 止损检查
                stop_loss_trigger = (positions > 0) & (price_change < -stop_loss)
                positions = torch.where(stop_loss_trigger, torch.zeros_like(positions), positions)
                
                # 交易信号处理
                can_buy = (positions == 0) & buy_signals[:, t] & (~force_close)
                new_position = torch.where(can_buy, max_position, positions)
                
                can_sell = (positions > 0) & sell_signals[:, t]
                new_position = torch.where(can_sell, torch.zeros_like(positions), new_position)
                
                # 统计交易次数
                trade_counts += (new_position != positions).float()
                
                positions = new_position
            
            if chunk_start % (chunk_size * 5) == 0:
                self.gpu_manager.clear_cache()
        
        # 计算适应度指标
        fitness_metrics = self._calculate_fitness_metrics_windows(
            sum_returns, sum_sq_returns, downside_sum_sq_returns, trade_counts, n_samples, equity, peak_equity
        )
        
        return fitness_metrics
    
    def _calculate_fitness_metrics_windows(self, sum_returns: torch.Tensor, sum_sq_returns: torch.Tensor, 
                                         downside_sum_sq_returns: torch.Tensor, 
                                         trade_counts: torch.Tensor, n_samples: int, 
                                         equity: torch.Tensor, peak_equity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Windows优化的适应度指标计算 (内存优化版)

        Args:
            sum_returns: 总收益
            sum_sq_returns: 收益平方和
            trade_counts: 交易次数
            n_samples: 样本总数
            equity: 最终净值
            peak_equity: 历史最高净值

        Returns:
            一个元组，包含 (综合适应度分数, 夏普比率, 索提诺比率)
        """
        # 计算夏普率
        mean_returns = sum_returns / n_samples
        variance = sum_sq_returns / n_samples - mean_returns.pow(2)
        variance = torch.clamp(variance, min=0)
        std_returns = torch.sqrt(variance)
        sharpe_ratios = mean_returns / (std_returns + 1e-9) * np.sqrt(252)

        # 计算最大回撤
        max_drawdowns = (peak_equity - equity) / peak_equity

        # 计算交易频率稳定性
        stability_scores = 1.0 / (1.0 + trade_counts / n_samples)

        # 计算索提诺比率
        downside_variance = downside_sum_sq_returns / n_samples
        downside_variance = torch.clamp(downside_variance, min=0)
        downside_std = torch.sqrt(downside_variance)
        sortino_ratios = mean_returns / (downside_std + 1e-9) * np.sqrt(252)

        # 综合适应度函数
        fitness = (0.5 * sharpe_ratios -
                   0.3 * max_drawdowns +
                   0.2 * stability_scores)

        # 使用 torch.nan_to_num 替换所有 NaN
        fitness = torch.nan_to_num(fitness, nan=-1e9) # 将NaN替换为一个非常小的值
        sharpe_ratios = torch.nan_to_num(sharpe_ratios, nan=0.0)
        sortino_ratios = torch.nan_to_num(sortino_ratios, nan=0.0)

        return fitness, sharpe_ratios, sortino_ratios
    
    def tournament_selection(self, tournament_size: Optional[int] = None) -> torch.Tensor:
        """
        锦标赛选择 (Windows GPU加速)
        
        Args:
            tournament_size: 锦标赛大小
            
        Returns:
            选中的个体索引
        """
        if tournament_size is None:
            tournament_size = self.config.tournament_size
        
        pop_size = self.config.population_size
        
        # 随机选择锦标赛参与者
        tournament_indices = torch.randint(
            0, pop_size, 
            (pop_size, tournament_size),
            device=self.device
        )
        
        # 获取参与者的适应度
        tournament_fitness = self.fitness_scores[tournament_indices]
        
        # 选择每个锦标赛的获胜者
        winners = torch.argmax(tournament_fitness, dim=1)
        selected_indices = tournament_indices[torch.arange(pop_size), winners]
        
        return selected_indices
    
    def crossover_and_mutation(self, parent_indices: torch.Tensor) -> torch.Tensor:
        """
        交叉和变异操作 (Windows GPU加速)
        
        Args:
            parent_indices: 父代个体索引
            
        Returns:
            新一代种群
        """
        start_time = time.time()
        
        pop_size = self.config.population_size
        new_population = torch.zeros_like(self.population)
        
        # 精英保留
        elite_count = int(pop_size * self.config.elite_ratio)
        # 确保至少保留一个个体，避免k=0的边界情况
        if elite_count == 0 and pop_size > 0:
            elite_count = 1
        

        
        # 交叉操作 (Windows优化：分批处理)

        elite_indices = torch.topk(self.fitness_scores, elite_count).indices
        new_population[:elite_count] = self.population[elite_indices]
        
        # 交叉操作 (Windows优化：分批处理)""
        
        # 交叉操作 (Windows优化：分批处理)
        batch_size = self.config.batch_size
        for i in range(elite_count, pop_size, batch_size):
            end_idx = min(i + batch_size, pop_size)
            batch_size_actual = end_idx - i
            
            # 处理当前批次
            for j in range(0, batch_size_actual, 2):
                idx1, idx2 = i + j, min(i + j + 1, pop_size - 1)
                
                parent1_idx = parent_indices[idx1]
                parent2_idx = parent_indices[idx2]
                
                if torch.rand(1, device=self.device) < self.config.crossover_rate:
                    # 均匀交叉
                    mask = torch.rand(self.config.gene_length, device=self.device) < 0.5
                    
                    child1 = torch.where(mask, 
                                       self.population[parent1_idx], 
                                       self.population[parent2_idx])
                    child2 = torch.where(mask, 
                                       self.population[parent2_idx], 
                                       self.population[parent1_idx])
                else:
                    child1 = self.population[parent1_idx].clone()
                    child2 = self.population[parent2_idx].clone()
                
                new_population[idx1] = child1
                if idx2 < pop_size:
                    new_population[idx2] = child2
            
            # Windows优化：定期清理内存
            if i % (batch_size * 4) == 0:
                self.gpu_manager.clear_cache()
        
        # 变异操作
        mutation_mask = torch.rand(pop_size, self.config.gene_length, device=self.device) < self.config.mutation_rate
        mutation_values = torch.randn(pop_size, self.config.gene_length, device=self.device) * 0.01
        
        new_population[elite_count:] += mutation_mask[elite_count:] * mutation_values[elite_count:]
        
        # 约束风险参数范围
        self._constrain_risk_parameters(new_population)
        
        self.population = new_population
        self.stats['genetic_op_times'].append(time.time() - start_time)
        
        return new_population
    
    def _constrain_risk_parameters(self, population: torch.Tensor):
        """约束风险参数到合理范围"""
        # 决策阈值 [0.001, 0.2]
        population[:, self.config.feature_dim:self.config.feature_dim+2] = torch.clamp(
            population[:, self.config.feature_dim:self.config.feature_dim+2], 0.001, 0.2
        )
        
        # 止损比例 [0.01, 0.1]
        population[:, self.config.feature_dim+2] = torch.clamp(
            population[:, self.config.feature_dim+2], 0.01, 0.1
        )
        
        # 最大仓位 [0.1, 1.0]
        population[:, self.config.feature_dim+3] = torch.clamp(
            population[:, self.config.feature_dim+3], 0.1, 1.0
        )
        
        # 最大回撤 [0.02, 0.3]
        population[:, self.config.feature_dim+4] = torch.clamp(
            population[:, self.config.feature_dim+4], 0.02, 0.3
        )
    
    def evolve_one_generation(self, features: torch.Tensor, prices: torch.Tensor) -> Dict[str, float]:
        """
        进化一代 (Windows优化版本)
        
        Args:
            features: 特征矩阵
            prices: 价格序列
            
        Returns:
            当代统计信息
        """
        gen_start_time = time.time()
        
        # 适应度评估
        fitness_scores, sharpe_ratios, sortino_ratios = self.batch_fitness_evaluation(features, prices)
        
        # 更新最优个体
        best_idx = torch.argmax(fitness_scores).item()
        current_best_fitness = fitness_scores[best_idx].item()
        
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_individual = self.population[best_idx].clone()
        
        # 选择
        parent_indices = self.tournament_selection()
        
        # 交叉和变异
        self.crossover_and_mutation(parent_indices)
        
        self.generation += 1
        
        # 统计信息
        gen_time = time.time() - gen_start_time
        self.stats['generation_times'].append(gen_time)
        
        used_memory, total_memory = self.gpu_manager.get_memory_usage()
        self.stats['memory_usage'].append(used_memory)
        
        # Windows优化：定期清理内存
        if self.generation % 5 == 0:
            self.gpu_manager.clear_cache()
        
        stats = {
            'generation': self.generation,
            'best_fitness': current_best_fitness,
            'mean_fitness': torch.mean(fitness_scores).item(),
            'std_fitness': torch.std(fitness_scores).item(),
            'generation_time': gen_time,
            'system_memory_gb': used_memory,
            'mean_sharpe_ratio': torch.mean(sharpe_ratios).item(),
            'mean_sortino_ratio': torch.mean(sortino_ratios).item()
        }
        
        return stats
    
    def evolve(self, features: torch.Tensor, prices: torch.Tensor,
               save_checkpoints: bool = False, checkpoint_dir: Optional[Path] = None,
               checkpoint_interval: int = 50) -> Dict[str, Any]:
        """
        执行遗传算法进化过程

        Args:
            features: 特征矩阵
            prices: 价格序列
            save_checkpoints: 是否保存检查点
            checkpoint_dir: 检查点保存目录
            checkpoint_interval: 检查点保存间隔

        Returns:
            包含训练结果的字典
        """
        total_start_time = time.time()
        fitness_history = []
        no_improvement_count = 0
        
        # 注意：种群初始化现在由外部调用者（如main.py）处理
        if self.population is None:
            tqdm.write("错误：种群未初始化，请在调用evolve之前调用initialize_population或load_checkpoint")
            raise ValueError("Population not initialized")

        print("--- 开始进化 ---")
        print(f"输入特征形状: {features.shape}")
        print(f"输入价格形状: {prices.shape}")
        print("------------------")

        # 从已加载的代数开始，或从0开始
        start_gen = self.generation

        # 使用tqdm创建进度条
        for gen in range(start_gen, self.config.max_generations):
            stats = self.evolve_one_generation(features, prices)
            fitness_history.append(stats)
            
            # 重新加入日志记录
            tqdm.write(f"第 {stats['generation']} 代: 最佳适应度={stats['best_fitness']:.4f}, "
                       f"平均适应度={stats['mean_fitness']:.4f}, "
                       f"夏普比率={stats['mean_sharpe_ratio']:.4f}, "
                       f"索提诺比率={stats['mean_sortino_ratio']:.4f}, "
                       f"用时={stats['generation_time']:.2f}秒, "
                       f"内存={stats['system_memory_gb']:.2f}GB")
            
            # 检查早期停止
            if stats['best_fitness'] > self.best_fitness:
                self.best_fitness = stats['best_fitness']
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= self.config.early_stop_patience:
                tqdm.write(f"连续{self.config.early_stop_patience}代没有改进，提前停止。")
                break
            
            # 保存检查点
            if save_checkpoints and checkpoint_dir and (gen + 1) % checkpoint_interval == 0:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"checkpoint_gen_{gen+1}.pt"
                self.save_checkpoint(str(checkpoint_path))
        
        total_time = time.time() - total_start_time
        tqdm.write(f"遗传算法进化完成，总用时: {total_time:.2f}秒")
        
        return {
            'best_individual': self.gpu_manager.to_cpu(self.best_individual),
            'best_fitness': self.best_fitness,
            'fitness_history': fitness_history,
            'total_time': total_time,
            'final_generation': self.generation
        }
    
    def get_best_individual(self) -> Tuple[torch.Tensor, float]:
        """
        获取最优个体
        
        Returns:
            (最优个体基因, 最优适应度)
        """
        return self.best_individual, self.best_fitness
    
    def save_checkpoint(self, filepath: str):
        """
        保存训练检查点
        """
        checkpoint = {
            'generation': self.generation,
            'population': self.gpu_manager.to_cpu(self.population),
            'best_individual': self.gpu_manager.to_cpu(self.best_individual) if self.best_individual is not None else None,
            'best_fitness': self.best_fitness,
            'config': self.config,
            'stats': self.stats
        }
        torch.save(checkpoint, filepath)
        tqdm.write(f"检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        加载训练检查点
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.generation = checkpoint['generation']
        self.population = self.gpu_manager.to_gpu(checkpoint['population'])
        self.best_individual = self.gpu_manager.to_gpu(checkpoint['best_individual']) if checkpoint['best_individual'] is not None else None
        self.best_fitness = checkpoint['best_fitness']
        self.stats = checkpoint['stats']
        
        tqdm.write(f"检查点已加载: {filepath}, 代数: {self.generation}")


if __name__ == "__main__":
    # 测试Windows GPU加速遗传算法
    print("=== Windows GPU加速遗传算法测试 ===")
    
    # 配置
    config = WindowsGAConfig(
        population_size=50,  # 小规模测试
        max_generations=5
    )
    
    # 创建模拟数据
    n_samples = 1000
    features = torch.randn(n_samples, config.feature_dim)
    prices = torch.cumsum(torch.randn(n_samples) * 0.01, dim=0) + 100
    
    # 初始化遗传算法
    ga = WindowsGPUAcceleratedGA(config)
    ga.initialize_population(seed=42)
    
    print(f"种群大小: {config.population_size}")
    print(f"基因长度: {config.gene_length}")
    print(f"使用设备: {ga.device}")
    
    # 运行几代进化
    for gen in range(3):
        stats = ga.evolve_one_generation(features, prices)
        print(f"第{stats['generation']}代: "
              f"最优适应度={stats['best_fitness']:.4f}, "
              f"平均适应度={stats['mean_fitness']:.4f}, "
              f"用时={stats['generation_time']:.2f}s")
    
    print("Windows GPU测试完成！")