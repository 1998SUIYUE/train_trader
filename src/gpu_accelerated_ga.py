"""
Windows版GPU加速遗传算法模块
使用DirectML后端实现AMD GPU加速
"""

import torch
import json
import numpy as np
from typing import Tuple, Optional, Dict, Any
import time
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import torch.jit


from gpu_utils import WindowsGPUManager, get_windows_gpu_manager
try:
    from gpu_performance_monitor import GPUPerformanceMonitor, PerformanceContext
except ImportError:
    # 如果性能监控模块不可用，创建一个空的上下文管理器
    class PerformanceContext:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    GPUPerformanceMonitor = None

@dataclass
class WindowsGAConfig:
    """Windows遗传算法配置"""
    population_size: int = 500  # Windows上建议较小的种群
    gene_length: int = 1400  # 只有1400个特征权重
    feature_dim: int = 1400
    mutation_rate: float = 0.01
    crossover_rate: float = 0.8
    elite_ratio: float = 0.1
    tournament_size: int = 5
    max_generations: int = 500  # Windows上建议较少代数
    early_stop_patience: int = 30
    
    # 交易策略参数
    buy_threshold: float = 0.1
    sell_threshold: float = 0.1
    
    # 风险管理参数
    stop_loss: float = 0.05
    max_position: float = 1.0
    max_drawdown: float = 0.2
    
    # 适应度函数权重
    sharpe_weight: float = 0.5
    drawdown_weight: float = 0.3
    stability_weight: float = 0.2
    
    # Windows GPU优化参数
    batch_size: int = 500
    use_mixed_precision: bool = False  # DirectML混合精度支持有限
    memory_efficient: bool = True
    use_torch_scan: bool = True  # 是否使用torch.scan优化交易回测


@torch.jit.script
def _calculate_fitness_metrics_jit(sum_returns: torch.Tensor, sum_sq_returns: torch.Tensor,
                                   downside_sum_sq_returns: torch.Tensor,
                                   trade_counts: torch.Tensor, n_samples: int,
                                   equity: torch.Tensor, peak_equity: torch.Tensor,
                                   sharpe_weight: float, drawdown_weight: float,
                                   stability_weight: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    JIT编译的适应度指标计算
    """
    # 夏普比率
    mean_returns = sum_returns / n_samples
    variance = torch.clamp(sum_sq_returns / n_samples - mean_returns.pow(2), min=0)
    std_returns = torch.sqrt(variance)
    sharpe_ratios = mean_returns / (std_returns + 1e-9) * torch.sqrt(torch.tensor(252.0, device=sum_returns.device))

    # 最大回撤
    max_drawdowns = (peak_equity - equity) / peak_equity

    # 交易频率稳定性
    stability_scores = 1.0 / (1.0 + trade_counts / n_samples)

    # 索提诺比率
    downside_variance = torch.clamp(downside_sum_sq_returns / n_samples, min=0)
    downside_std = torch.sqrt(downside_variance)
    sortino_ratios = mean_returns / (downside_std + 1e-9) * torch.sqrt(torch.tensor(252.0, device=sum_returns.device))

    # 综合适应度函数
    fitness = (sharpe_weight * sharpe_ratios -
               drawdown_weight * max_drawdowns +
               stability_weight * stability_scores)

    # 替换NaN
    fitness = torch.where(torch.isnan(fitness), -1e9, fitness)
    sharpe_ratios = torch.where(torch.isnan(sharpe_ratios), 0.0, sharpe_ratios)
    sortino_ratios = torch.where(torch.isnan(sortino_ratios), 0.0, sortino_ratios)

    return fitness, sharpe_ratios, sortino_ratios, max_drawdowns, equity

@torch.jit.script
def _trading_step_scan(carry: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用torch.scan优化的交易步进函数
    
    Args:
        carry: 状态张量 [pop_size, 7] - [positions, equity, peak_equity, sum_returns, sum_sq_returns, downside_sum_sq_returns, trade_counts]
        x: 输入张量 [pop_size, 6] - [buy_signal, sell_signal, price_change, max_drawdown, stop_loss, max_position]
    
    Returns:
        (updated_carry, output)
    """
    # 解包状态 [pop_size, 7]
    positions = carry[:, 0]
    equity = carry[:, 1] 
    peak_equity = carry[:, 2]
    sum_returns = carry[:, 3]
    sum_sq_returns = carry[:, 4]
    downside_sum_sq_returns = carry[:, 5]
    trade_counts = carry[:, 6]
    
    # 解包输入 [pop_size, 6]
    buy_signal_t = x[:, 0]
    sell_signal_t = x[:, 1]
    price_change_t = x[:, 2]
    max_drawdown = x[:, 3]
    stop_loss = x[:, 4]
    max_position = x[:, 5]

    # --- 核心回测逻辑 (向量化) ---
    period_return = positions * price_change_t
    equity = equity + period_return

    sum_returns = sum_returns + period_return
    sum_sq_returns = sum_sq_returns + period_return.pow(2)
    downside_sum_sq_returns = downside_sum_sq_returns + torch.where(
        period_return < 0, period_return.pow(2), torch.zeros_like(period_return)
    )

    peak_equity = torch.maximum(peak_equity, equity)
    current_drawdown = (peak_equity - equity) / (peak_equity + 1e-8)  # 避免除零
    
    # 风险管理
    force_close = current_drawdown > max_drawdown
    positions = torch.where(force_close, torch.zeros_like(positions), positions)
    
    stop_loss_trigger = (positions > 0) & (price_change_t < -stop_loss)
    positions = torch.where(stop_loss_trigger, torch.zeros_like(positions), positions)
    
    # 交易信号处理
    can_buy = (positions == 0) & (buy_signal_t > 0.5) & (~force_close)
    new_position = torch.where(can_buy, max_position, positions)
    
    can_sell = (positions > 0) & (sell_signal_t > 0.5)
    new_position = torch.where(can_sell, torch.zeros_like(positions), new_position)
    
    trade_counts = trade_counts + (new_position != positions).float()
    positions = new_position

    # 重新打包状态
    new_carry = torch.stack([
        positions, equity, peak_equity, sum_returns, 
        sum_sq_returns, downside_sum_sq_returns, trade_counts
    ], dim=1)
    
    # 输出当前状态用于调试（可选）
    output = torch.stack([positions, equity, period_return], dim=1)
    
    return new_carry, output

@torch.jit.script  
def _legacy_step_function_jit(carry: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                       x: torch.Tensor,
                       max_drawdown: float,
                       stop_loss: float,
                       max_position: float) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None]:
    """
    传统的JIT编译回测步进函数（保留作为备用）
    """
    # 解包状态
    positions, equity, peak_equity, sum_returns, sum_sq_returns, downside_sum_sq_returns, trade_counts = carry
    
    # 解包当前时间步输入
    buy_signal_t, sell_signal_t, price_change_t = x[..., 0], x[..., 1], x[..., 2]

    # --- 核心回测逻辑 ---
    period_return = positions * price_change_t
    equity += period_return

    sum_returns += period_return
    sum_sq_returns += period_return.pow(2)
    downside_sum_sq_returns += torch.where(period_return < 0, period_return.pow(2), torch.zeros_like(period_return))

    peak_equity = torch.maximum(peak_equity, equity)
    current_drawdown = (peak_equity - equity) / peak_equity
    
    force_close = current_drawdown > max_drawdown
    positions = torch.where(force_close, torch.zeros_like(positions), positions)
    
    stop_loss_trigger = (positions > 0) & (price_change_t < -stop_loss)
    positions = torch.where(stop_loss_trigger, torch.zeros_like(positions), positions)
    
    can_buy = (positions == 0) & (buy_signal_t > 0.5) & (~force_close)
    new_position = torch.where(can_buy, torch.full_like(positions, max_position), positions)
    
    can_sell = (positions > 0) & (sell_signal_t > 0.5)
    new_position = torch.where(can_sell, torch.zeros_like(positions), new_position)
    
    trade_counts += (new_position != positions).float()
    positions = new_position

    # --- 返回更新后的状态 ---
    return (
        positions,
        equity,
        peak_equity,
        sum_returns,
        sum_sq_returns,
        downside_sum_sq_returns,
        trade_counts
    ), None


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
        
        # 在GPU上直接生成随机种群 - 只有1400个特征权重
        population = torch.randn(
            self.config.population_size, 
            self.config.gene_length,  # 现在等于feature_dim (1400)
            device=self.device,
            dtype=torch.float32
        ) * 0.1  # 小的初始权重
        
        self.population = population
        print(f"种群初始化完成: {population.shape}")
        print("使用纯特征权重模式 - 所有交易决策都基于1400个特征权重")
        return population
    
    def batch_fitness_evaluation(self, features: torch.Tensor, prices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量适应度评估 (Windows GPU加速)
        
        Args:
            features: 特征矩阵 (n_samples, feature_dim)
            prices: 价格序列 (n_samples,)
            
        Returns:
            一个元组，包含 (适应度分数, 夏普比率, 索提诺比率, 最大回撤, 最终净值)
        """
        start_time = time.time()
        
        # 确保数据在GPU上
        if features.device != self.device:
            features = features.to(self.device)
        if prices.device != self.device:
            prices = prices.to(self.device)
        
        # 现在只有特征权重，所有决策都基于这1400个权重
        weights = self.population
        
        # 批量计算决策分数 (GPU矩阵乘法)
        raw_scores = torch.mm(weights, features.T)
        
        # 使用Sigmoid激活函数将分数映射到[0,1]区间
        scores = torch.sigmoid(raw_scores)

        # 从配置中获取交易策略参数 - 转换为GPU张量
        buy_threshold = torch.tensor(getattr(self.config, 'buy_threshold', 0.6), device=self.device)
        sell_threshold = torch.tensor(getattr(self.config, 'sell_threshold', 0.4), device=self.device)
        
        # 向量化交易信号生成 (完全在GPU上)
        buy_signals = scores > buy_threshold
        sell_signals = scores < sell_threshold
        
        # 交易信号统计 (GPU计算)
        total_signals = torch.numel(scores)
        buy_count = torch.sum(buy_signals)
        sell_count = torch.sum(sell_signals)
        neutral_count = total_signals - buy_count - sell_count
        
        # 只在需要打印时才转换到CPU
        if self.generation % 10 == 0:
            buy_ratio = (buy_count.float() / total_signals * 100).item()
            sell_ratio = (sell_count.float() / total_signals * 100).item()
            neutral_ratio = (neutral_count.float() / total_signals * 100).item()
            tqdm.write(f"  交易信号: 买入{buy_count.item()}次({buy_ratio:.1f}%), 卖出{sell_count.item()}次({sell_ratio:.1f}%), 中性{neutral_count.item()}次({neutral_ratio:.1f}%)")
            tqdm.write(f"  阈值设置: 买入>{buy_threshold.item()}, 卖出<{sell_threshold.item()}, 中性区间[{sell_threshold.item()}, {buy_threshold.item()}]")
        else:
            buy_ratio = (buy_count.float() / total_signals * 100).item()
            sell_ratio = (sell_count.float() / total_signals * 100).item()
            neutral_ratio = (neutral_count.float() / total_signals * 100).item()
            tqdm.write(f"  信号: 买入{buy_ratio:.1f}%, 卖出{sell_ratio:.1f}%, 中性{neutral_ratio:.1f}%")
        
        # 从配置中获取风险管理参数
        stop_loss = getattr(self.config, 'stop_loss', 0.05)
        max_position = getattr(self.config, 'max_position', 1.0)
        max_drawdown = getattr(self.config, 'max_drawdown', 0.2)
        
        # --- torch.scan优化的回测逻辑 ---
        pop_size, n_samples = buy_signals.shape
        device = self.device

        # 在GPU上计算价格变化
        price_changes = torch.zeros_like(prices)
        price_changes[1:] = (prices[1:] - prices[:-1]) / prices[:-1]

        # 准备torch.scan的输入数据 [n_samples, pop_size, 6]
        # 扩展风险管理参数到所有时间步和个体
        max_drawdown_tensor = torch.full((n_samples, pop_size), max_drawdown, device=device)
        stop_loss_tensor = torch.full((n_samples, pop_size), stop_loss, device=device)  
        max_position_tensor = torch.full((n_samples, pop_size), max_position, device=device)
        
        # 组合输入张量 [n_samples, pop_size, 6]
        scan_inputs = torch.stack([
            buy_signals.float().T,           # [n_samples, pop_size] - 买入信号
            sell_signals.float().T,          # [n_samples, pop_size] - 卖出信号  
            price_changes.unsqueeze(1).expand(-1, pop_size),  # [n_samples, pop_size] - 价格变化
            max_drawdown_tensor,             # [n_samples, pop_size] - 最大回撤限制
            stop_loss_tensor,                # [n_samples, pop_size] - 止损比例
            max_position_tensor              # [n_samples, pop_size] - 最大仓位
        ], dim=2)

        # 初始化状态张量 [pop_size, 7]
        init_carry = torch.stack([
            torch.zeros(pop_size, device=device),  # positions
            torch.ones(pop_size, device=device),   # equity  
            torch.ones(pop_size, device=device),   # peak_equity
            torch.zeros(pop_size, device=device),  # sum_returns
            torch.zeros(pop_size, device=device),  # sum_sq_returns
            torch.zeros(pop_size, device=device),  # downside_sum_sq_returns
            torch.zeros(pop_size, device=device)   # trade_counts
        ], dim=1)

        # 根据配置选择回测方法
        if self.config.use_torch_scan:
            # 使用torch.scan进行高效的GPU并行回测
            tqdm.write(f"  使用torch.scan进行GPU并行回测 ({n_samples}步)")
            start_scan_time = time.time()
            
            try:
                # torch.scan: 高效的GPU并行扫描操作
                final_carry, scan_outputs = torch.scan(_trading_step_scan, init_carry, scan_inputs)
                scan_time = time.time() - start_scan_time
                tqdm.write(f"  torch.scan回测完成，用时: {scan_time:.3f}秒")
                
            except Exception as e:
                # 如果torch.scan失败，回退到传统方法
                tqdm.write(f"  torch.scan失败: {e}")
                tqdm.write(f"  回退到传统循环方法")
                final_carry = self._fallback_to_legacy_backtest(
                    buy_signals, sell_signals, price_changes, pop_size, n_samples, 
                    max_drawdown, stop_loss, max_position, device
                )
        else:
            # 使用传统循环方法
            tqdm.write(f"  使用传统循环回测 ({n_samples}步)")
            final_carry = self._fallback_to_legacy_backtest(
                buy_signals, sell_signals, price_changes, pop_size, n_samples,
                max_drawdown, stop_loss, max_position, device
            )
        
        # 解包最终状态 [pop_size, 7]
        final_positions = final_carry[:, 0]
        final_equity = final_carry[:, 1]
        final_peak_equity = final_carry[:, 2] 
        final_sum_returns = final_carry[:, 3]
        final_sum_sq_returns = final_carry[:, 4]
        final_downside_sum_sq_returns = final_carry[:, 5]
        final_trade_counts = final_carry[:, 6]

        # GPU上计算适应度指标
        fitness_scores_tuple = _calculate_fitness_metrics_jit(
            final_sum_returns, final_sum_sq_returns, final_downside_sum_sq_returns,
            final_trade_counts, n_samples, final_equity, final_peak_equity,
            self.config.sharpe_weight, self.config.drawdown_weight, self.config.stability_weight
        )
        
        # 解包返回的元组
        fitness_scores, sharpe_ratios, sortino_ratios, max_drawdowns, final_equity = fitness_scores_tuple
        
        self.fitness_scores = fitness_scores
        self.stats['fitness_times'].append(time.time() - start_time)
        
        return fitness_scores, sharpe_ratios, sortino_ratios, max_drawdowns, final_equity
    
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
        交叉和变异操作 (GPU优化版本)
        
        Args:
            parent_indices: 父代个体索引
            
        Returns:
            新一代种群
        """
        start_time = time.time()
        
        pop_size = self.config.population_size
        new_population = torch.zeros_like(self.population)
        
        # 精英保留 (GPU操作)
        elite_count = int(pop_size * self.config.elite_ratio)
        if elite_count == 0 and pop_size > 0:
            elite_count = 1
        
        elite_indices = torch.topk(self.fitness_scores, elite_count).indices
        new_population[:elite_count] = self.population[elite_indices]
        
        # GPU优化的向量化交叉操作
        remaining_size = pop_size - elite_count
        if remaining_size > 0:
            # 确保偶数个个体进行交叉
            if remaining_size % 2 == 1:
                remaining_size -= 1
            
            # 向量化交叉操作
            parent1_indices = parent_indices[elite_count:elite_count + remaining_size:2]
            parent2_indices = parent_indices[elite_count + 1:elite_count + remaining_size:2]
            
            # 获取父代基因
            parents1 = self.population[parent1_indices]  # (n_pairs, gene_length)
            parents2 = self.population[parent2_indices]  # (n_pairs, gene_length)
            
            # 生成交叉掩码 (GPU向量化)
            n_pairs = len(parents1)
            crossover_mask = torch.rand(n_pairs, 1, device=self.device) < self.config.crossover_rate
            gene_mask = torch.rand(n_pairs, self.config.gene_length, device=self.device) < 0.5
            
            # 向量化交叉
            children1 = torch.where(
                crossover_mask & gene_mask, 
                parents1, 
                torch.where(crossover_mask, parents2, parents1)
            )
            children2 = torch.where(
                crossover_mask & gene_mask, 
                parents2, 
                torch.where(crossover_mask, parents1, parents2)
            )
            
            # 将子代放入新种群
            new_population[elite_count:elite_count + n_pairs] = children1
            new_population[elite_count + n_pairs:elite_count + 2 * n_pairs] = children2
            
            # 处理剩余的奇数个体
            if elite_count + 2 * n_pairs < pop_size:
                remaining_idx = elite_count + 2 * n_pairs
                new_population[remaining_idx] = self.population[parent_indices[remaining_idx]].clone()
        
        # GPU向量化变异操作
        mutation_mask = torch.rand(pop_size, self.config.gene_length, device=self.device) < self.config.mutation_rate
        mutation_values = torch.randn(pop_size, self.config.gene_length, device=self.device) * 0.01
        
        # 只对非精英个体进行变异
        elite_mask = torch.zeros(pop_size, dtype=torch.bool, device=self.device)
        elite_mask[:elite_count] = True
        mutation_mask[elite_mask] = False
        
        new_population += mutation_mask * mutation_values
        
        self.population = new_population
        self.stats['genetic_op_times'].append(time.time() - start_time)
        
        # GPU内存优化：定期清理
        if self.generation % 10 == 0:
            self.gpu_manager.clear_cache()
        
        return new_population
    
    def _fallback_to_legacy_backtest(self, buy_signals: torch.Tensor, sell_signals: torch.Tensor, 
                                   price_changes: torch.Tensor, pop_size: int, n_samples: int,
                                   max_drawdown: float, stop_loss: float, max_position: float, 
                                   device: torch.device) -> torch.Tensor:
        """
        传统循环回测方法（作为torch.scan的备用方案）
        
        Returns:
            final_carry: [pop_size, 7] 最终状态张量
        """
        # 重新组织数据为传统格式
        xs = torch.stack([
            buy_signals.float(),
            sell_signals.float(), 
            price_changes.unsqueeze(0).expand(pop_size, -1)
        ], dim=2).permute(1, 0, 2)
        
        # 传统循环方法
        carry = (
            torch.zeros(pop_size, device=device),  # positions
            torch.ones(pop_size, device=device),   # equity
            torch.ones(pop_size, device=device),   # peak_equity
            torch.zeros(pop_size, device=device),  # sum_returns
            torch.zeros(pop_size, device=device),  # sum_sq_returns
            torch.zeros(pop_size, device=device),  # downside_sum_sq_returns
            torch.zeros(pop_size, device=device)   # trade_counts
        )
        
        # 批量处理以提高效率
        if n_samples > 1000:
            with tqdm(total=n_samples, desc=f"  第 {self.generation} 代传统回测", unit="步", leave=False) as pbar:
                for i in range(n_samples):
                    carry, _ = _legacy_step_function_jit(
                        carry, xs[i], max_drawdown, stop_loss, max_position
                    )
                    if i % 100 == 0:
                        pbar.update(100)
                pbar.update(n_samples % 100)
        else:
            for i in range(n_samples):
                carry, _ = _legacy_step_function_jit(
                    carry, xs[i], max_drawdown, stop_loss, max_position
                )
        
        # 转换为新格式 [pop_size, 7]
        final_carry = torch.stack(carry, dim=1)
        return final_carry
    
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
        fitness_scores, sharpe_ratios, sortino_ratios, max_drawdowns, final_equity = self.batch_fitness_evaluation(features, prices)
        
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
            'mean_sortino_ratio': torch.mean(sortino_ratios).item(),
            'mean_max_drawdown': torch.mean(max_drawdowns).item(), # 添加平均最大回撤
            'mean_overall_return': (torch.mean(final_equity).item() - 1) * 100 # 添加平均整体收益率
        }
        
        return stats
    
    def evolve(self, features: torch.Tensor, prices: torch.Tensor,
               save_checkpoints: bool = False, checkpoint_dir: Optional[Path] = None,
               checkpoint_interval: int = 50, save_generation_results: bool = False, 
               generation_log_file: Optional[Path] = None, generation_log_interval: int = 1, 
               auto_save_best: bool = False, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        执行遗传算法进化过程

        Args:
            features: 特征矩阵
            prices: 价格序列
            save_checkpoints: 是否保存检查点
            checkpoint_dir: 检查点保存目录
            checkpoint_interval: 检查点保存间隔
            save_generation_results: 是否每代保存结果
            generation_log_file: 每代结果日志文件路径
            generation_log_interval: 每隔多少代记录到文件
            auto_save_best: 是否自动保存最佳个体
            output_dir: 输出目录

        Returns:
            包含训练结果的字典
        """
        total_start_time = time.time()
        fitness_history = []
        no_improvement_count = 0
        last_best_fitness = self.best_fitness
        
        # 注意：种群初始化现在由外部调用者（如main.py）处理
        if self.population is None:
            tqdm.write("错误：种群未初始化，请在调用evolve之前调用initialize_population或load_checkpoint")
            raise ValueError("Population not initialized")

        print("--- 开始进化 ---")
        print(f"输入特征形状: {features.shape}")
        print(f"输入价格形状: {prices.shape}")
        print(f"训练代数: {self.config.max_generations if self.config.max_generations > 0 else '无限'}")
        print(f"每代结果记录: {'启用' if save_generation_results else '禁用'}")
        if generation_log_file:
            print(f"结果日志文件: {generation_log_file}")
        print("------------------")

        # 从已加载的代数开始，或从0开始
        start_gen = self.generation
        
        # 确定最大代数
        if self.config.max_generations == -1:
            max_generations = float('inf')
            print("🔄 无限训练模式：将无限期训练，按Ctrl+C停止")
        else:
            max_generations = self.config.max_generations

        # 初始化每代结果记录
        def save_generation_log(stats_data):
            """保存每代结果到日志文件"""
            if save_generation_results and generation_log_file:
                try:
                    # 添加时间戳
                    stats_data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 以追加模式写入JSONL格式
                    with open(generation_log_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(stats_data, ensure_ascii=False) + '\n')
                except Exception as e:
                    tqdm.write(f"警告：保存每代结果失败: {e}")

        # 主训练循环
        gen = start_gen
        try:
            while gen < max_generations:
                stats = self.evolve_one_generation(features, prices)
                fitness_history.append(stats)
                
                # 每10代使用tqdm.write输出一次详细信息
                if gen % 10 == 0:
                    tqdm.write(f"第 {stats['generation']} 代: 最佳适应度={stats['best_fitness']:.4f}, "
                               f"平均适应度={stats['mean_fitness']:.4f}, "
                               f"夏普比率={stats['mean_sharpe_ratio']:.4f}, "
                               f"索提诺比率={stats['mean_sortino_ratio']:.4f}, "
                               f"最大回撤={stats['mean_max_drawdown']:.4f}, "
                               f"整体收益率={stats['mean_overall_return']:.2f}%, "
                               f"用时={stats['generation_time']:.2f}秒, "
                               f"内存={stats['system_memory_gb']:.2f}GB")

                # 每代结果记录
                if save_generation_results and stats['generation'] % generation_log_interval == 0:
                    save_generation_log(stats)
                
                # 自动保存最佳个体
                if auto_save_best and output_dir and stats['best_fitness'] > last_best_fitness:
                    try:
                        best_path = output_dir / f"best_individual_gen_{stats['generation']}.npy"
                        np.save(best_path, self.gpu_manager.to_cpu(self.best_individual))
                        tqdm.write(f"💾 新最佳个体已保存: {best_path}")
                        last_best_fitness = stats['best_fitness']
                    except Exception as e:
                        tqdm.write(f"警告：保存最佳个体失败: {e}")
                
                # 检查早期停止（仅在有限代数训练时）
                if self.config.max_generations > 0:
                    if stats['best_fitness'] > self.best_fitness:
                        self.best_fitness = stats['best_fitness']
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    if no_improvement_count >= self.config.early_stop_patience:
                        tqdm.write(f"\n连续{self.config.early_stop_patience}代没有改进，提前停止。")
                        break
                
                # 保存检查点
                if save_checkpoints and checkpoint_dir and (gen + 1) % checkpoint_interval == 0:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = checkpoint_dir / f"checkpoint_gen_{gen+1}.pt"
                    self.save_checkpoint(str(checkpoint_path))
                
                gen += 1
                
        except KeyboardInterrupt:
            tqdm.write("\n🛑 用户中断训练")
            if self.config.max_generations == -1:
                tqdm.write("无限训练已停止")
        
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