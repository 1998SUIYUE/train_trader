"""
CUDA版GPU加速遗传算法实现
支持NVIDIA GPU CUDA加速
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import time
import json
from pathlib import Path
import gc

from cuda_gpu_utils import CudaGPUManager

try:
    from cuda_backtest_optimizer import CudaBacktestOptimizer
    BACKTEST_OPTIMIZER_AVAILABLE = True
except ImportError:
    BACKTEST_OPTIMIZER_AVAILABLE = False

try:
    from training_progress_monitor import TrainingProgressMonitor, SimpleProgressDisplay
    PROGRESS_MONITOR_AVAILABLE = True
except ImportError:
    PROGRESS_MONITOR_AVAILABLE = False


@dataclass
class CudaGAConfig:
    """CUDA遗传算法配置"""
    # 基本参数
    population_size: int = 1000
    max_generations: int = 100
    mutation_rate: float = 0.01
    crossover_rate: float = 0.8
    elite_ratio: float = 0.1
    feature_dim: int = 1400
    
    # 交易策略参数
    buy_threshold: float = 0.6
    sell_threshold: float = 0.4
    
    # 风险管理参数
    stop_loss: float = 0.05
    max_position: float = 1.0
    max_drawdown: float = 0.2
    
    # 适应度函数权重
    sharpe_weight: float = 0.5
    drawdown_weight: float = 0.3
    stability_weight: float = 0.2
    
    # GPU优化参数
    batch_size: int = 1000
    early_stop_patience: int = 50
    use_torch_scan: bool = True
    
    def __post_init__(self):
        """验证配置参数"""
        assert self.population_size > 0, "种群大小必须大于0"
        assert 0 < self.mutation_rate < 1, "变异率必须在(0,1)之间"
        assert 0 < self.crossover_rate < 1, "交叉率必须在(0,1)之间"
        assert 0 < self.elite_ratio < 1, "精英比例必须在(0,1)之间"
        assert abs(self.sharpe_weight + self.drawdown_weight + self.stability_weight - 1.0) < 1e-6, \
            "适应度权重之和必须等于1.0"


class CudaGPUAcceleratedGA:
    """CUDA GPU加速遗传算法"""
    
    def __init__(self, config: CudaGAConfig, gpu_manager: CudaGPUManager):
        """
        初始化遗传算法
        
        Args:
            config: 算法配置
            gpu_manager: GPU管理器
        """
        self.config = config
        self.gpu_manager = gpu_manager
        self.device = gpu_manager.device
        
        # 算法状态
        self.generation = 0
        self.best_fitness = -float('inf')
        self.best_individual = None
        self.fitness_history = []
        self.no_improvement_count = 0
        
        # GPU张量
        self.population = None
        self.fitness_scores = None
        
        # 初始化回测优化器
        if BACKTEST_OPTIMIZER_AVAILABLE:
            self.backtest_optimizer = CudaBacktestOptimizer(self.device)
            print("CUDA回测优化器已启用")
        else:
            self.backtest_optimizer = None
            print("使用内置回测方法")
        
        # 初始化进度监控器
        self.progress_monitor = None
        self.use_detailed_progress = True
        
        print(f"CudaGPUAcceleratedGA初始化完成")
        print(f"设备: {self.device}")
        print(f"种群大小: {config.population_size}")
        print(f"特征维度: {config.feature_dim}")
    
    def initialize_population(self, seed: Optional[int] = None) -> None:
        """
        初始化种群
        
        Args:
            seed: 随机种子
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        print("初始化种群...")
        
        # 在GPU上创建种群
        # 每个个体包含: [权重(feature_dim), 偏置, 买入阈值, 卖出阈值, 止损, 最大仓位]
        individual_size = self.config.feature_dim + 5
        
        self.population = torch.randn(
            self.config.population_size, 
            individual_size,
            device=self.device,
            dtype=torch.float32
        )
        
        # 初始化权重部分 (前feature_dim个参数)
        self.population[:, :self.config.feature_dim] *= 0.1
        
        # 初始化其他参数
        self.population[:, self.config.feature_dim] = torch.randn(self.config.population_size, device=self.device) * 0.1  # 偏置
        self.population[:, self.config.feature_dim + 1] = torch.sigmoid(torch.randn(self.config.population_size, device=self.device)) * 0.3 + 0.5  # 买入阈值 [0.5, 0.8]
        self.population[:, self.config.feature_dim + 2] = torch.sigmoid(torch.randn(self.config.population_size, device=self.device)) * 0.3 + 0.2  # 卖出阈值 [0.2, 0.5]
        self.population[:, self.config.feature_dim + 3] = torch.sigmoid(torch.randn(self.config.population_size, device=self.device)) * 0.08 + 0.02  # 止损 [0.02, 0.1]
        self.population[:, self.config.feature_dim + 4] = torch.sigmoid(torch.randn(self.config.population_size, device=self.device)) * 0.8 + 0.2  # 最大仓位 [0.2, 1.0]
        
        # 初始化适应度分数
        self.fitness_scores = torch.zeros(self.config.population_size, device=self.device)
        
        print(f"种群初始化完成: {self.population.shape}")
    
    def evaluate_fitness_batch(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        批量评估种群适应度
        
        Args:
            features: 特征数据 [n_samples, feature_dim]
            labels: 标签数据 [n_samples]
            
        Returns:
            适应度分数 [population_size]
        """
        n_samples = features.shape[0]
        population_size = self.population.shape[0]
        
        # 提取个体参数
        weights = self.population[:, :self.config.feature_dim]  # [pop_size, feature_dim]
        biases = self.population[:, self.config.feature_dim]    # [pop_size]
        buy_thresholds = self.population[:, self.config.feature_dim + 1]   # [pop_size]
        sell_thresholds = self.population[:, self.config.feature_dim + 2]  # [pop_size]
        stop_losses = self.population[:, self.config.feature_dim + 3]      # [pop_size]
        max_positions = self.population[:, self.config.feature_dim + 4]    # [pop_size]
        
        # 计算预测信号 [pop_size, n_samples]
        signals = torch.sigmoid(torch.matmul(weights, features.T) + biases.unsqueeze(1))
        
        # 使用CUDA优化的向量化回测
        if self.backtest_optimizer is not None:
            # 使用专门的CUDA回测优化器
            if self.config.use_torch_scan:
                # 高精度模式
                fitness_scores = self.backtest_optimizer.vectorized_backtest_v3(
                    signals, labels, buy_thresholds, sell_thresholds, 
                    max_positions, stop_losses
                )
            else:
                # 高速模式
                fitness_scores = self.backtest_optimizer.vectorized_backtest_v2(
                    signals, labels, buy_thresholds, sell_thresholds, max_positions
                )
        else:
            # 使用内置回测方法
            if self.config.use_torch_scan:
                fitness_scores = self._advanced_vectorized_backtest(
                    signals, labels, buy_thresholds, sell_thresholds, 
                    stop_losses, max_positions
                )
            else:
                fitness_scores = self._vectorized_backtest(
                    signals, labels, buy_thresholds, sell_thresholds,
                    stop_losses, max_positions
                )
        
        return fitness_scores
    
    def _backtest_with_scan(self, signals: torch.Tensor, labels: torch.Tensor,
                           buy_thresholds: torch.Tensor, sell_thresholds: torch.Tensor,
                           stop_losses: torch.Tensor, max_positions: torch.Tensor) -> torch.Tensor:
        """使用torch.scan的优化回测"""
        population_size, n_samples = signals.shape
        
        # 初始状态
        initial_state = {
            'position': torch.zeros(population_size, device=self.device),
            'cash': torch.ones(population_size, device=self.device),
            'portfolio_value': torch.ones(population_size, device=self.device),
            'entry_price': torch.zeros(population_size, device=self.device),
            'returns': torch.zeros(population_size, device=self.device),
            'trade_count': torch.zeros(population_size, device=self.device),
            'max_portfolio': torch.ones(population_size, device=self.device),
        }
        
        def scan_fn(state, inputs):
            signal, price_return = inputs
            
            # 当前价格变化
            current_return = price_return
            
            # 更新持仓价值
            new_portfolio_value = state['cash'] + state['position'] * (1 + current_return)
            
            # 交易决策
            buy_signal = (signal > buy_thresholds.unsqueeze(1)) & (state['position'] == 0)
            sell_signal = (signal < sell_thresholds.unsqueeze(1)) & (state['position'] > 0)
            
            # 止损检查
            if torch.any(state['position'] > 0):
                current_loss = (current_return - state['entry_price']) / state['entry_price']
                stop_loss_signal = (current_loss < -stop_losses.unsqueeze(1)) & (state['position'] > 0)
                sell_signal = sell_signal | stop_loss_signal
            
            # 执行买入
            new_position = torch.where(
                buy_signal.squeeze(),
                torch.clamp(max_positions * new_portfolio_value, 0, new_portfolio_value),
                state['position']
            )
            new_cash = torch.where(
                buy_signal.squeeze(),
                new_portfolio_value - new_position,
                state['cash']
            )
            new_entry_price = torch.where(
                buy_signal.squeeze(),
                torch.zeros_like(state['entry_price']),  # 以当前价格买入
                state['entry_price']
            )
            
            # 执行卖出
            new_cash = torch.where(
                sell_signal.squeeze(),
                new_cash + new_position * (1 + current_return),
                new_cash
            )
            new_position = torch.where(
                sell_signal.squeeze(),
                torch.zeros_like(new_position),
                new_position
            )
            
            # 更新统计
            new_trade_count = state['trade_count'] + buy_signal.squeeze().float() + sell_signal.squeeze().float()
            new_max_portfolio = torch.maximum(state['max_portfolio'], new_portfolio_value)
            
            # 计算收益
            new_returns = (new_portfolio_value - 1.0)
            
            new_state = {
                'position': new_position,
                'cash': new_cash,
                'portfolio_value': new_portfolio_value,
                'entry_price': new_entry_price,
                'returns': new_returns,
                'trade_count': new_trade_count,
                'max_portfolio': new_max_portfolio,
            }
            
            return new_state, new_returns
        
        # 准备输入数据
        inputs = (signals.T.unsqueeze(-1), labels.unsqueeze(0).unsqueeze(-1))  # [n_samples, pop_size, 1]
        
        # torch.func.scan在某些PyTorch版本中不可用，直接使用优化的传统方法
        return self._backtest_traditional(signals, labels, buy_thresholds, sell_thresholds, stop_losses, max_positions)
        
        # 综合适应度
        fitness = (self.config.sharpe_weight * sharpe_ratios - 
                  self.config.drawdown_weight * drawdowns +
                  self.config.stability_weight * normalized_trade_counts)
        
        return fitness
    
    def _backtest_traditional(self, signals: torch.Tensor, labels: torch.Tensor,
                             buy_thresholds: torch.Tensor, sell_thresholds: torch.Tensor,
                             stop_losses: torch.Tensor, max_positions: torch.Tensor) -> torch.Tensor:
        """CUDA优化的向量化回测方法"""
        population_size, n_samples = signals.shape
        
        # 使用向量化操作进行批量回测，避免循环
        return self._vectorized_backtest(signals, labels, buy_thresholds, sell_thresholds, stop_losses, max_positions)
    
    def _vectorized_backtest(self, signals: torch.Tensor, labels: torch.Tensor,
                           buy_thresholds: torch.Tensor, sell_thresholds: torch.Tensor,
                           stop_losses: torch.Tensor, max_positions: torch.Tensor) -> torch.Tensor:
        """完全向量化的CUDA回测实现"""
        population_size, n_samples = signals.shape
        
        # 扩展阈值维度以匹配信号
        buy_thresholds = buy_thresholds.unsqueeze(1)  # [pop_size, 1]
        sell_thresholds = sell_thresholds.unsqueeze(1)  # [pop_size, 1]
        stop_losses = stop_losses.unsqueeze(1)  # [pop_size, 1]
        max_positions = max_positions.unsqueeze(1)  # [pop_size, 1]
        
        # 生成交易信号矩阵
        buy_signals = (signals > buy_thresholds).float()  # [pop_size, n_samples]
        sell_signals = (signals < sell_thresholds).float()  # [pop_size, n_samples]
        
        # 计算累积收益
        cumulative_returns = torch.cumprod(1 + labels.unsqueeze(0).expand(population_size, -1), dim=1)
        
        # 简化的交易模拟：使用信号强度作为权重
        signal_strength = torch.sigmoid((signals - 0.5) * 4)  # 将信号映射到[0,1]
        
        # 计算每个时间点的仓位（基于信号强度）
        positions = signal_strength * max_positions.squeeze(1).unsqueeze(1)
        
        # 计算每个时间点的收益
        period_returns = labels.unsqueeze(0).expand(population_size, -1)
        portfolio_returns = positions * period_returns
        
        # 计算累积组合价值
        portfolio_values = torch.cumprod(1 + portfolio_returns, dim=1)
        final_values = portfolio_values[:, -1]
        
        # 计算夏普比率
        returns_std = torch.std(portfolio_returns, dim=1) + 1e-8
        mean_returns = torch.mean(portfolio_returns, dim=1)
        sharpe_ratios = mean_returns / returns_std
        
        # 计算最大回撤
        running_max = torch.cummax(portfolio_values, dim=1)[0]
        drawdowns = (running_max - portfolio_values) / running_max
        max_drawdowns = torch.max(drawdowns, dim=1)[0]
        
        # 计算交易活跃度
        position_changes = torch.abs(torch.diff(positions, dim=1))
        trade_activity = torch.mean(position_changes, dim=1)
        normalized_activity = torch.clamp(trade_activity, 0.0, 1.0)
        
        # 综合适应度评分
        fitness = (self.config.sharpe_weight * sharpe_ratios - 
                  self.config.drawdown_weight * max_drawdowns +
                  self.config.stability_weight * normalized_activity)
        
        return fitness
    
    def _advanced_vectorized_backtest(self, signals: torch.Tensor, labels: torch.Tensor,
                                     buy_thresholds: torch.Tensor, sell_thresholds: torch.Tensor,
                                     stop_losses: torch.Tensor, max_positions: torch.Tensor) -> torch.Tensor:
        """高级CUDA向量化回测，更精确的交易模拟"""
        population_size, n_samples = signals.shape
        
        # 扩展维度
        buy_thresholds = buy_thresholds.unsqueeze(1)
        sell_thresholds = sell_thresholds.unsqueeze(1)
        stop_losses = stop_losses.unsqueeze(1)
        max_positions = max_positions.unsqueeze(1)
        
        # 生成交易决策矩阵
        buy_signals = (signals > buy_thresholds).float()
        sell_signals = (signals < sell_thresholds).float()
        
        # 初始化状态矩阵
        positions = torch.zeros(population_size, n_samples + 1, device=self.device)
        portfolio_values = torch.ones(population_size, n_samples + 1, device=self.device)
        
        # 向量化交易模拟
        for t in range(n_samples):
            current_return = labels[t]
            
            # 当前仓位状态
            current_positions = positions[:, t]
            current_values = portfolio_values[:, t]
            
            # 更新投资组合价值（考虑持仓收益）
            updated_values = current_values * (1 + current_positions * current_return)
            
            # 交易决策
            can_buy = (current_positions == 0) & (buy_signals[:, t] == 1)
            can_sell = (current_positions > 0) & (sell_signals[:, t] == 1)
            
            # 执行交易
            new_positions = current_positions.clone()
            
            # 买入：设置仓位
            buy_position = max_positions.squeeze(1) * can_buy.float()
            new_positions = torch.where(can_buy, buy_position, new_positions)
            
            # 卖出：清空仓位
            new_positions = torch.where(can_sell, torch.zeros_like(new_positions), new_positions)
            
            # 更新状态
            positions[:, t + 1] = new_positions
            portfolio_values[:, t + 1] = updated_values
        
        # 计算性能指标
        final_values = portfolio_values[:, -1]
        
        # 计算收益序列
        returns = torch.diff(portfolio_values, dim=1) / portfolio_values[:, :-1]
        
        # 夏普比率
        mean_returns = torch.mean(returns, dim=1)
        std_returns = torch.std(returns, dim=1) + 1e-8
        sharpe_ratios = mean_returns / std_returns
        
        # 最大回撤
        running_max = torch.cummax(portfolio_values, dim=1)[0]
        drawdowns = (running_max - portfolio_values) / (running_max + 1e-8)
        max_drawdowns = torch.max(drawdowns, dim=1)[0]
        
        # 交易频率
        position_changes = torch.abs(torch.diff(positions, dim=1))
        trade_frequency = torch.sum(position_changes > 0, dim=1).float() / n_samples
        normalized_frequency = torch.clamp(trade_frequency, 0.0, 1.0)
        
        # 综合适应度
        fitness = (self.config.sharpe_weight * sharpe_ratios - 
                  self.config.drawdown_weight * max_drawdowns +
                  self.config.stability_weight * normalized_frequency)
        
        return fitness
    
    def selection(self) -> torch.Tensor:
        """选择操作 - 锦标赛选择"""
        tournament_size = max(2, self.config.population_size // 20)
        selected_indices = torch.zeros(self.config.population_size, dtype=torch.long, device=self.device)
        
        for i in range(self.config.population_size):
            # 随机选择锦标赛参与者
            tournament_indices = torch.randint(
                0, self.config.population_size, (tournament_size,), device=self.device
            )
            tournament_fitness = self.fitness_scores[tournament_indices]
            
            # 选择最佳个体
            winner_idx = tournament_indices[torch.argmax(tournament_fitness)]
            selected_indices[i] = winner_idx
        
        return self.population[selected_indices]
    
    def crossover(self, parents: torch.Tensor) -> torch.Tensor:
        """交叉操作 - 均匀交叉"""
        population_size, individual_size = parents.shape
        offspring = parents.clone()
        
        # 随机配对
        pairs = torch.randperm(population_size, device=self.device).view(-1, 2)
        
        for i in range(0, len(pairs), 2):
            if i + 1 < len(pairs):
                parent1_idx, parent2_idx = pairs[i], pairs[i + 1]
                
                if torch.rand(1, device=self.device) < self.config.crossover_rate:
                    # 均匀交叉
                    mask = torch.rand(individual_size, device=self.device) < 0.5
                    
                    temp = offspring[parent1_idx].clone()
                    offspring[parent1_idx] = torch.where(mask, offspring[parent2_idx], offspring[parent1_idx])
                    offspring[parent2_idx] = torch.where(mask, temp, offspring[parent2_idx])
        
        return offspring
    
    def mutation(self, population: torch.Tensor) -> torch.Tensor:
        """变异操作"""
        population_size, individual_size = population.shape
        mutated = population.clone()
        
        # 权重变异
        weight_mask = torch.rand(population_size, self.config.feature_dim, device=self.device) < self.config.mutation_rate
        weight_noise = torch.randn(population_size, self.config.feature_dim, device=self.device) * 0.01
        mutated[:, :self.config.feature_dim] += weight_mask * weight_noise
        
        # 其他参数变异
        param_mask = torch.rand(population_size, 5, device=self.device) < self.config.mutation_rate
        param_noise = torch.randn(population_size, 5, device=self.device) * 0.01
        
        # 确保参数在合理范围内
        mutated[:, self.config.feature_dim:] += param_mask * param_noise
        mutated[:, self.config.feature_dim + 1] = torch.clamp(mutated[:, self.config.feature_dim + 1], 0.5, 0.8)  # 买入阈值
        mutated[:, self.config.feature_dim + 2] = torch.clamp(mutated[:, self.config.feature_dim + 2], 0.2, 0.5)  # 卖出阈值
        mutated[:, self.config.feature_dim + 3] = torch.clamp(mutated[:, self.config.feature_dim + 3], 0.02, 0.1)  # 止损
        mutated[:, self.config.feature_dim + 4] = torch.clamp(mutated[:, self.config.feature_dim + 4], 0.2, 1.0)  # 最大仓位
        
        return mutated
    
    def evolve_one_generation(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """进化一代"""
        start_time = time.time()
        
        # 评估适应度
        self.fitness_scores = self.evaluate_fitness_batch(features, labels)
        
        # 记录最佳个体
        best_idx = torch.argmax(self.fitness_scores)
        current_best_fitness = self.fitness_scores[best_idx].item()
        
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_individual = self.gpu_manager.to_cpu(self.population[best_idx])
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # 精英保留
        elite_size = int(self.config.population_size * self.config.elite_ratio)
        elite_indices = torch.topk(self.fitness_scores, elite_size).indices
        elite_population = self.population[elite_indices]
        
        # 选择、交叉、变异
        selected = self.selection()
        offspring = self.crossover(selected)
        mutated = self.mutation(offspring)
        
        # 新种群 = 精英 + 变异后代
        new_population = torch.cat([elite_population, mutated[elite_size:]], dim=0)
        self.population = new_population
        
        self.generation += 1
        generation_time = time.time() - start_time
        
        # 记录历史
        stats = {
            'generation': self.generation,
            'best_fitness': current_best_fitness,
            'avg_fitness': torch.mean(self.fitness_scores).item(),
            'std_fitness': torch.std(self.fitness_scores).item(),
            'generation_time': generation_time,
            'no_improvement_count': self.no_improvement_count
        }
        
        self.fitness_history.append(stats)
        
        return stats
    
    def evolve(self, features: torch.Tensor, labels: torch.Tensor,
               save_checkpoints: bool = True,
               checkpoint_dir: Optional[Path] = None,
               checkpoint_interval: int = 50,
               save_generation_results: bool = True,
               generation_log_file: Optional[Path] = None,
               generation_log_interval: int = 1,
               auto_save_best: bool = True,
               output_dir: Optional[Path] = None,
               show_detailed_progress: bool = True,
               progress_update_interval: float = 1.0) -> Dict[str, Any]:
        """
        主进化循环
        
        Args:
            features: 训练特征
            labels: 训练标签
            save_checkpoints: 是否保存检查点
            checkpoint_dir: 检查点目录
            checkpoint_interval: 检查点保存间隔
            save_generation_results: 是否保存每代结果
            generation_log_file: 日志文件路径
            generation_log_interval: 日志记录间隔
            auto_save_best: 是否自动保存最佳个体
            output_dir: 输出目录
            show_detailed_progress: 是否显示详细进度
            progress_update_interval: 进度更新间隔
            
        Returns:
            训练结果
        """
        # 初始化进度监控器
        if PROGRESS_MONITOR_AVAILABLE and show_detailed_progress:
            self.progress_monitor = TrainingProgressMonitor(
                log_file=generation_log_file,
                update_interval=progress_update_interval
            )
            self.progress_monitor.start_training(
                total_generations=self.config.max_generations,
                early_stop_patience=self.config.early_stop_patience
            )
        else:
            # 使用简化显示器
            self.progress_monitor = SimpleProgressDisplay()
            self.progress_monitor.start_training(self.config.max_generations)
        
        start_time = time.time()
        
        # 确保数据在GPU上
        features = self.gpu_manager.to_gpu(features)
        labels = self.gpu_manager.to_gpu(labels)
        
        try:
            while True:
                # 检查停止条件
                if self.config.max_generations > 0 and self.generation >= self.config.max_generations:
                    print(f"达到最大代数 {self.config.max_generations}，停止训练")
                    break
                
                if self.no_improvement_count >= self.config.early_stop_patience:
                    print(f"连续 {self.config.early_stop_patience} 代无改进，早停")
                    break
                
                # 进化一代
                stats = self.evolve_one_generation(features, labels)
                
                # 添加GPU内存信息到统计数据
                if torch.cuda.is_available():
                    stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1e9
                    stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1e9
                
                # 添加系统内存信息
                import psutil
                stats['system_memory_gb'] = psutil.virtual_memory().used / 1e9
                
                # 更新进度监控器
                if hasattr(self, 'progress_monitor') and self.progress_monitor:
                    self.progress_monitor.update_generation(self.generation, stats)
                
                # 保存日志
                if save_generation_results and generation_log_file and self.generation % generation_log_interval == 0:
                    with open(generation_log_file, 'a', encoding='utf-8') as f:
                        json.dump(stats, f, ensure_ascii=False)
                        f.write('\n')
                
                # 保存检查点
                if save_checkpoints and checkpoint_dir and self.generation % checkpoint_interval == 0:
                    checkpoint_path = checkpoint_dir / f"checkpoint_gen_{self.generation}.pt"
                    self.save_checkpoint(str(checkpoint_path))
                
                # 自动保存最佳个体
                if auto_save_best and output_dir and self.no_improvement_count == 0:
                    best_path = output_dir / f"best_individual_gen_{self.generation}.npy"
                    np.save(best_path, self.best_individual)
                
                # 定期清理GPU缓存
                if self.generation % 10 == 0:
                    self.gpu_manager.clear_cache()
        
        except KeyboardInterrupt:
            print("\n训练被用户中断")
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            raise
        
        total_time = time.time() - start_time
        
        # 显示最终总结
        final_results = {
            'best_fitness': self.best_fitness,
            'best_individual': self.best_individual,
            'final_generation': self.generation,
            'total_time': total_time,
            'fitness_history': self.fitness_history
        }
        
        if hasattr(self, 'progress_monitor') and self.progress_monitor:
            self.progress_monitor.display_final_summary(final_results)
        else:
            print(f"\n训练完成!")
            print(f"总代数: {self.generation}")
            print(f"最佳适应度: {self.best_fitness:.4f}")
            print(f"总训练时间: {total_time:.2f}秒")
        
        return final_results
    
    def save_checkpoint(self, filepath: str) -> None:
        """保存检查点"""
        checkpoint = {
            'generation': self.generation,
            'population': self.gpu_manager.to_cpu(self.population),
            'fitness_scores': self.gpu_manager.to_cpu(self.fitness_scores),
            'best_fitness': self.best_fitness,
            'best_individual': self.best_individual,
            'fitness_history': self.fitness_history,
            'no_improvement_count': self.no_improvement_count,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        print(f"检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.generation = checkpoint['generation']
        self.population = self.gpu_manager.to_gpu(checkpoint['population'])
        self.fitness_scores = self.gpu_manager.to_gpu(checkpoint['fitness_scores'])
        self.best_fitness = checkpoint['best_fitness']
        self.best_individual = checkpoint['best_individual']
        self.fitness_history = checkpoint['fitness_history']
        self.no_improvement_count = checkpoint['no_improvement_count']
        
        print(f"检查点已加载: {filepath}")
        print(f"恢复到第 {self.generation} 代，最佳适应度: {self.best_fitness:.4f}")


if __name__ == "__main__":
    # 测试CUDA遗传算法
    print("=== CUDA遗传算法测试 ===")
    
    from cuda_gpu_utils import get_cuda_gpu_manager
    
    # 初始化GPU管理器
    gpu_manager = get_cuda_gpu_manager()
    
    # 创建测试配置
    config = CudaGAConfig(
        population_size=100,
        max_generations=5,
        feature_dim=50,
        batch_size=500
    )
    
    # 创建测试数据
    n_samples = 1000
    features = np.random.randn(n_samples, config.feature_dim).astype(np.float32)
    labels = np.random.randn(n_samples).astype(np.float32) * 0.01  # 模拟价格变化
    
    print(f"测试数据: features {features.shape}, labels {labels.shape}")
    
    # 初始化遗传算法
    ga = CudaGPUAcceleratedGA(config, gpu_manager)
    ga.initialize_population(seed=42)
    
    # 运行测试
    start_time = time.time()
    results = ga.evolve(features, labels)
    test_time = time.time() - start_time
    
    print(f"\n测试完成!")
    print(f"最佳适应度: {results['best_fitness']:.4f}")
    print(f"总代数: {results['final_generation']}")
    print(f"测试时间: {test_time:.2f}秒")
    print("CUDA遗传算法测试完成！")