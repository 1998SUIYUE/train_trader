"""
多目标优化器 - 帕累托前沿分析
Multi-Objective Optimizer - Pareto Front Analysis
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

class ObjectiveType(Enum):
    """目标类型枚举"""
    MAXIMIZE = "maximize"  # 最大化目标
    MINIMIZE = "minimize"  # 最小化目标

@dataclass
class ObjectiveConfig:
    """目标配置"""
    name: str
    objective_type: ObjectiveType
    weight: float = 1.0
    normalize: bool = True

@dataclass
class MultiObjectiveConfig:
    """多目标优化配置"""
    objectives: List[ObjectiveConfig]
    pareto_front_size: int = 100
    crowding_distance_weight: float = 0.5
    enable_hypervolume: bool = True
    reference_point: Optional[List[float]] = None

class MultiObjectiveOptimizer:
    """多目标优化器"""
    
    def __init__(self, config: MultiObjectiveConfig):
        """
        初始化多目标优化器
        
        Args:
            config: 多目标优化配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 目标函数映射
        self.objective_functions = {
            'total_return': self._calculate_total_return,
            'sharpe_ratio': self._calculate_sharpe_ratio,
            'max_drawdown': self._calculate_max_drawdown,
            'volatility': self._calculate_volatility,
            'win_rate': self._calculate_win_rate,
            'profit_factor': self._calculate_profit_factor,
            'calmar_ratio': self._calculate_calmar_ratio,
            'sortino_ratio': self._calculate_sortino_ratio,
            'trade_frequency': self._calculate_trade_frequency,
            'stability_score': self._calculate_stability_score,
        }
        
        # 验证目标配置
        for obj_config in self.config.objectives:
            if obj_config.name not in self.objective_functions:
                raise ValueError(f"未知的目标函数: {obj_config.name}")
        
        self.logger.info(f"多目标优化器初始化完成，目标数量: {len(self.config.objectives)}")
    
    def evaluate_all_objectives(self, signals: torch.Tensor, labels: torch.Tensor,
                               buy_thresholds: torch.Tensor, sell_thresholds: torch.Tensor,
                               stop_losses: torch.Tensor, max_positions: torch.Tensor,
                               max_drawdowns: torch.Tensor, trade_positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        评估所有目标函数
        
        Args:
            signals: 预测信号 [pop_size, n_samples]
            labels: 真实标签 [n_samples]
            buy_thresholds: 买入阈值 [pop_size]
            sell_thresholds: 卖出阈值 [pop_size]
            stop_losses: 止损比例 [pop_size]
            max_positions: 最大仓位 [pop_size]
            max_drawdowns: 最大回撤 [pop_size]
            trade_positions: 交易仓位 [pop_size]
            
        Returns:
            所有目标的评估结果
        """
        # 首先进行回测获取基础数据
        backtest_results = self._vectorized_backtest(
            signals, labels, buy_thresholds, sell_thresholds,
            stop_losses, max_positions, max_drawdowns, trade_positions
        )
        
        # 计算所有目标
        objectives = {}
        for obj_config in self.config.objectives:
            obj_values = self.objective_functions[obj_config.name](backtest_results)
            
            # 标准化处理
            if obj_config.normalize:
                obj_values = self._normalize_objective(obj_values)
            
            # 处理最小化目标（转换为最大化）
            if obj_config.objective_type == ObjectiveType.MINIMIZE:
                obj_values = -obj_values
            
            objectives[obj_config.name] = obj_values
        
        return objectives
    
    def _vectorized_backtest(self, signals: torch.Tensor, labels: torch.Tensor,
                            buy_thresholds: torch.Tensor, sell_thresholds: torch.Tensor,
                            stop_losses: torch.Tensor, max_positions: torch.Tensor,
                            max_drawdowns: torch.Tensor, trade_positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        向量化回测，返回详细的交易结果
        
        Returns:
            包含详细交易统计的字典
        """
        population_size, n_samples = signals.shape
        device = signals.device
        
        # 扩展维度
        buy_thresholds = buy_thresholds.unsqueeze(1)
        sell_thresholds = sell_thresholds.unsqueeze(1)
        stop_losses = stop_losses.unsqueeze(1)
        max_positions = max_positions.unsqueeze(1)
        trade_positions = trade_positions.unsqueeze(1)
        
        # 生成交易信号
        buy_signals = (signals > buy_thresholds).float()
        sell_signals = (signals < sell_thresholds).float()
        
        # 计算仓位（基于信号强度和交易仓位参数）
        signal_strength = torch.sigmoid((signals - 0.5) * 4)
        positions = signal_strength * max_positions * trade_positions
        
        # 计算每日收益
        period_returns = labels.unsqueeze(0).expand(population_size, -1)
        portfolio_returns = positions * period_returns
        
        # 计算累积组合价值
        portfolio_values = torch.cumprod(1 + portfolio_returns, dim=1)
        
        # 计算回撤
        running_max = torch.cummax(portfolio_values, dim=1)[0]
        drawdowns = (running_max - portfolio_values) / (running_max + 1e-8)
        
        # 计算交易次数（仓位变化）
        position_changes = torch.abs(torch.diff(positions, dim=1))
        trade_counts = torch.sum(position_changes > 0.01, dim=1)  # 仓位变化超过1%算一次交易
        
        # 计算盈利和亏损交易
        daily_pnl = torch.diff(portfolio_values, dim=1)
        winning_trades = daily_pnl > 0
        losing_trades = daily_pnl < 0
        
        # 计算连续交易统计
        win_streaks = self._calculate_streaks(winning_trades)
        loss_streaks = self._calculate_streaks(losing_trades)
        
        return {
            'portfolio_returns': portfolio_returns,
            'portfolio_values': portfolio_values,
            'drawdowns': drawdowns,
            'positions': positions,
            'trade_counts': trade_counts,
            'daily_pnl': daily_pnl,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_streaks': win_streaks,
            'loss_streaks': loss_streaks,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
        }
    
    def _calculate_total_return(self, backtest_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算总收益率"""
        portfolio_values = backtest_results['portfolio_values']
        return portfolio_values[:, -1] - 1.0  # 最终价值 - 初始价值(1.0)
    
    def _calculate_sharpe_ratio(self, backtest_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算夏普比率"""
        portfolio_returns = backtest_results['portfolio_returns']
        mean_returns = torch.mean(portfolio_returns, dim=1)
        std_returns = torch.std(portfolio_returns, dim=1) + 1e-8
        return mean_returns / std_returns
    
    def _calculate_max_drawdown(self, backtest_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算最大回撤"""
        drawdowns = backtest_results['drawdowns']
        return torch.max(drawdowns, dim=1)[0]
    
    def _calculate_volatility(self, backtest_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算波动率"""
        portfolio_returns = backtest_results['portfolio_returns']
        return torch.std(portfolio_returns, dim=1)
    
    def _calculate_win_rate(self, backtest_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算胜率"""
        winning_trades = backtest_results['winning_trades']
        total_trades = winning_trades.shape[1]
        return torch.sum(winning_trades, dim=1).float() / total_trades
    
    def _calculate_profit_factor(self, backtest_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算盈亏比"""
        daily_pnl = backtest_results['daily_pnl']
        
        gross_profit = torch.sum(torch.clamp(daily_pnl, min=0), dim=1)
        gross_loss = torch.sum(torch.clamp(daily_pnl, max=0), dim=1)
        
        # 避免除零
        profit_factor = torch.where(
            torch.abs(gross_loss) > 1e-8,
            gross_profit / torch.abs(gross_loss),
            torch.ones_like(gross_profit)
        )
        
        return profit_factor
    
    def _calculate_calmar_ratio(self, backtest_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算卡玛比率（年化收益率/最大回撤）"""
        total_return = self._calculate_total_return(backtest_results)
        max_drawdown = self._calculate_max_drawdown(backtest_results)
        
        # 假设数据是日频，年化收益率
        annualized_return = total_return * 252 / backtest_results['portfolio_returns'].shape[1]
        
        calmar_ratio = torch.where(
            max_drawdown > 1e-8,
            annualized_return / max_drawdown,
            torch.zeros_like(annualized_return)
        )
        
        return calmar_ratio
    
    def _calculate_sortino_ratio(self, backtest_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算索提诺比率（考虑下行风险的夏普比率）"""
        portfolio_returns = backtest_results['portfolio_returns']
        mean_returns = torch.mean(portfolio_returns, dim=1)
        
        # 只考虑负收益的标准差
        negative_returns = torch.clamp(portfolio_returns, max=0)
        downside_std = torch.std(negative_returns, dim=1) + 1e-8
        
        return mean_returns / downside_std
    
    def _calculate_trade_frequency(self, backtest_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算交易频率"""
        trade_counts = backtest_results['trade_counts']
        total_periods = backtest_results['portfolio_returns'].shape[1]
        return trade_counts.float() / total_periods
    
    def _calculate_stability_score(self, backtest_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算稳定性得分（基于收益的一致性）"""
        portfolio_returns = backtest_results['portfolio_returns']
        
        # 计算滚动收益的标准差
        window_size = min(20, portfolio_returns.shape[1] // 5)
        rolling_stds = []
        
        for i in range(window_size, portfolio_returns.shape[1]):
            window_returns = portfolio_returns[:, i-window_size:i]
            rolling_std = torch.std(window_returns, dim=1)
            rolling_stds.append(rolling_std)
        
        if rolling_stds:
            rolling_stds = torch.stack(rolling_stds, dim=1)
            stability_score = 1.0 / (torch.mean(rolling_stds, dim=1) + 1e-8)
        else:
            stability_score = torch.ones(portfolio_returns.shape[0], device=portfolio_returns.device)
        
        return stability_score
    
    def _calculate_streaks(self, binary_series: torch.Tensor) -> torch.Tensor:
        """计算连续序列的最大长度"""
        # 这是一个简化版本，返回平均连续长度
        diff = torch.diff(binary_series.float(), dim=1)
        # 简化处理：返回True的比例作为连续性指标
        return torch.mean(binary_series.float(), dim=1)
    
    def _normalize_objective(self, values: torch.Tensor) -> torch.Tensor:
        """标准化目标值到[0, 1]范围"""
        min_val = torch.min(values)
        max_val = torch.max(values)
        
        if max_val - min_val < 1e-8:
            return torch.ones_like(values) * 0.5
        
        return (values - min_val) / (max_val - min_val)
    
    def calculate_pareto_front(self, objectives: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算帕累托前沿（带超时保护）
        
        Args:
            objectives: 目标值字典
            
        Returns:
            帕累托前沿的索引, 支配等级
        """
        try:
            # 将所有目标组合成矩阵 [population_size, n_objectives]
            obj_names = list(objectives.keys())
            obj_matrix = torch.stack([objectives[name] for name in obj_names], dim=1)
            
            population_size = obj_matrix.shape[0]
            
            # 对于大种群，使用快速近似算法
            if population_size > 1000:
                return self._fast_pareto_approximation(obj_matrix)
            
            # 计算支配关系（带超时保护）
            domination_counts = torch.zeros(population_size, device=obj_matrix.device)
            dominated_solutions = [[] for _ in range(population_size)]
            
            # 限制比较次数，避免O(n²)复杂度导致的卡死
            max_comparisons = min(population_size * population_size, 100000)
            comparison_count = 0
            
            for i in range(population_size):
                for j in range(i + 1, population_size):  # 只比较上三角，减少计算量
                    comparison_count += 1
                    if comparison_count > max_comparisons:
                        self.logger.warning(f"帕累托前沿计算达到最大比较次数限制: {max_comparisons}")
                        break
                    
                    if self._dominates(obj_matrix[i], obj_matrix[j]):
                        dominated_solutions[i].append(j)
                        domination_counts[j] += 1
                    elif self._dominates(obj_matrix[j], obj_matrix[i]):
                        dominated_solutions[j].append(i)
                        domination_counts[i] += 1
                
                if comparison_count > max_comparisons:
                    break
            
            # 找到帕累托前沿（支配计数为0的解）
            pareto_front = torch.where(domination_counts == 0)[0]
            
            # 如果没有找到帕累托前沿或前沿为空，返回最佳的几个解
            if len(pareto_front) == 0:
                # 选择第一个目标最好的解作为备用
                first_obj = obj_matrix[:, 0]
                _, best_indices = torch.topk(first_obj, min(self.config.pareto_front_size, population_size))
                pareto_front = best_indices
                self.logger.warning("未找到有效的帕累托前沿，使用第一个目标的最佳解")
            
            # 如果帕累托前沿太大，使用拥挤距离进行筛选
            elif len(pareto_front) > self.config.pareto_front_size:
                try:
                    crowding_distances = self._calculate_crowding_distance(
                        obj_matrix[pareto_front]
                    )
                    # 选择拥挤距离最大的解
                    _, selected_indices = torch.topk(
                        crowding_distances, 
                        self.config.pareto_front_size
                    )
                    pareto_front = pareto_front[selected_indices]
                except Exception as e:
                    self.logger.warning(f"拥挤距离计算失败: {e}，使用随机选择")
                    # 随机选择（确保设备一致性）
                    indices = torch.randperm(len(pareto_front), device=obj_matrix.device)[:self.config.pareto_front_size]
                    pareto_front = pareto_front[indices]
            
            return pareto_front, domination_counts
            
        except Exception as e:
            self.logger.error(f"帕累托前沿计算失败: {e}")
            # 返回随机选择的解作为备用
            population_size = len(list(objectives.values())[0])
            device = list(objectives.values())[0].device
            random_indices = torch.randperm(population_size, device=device)[:min(self.config.pareto_front_size, population_size)]
            domination_counts = torch.zeros(population_size, device=device)
            return random_indices, domination_counts
    
    def _fast_pareto_approximation(self, obj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """快速帕累托前沿近似算法（用于大种群）"""
        population_size = obj_matrix.shape[0]
        device = obj_matrix.device
        
        # 随机采样减少计算量（确保在同一设备上）
        sample_size = min(500, population_size)
        sample_indices = torch.randperm(population_size, device=device)[:sample_size]
        sample_matrix = obj_matrix[sample_indices]
        
        # 在采样中计算帕累托前沿
        domination_counts = torch.zeros(sample_size, device=device)
        
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                if self._dominates(sample_matrix[i], sample_matrix[j]):
                    domination_counts[j] += 1
                elif self._dominates(sample_matrix[j], sample_matrix[i]):
                    domination_counts[i] += 1
        
        # 找到采样中的帕累托前沿
        sample_pareto = torch.where(domination_counts == 0)[0]
        
        if len(sample_pareto) == 0:
            # 如果没有找到，选择第一个目标最好的
            first_obj = sample_matrix[:, 0]
            _, best_idx = torch.topk(first_obj, 1)
            sample_pareto = best_idx
        
        # 映射回原始索引（确保设备一致性）
        pareto_front = sample_indices[sample_pareto]
        
        # 扩展到目标大小
        if len(pareto_front) < self.config.pareto_front_size:
            # 添加更多随机解（确保在同一设备上）
            remaining = self.config.pareto_front_size - len(pareto_front)
            remaining_indices = torch.randperm(population_size, device=device)[:remaining]
            pareto_front = torch.cat([pareto_front, remaining_indices])
        
        # 创建全种群的支配计数（简化版）
        full_domination_counts = torch.zeros(population_size, device=device)
        
        return pareto_front[:self.config.pareto_front_size], full_domination_counts
    
    def _dominates(self, solution1: torch.Tensor, solution2: torch.Tensor) -> bool:
        """判断solution1是否支配solution2"""
        # 所有目标都不劣于solution2，且至少一个目标严格优于solution2
        better_or_equal = torch.all(solution1 >= solution2)
        strictly_better = torch.any(solution1 > solution2)
        return better_or_equal and strictly_better
    
    def _calculate_crowding_distance(self, front_objectives: torch.Tensor) -> torch.Tensor:
        """计算拥挤距离（带错误保护）"""
        try:
            n_solutions, n_objectives = front_objectives.shape
            device = front_objectives.device
            crowding_distances = torch.zeros(n_solutions, device=device)
            
            # 如果只有一个解，返回最大距离
            if n_solutions <= 1:
                return torch.full((n_solutions,), float('inf'), device=device)
            
            for obj_idx in range(n_objectives):
                # 按当前目标排序
                obj_values = front_objectives[:, obj_idx]
                sorted_indices = torch.argsort(obj_values)
                
                # 边界解设置为无穷大
                if n_solutions >= 2:
                    crowding_distances[sorted_indices[0]] = float('inf')
                    crowding_distances[sorted_indices[-1]] = float('inf')
                
                # 计算中间解的拥挤距离
                obj_range = obj_values.max() - obj_values.min()
                if obj_range > 1e-8 and n_solutions > 2:
                    for i in range(1, n_solutions - 1):
                        distance = (obj_values[sorted_indices[i + 1]] - 
                                   obj_values[sorted_indices[i - 1]]) / obj_range
                        crowding_distances[sorted_indices[i]] += distance
            
            return crowding_distances
            
        except Exception as e:
            self.logger.warning(f"拥挤距离计算失败: {e}")
            # 返回随机距离
            n_solutions = front_objectives.shape[0]
            device = front_objectives.device
            return torch.rand(n_solutions, device=device)
    
    def nsga2_selection(self, population: torch.Tensor, objectives: Dict[str, torch.Tensor],
                       selection_size: int) -> torch.Tensor:
        """
        NSGA-II选择算法
        
        Args:
            population: 种群 [population_size, individual_size]
            objectives: 目标值
            selection_size: 选择的个体数量
            
        Returns:
            选择的个体
        """
        population_size = population.shape[0]
        
        # 计算所有个体的支配等级
        obj_names = list(objectives.keys())
        obj_matrix = torch.stack([objectives[name] for name in obj_names], dim=1)
        
        # 非支配排序
        fronts = self._fast_non_dominated_sort(obj_matrix)
        
        # 选择个体
        selected_indices = []
        front_idx = 0
        
        while len(selected_indices) < selection_size and front_idx < len(fronts):
            current_front = fronts[front_idx]
            
            if len(selected_indices) + len(current_front) <= selection_size:
                # 整个前沿都可以选择
                selected_indices.extend(current_front)
            else:
                # 需要从当前前沿中选择一部分
                remaining_slots = selection_size - len(selected_indices)
                front_objectives = obj_matrix[current_front]
                crowding_distances = self._calculate_crowding_distance(front_objectives)
                
                # 选择拥挤距离最大的个体
                _, best_indices = torch.topk(crowding_distances, remaining_slots)
                selected_indices.extend([current_front[i] for i in best_indices])
            
            front_idx += 1
        
        return population[selected_indices]
    
    def _fast_non_dominated_sort(self, objectives: torch.Tensor) -> List[List[int]]:
        """快速非支配排序"""
        population_size = objectives.shape[0]
        
        domination_counts = torch.zeros(population_size)
        dominated_solutions = [[] for _ in range(population_size)]
        
        # 计算支配关系
        for i in range(population_size):
            for j in range(population_size):
                if i != j:
                    if self._dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives[j], objectives[i]):
                        domination_counts[i] += 1
        
        # 构建前沿
        fronts = []
        current_front = []
        
        # 第一前沿：支配计数为0的解
        for i in range(population_size):
            if domination_counts[i] == 0:
                current_front.append(i)
        
        fronts.append(current_front)
        
        # 构建后续前沿
        while len(current_front) > 0:
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            
            if len(next_front) > 0:
                fronts.append(next_front)
            current_front = next_front
        
        return fronts
    
    def calculate_hypervolume(self, pareto_front_objectives: torch.Tensor) -> float:
        """计算超体积指标（简化版本）"""
        if not self.config.enable_hypervolume:
            return 0.0
        
        # 这是一个简化的超体积计算
        # 实际应用中可能需要更复杂的算法
        if self.config.reference_point is None:
            # 使用最小值作为参考点
            reference_point = torch.min(pareto_front_objectives, dim=0)[0] - 0.1
        else:
            reference_point = torch.tensor(self.config.reference_point, 
                                         device=pareto_front_objectives.device)
        
        # 简化计算：所有解的体积之和
        volumes = torch.prod(pareto_front_objectives - reference_point, dim=1)
        return torch.sum(torch.clamp(volumes, min=0)).item()
    
    def get_optimization_summary(self, objectives: Dict[str, torch.Tensor]) -> Dict:
        """获取优化总结"""
        pareto_front, domination_counts = self.calculate_pareto_front(objectives)
        
        summary = {
            'pareto_front_size': len(pareto_front),
            'total_population': len(domination_counts),
            'pareto_ratio': len(pareto_front) / len(domination_counts),
            'objective_stats': {}
        }
        
        # 计算每个目标的统计信息
        for obj_name, obj_values in objectives.items():
            pareto_values = obj_values[pareto_front]
            summary['objective_stats'][obj_name] = {
                'mean': torch.mean(obj_values).item(),
                'std': torch.std(obj_values).item(),
                'min': torch.min(obj_values).item(),
                'max': torch.max(obj_values).item(),
                'pareto_mean': torch.mean(pareto_values).item(),
                'pareto_std': torch.std(pareto_values).item(),
            }
        
        # 计算超体积
        if self.config.enable_hypervolume and len(pareto_front) > 0:
            obj_matrix = torch.stack([objectives[name] for name in objectives.keys()], dim=1)
            pareto_objectives = obj_matrix[pareto_front]
            summary['hypervolume'] = self.calculate_hypervolume(pareto_objectives)
        
        return summary


if __name__ == "__main__":
    # 测试多目标优化器
    print("=== 多目标优化器测试 ===")
    
    # 创建测试配置
    objectives_config = [
        ObjectiveConfig("sharpe_ratio", ObjectiveType.MAXIMIZE, weight=1.0),
        ObjectiveConfig("max_drawdown", ObjectiveType.MINIMIZE, weight=1.0),
        ObjectiveConfig("total_return", ObjectiveType.MAXIMIZE, weight=1.0),
        ObjectiveConfig("win_rate", ObjectiveType.MAXIMIZE, weight=0.5),
    ]
    
    config = MultiObjectiveConfig(
        objectives=objectives_config,
        pareto_front_size=50,
        enable_hypervolume=True
    )
    
    optimizer = MultiObjectiveOptimizer(config)
    
    # 创建测试数据
    population_size = 200
    n_samples = 500
    
    signals = torch.sigmoid(torch.randn(population_size, n_samples))
    labels = torch.randn(n_samples) * 0.01
    buy_thresholds = torch.rand(population_size) * 0.3 + 0.5
    sell_thresholds = torch.rand(population_size) * 0.3 + 0.2
    stop_losses = torch.rand(population_size) * 0.06 + 0.02
    max_positions = torch.rand(population_size) * 0.5 + 0.5
    max_drawdowns = torch.rand(population_size) * 0.15 + 0.1
    trade_positions = torch.rand(population_size) * 0.8 + 0.2
    
    print(f"测试数据: 种群大小={population_size}, 样本数={n_samples}")
    
    # 评估所有目标
    objectives = optimizer.evaluate_all_objectives(
        signals, labels, buy_thresholds, sell_thresholds,
        stop_losses, max_positions, max_drawdowns, trade_positions
    )
    
    print(f"目标评估完成，目标数量: {len(objectives)}")
    
    # 计算帕累托前沿
    pareto_front, domination_counts = optimizer.calculate_pareto_front(objectives)
    print(f"帕累托前沿大小: {len(pareto_front)}")
    
    # 获取优化总结
    summary = optimizer.get_optimization_summary(objectives)
    print(f"优化总结:")
    print(f"  帕累托比例: {summary['pareto_ratio']:.3f}")
    print(f"  超体积: {summary.get('hypervolume', 'N/A')}")
    
    for obj_name, stats in summary['objective_stats'].items():
        print(f"  {obj_name}:")
        print(f"    总体: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
        print(f"    帕累托: 均值={stats['pareto_mean']:.4f}, 标准差={stats['pareto_std']:.4f}")
    
    print("\n=== 测试完成 ===")