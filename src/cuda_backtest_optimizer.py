"""
CUDA回测优化器
专门针对CUDA环境优化的高性能回测实现
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any
import time

class CudaBacktestOptimizer:
    """CUDA优化的回测引擎"""
    
    def __init__(self, device: torch.device):
        """
        初始化CUDA回测优化器
        
        Args:
            device: CUDA设备
        """
        self.device = device
        self.enable_memory_optimization = True
        self.batch_size = 1000  # 批处理大小
        
    def vectorized_backtest_v1(self, signals: torch.Tensor, returns: torch.Tensor,
                              buy_thresholds: torch.Tensor, sell_thresholds: torch.Tensor,
                              max_positions: torch.Tensor) -> torch.Tensor:
        """
        版本1：简化向量化回测（最快）
        适用于大种群快速评估
        """
        population_size, n_samples = signals.shape
        
        # 扩展阈值维度
        buy_thresh = buy_thresholds.unsqueeze(1)
        sell_thresh = sell_thresholds.unsqueeze(1)
        max_pos = max_positions.unsqueeze(1)
        
        # 计算信号强度（连续值而非离散交易）
        signal_strength = torch.sigmoid((signals - 0.5) * 6)  # 增强信号敏感度
        
        # 动态仓位分配
        positions = signal_strength * max_pos.squeeze(1).unsqueeze(1)
        
        # 计算组合收益
        period_returns = returns.unsqueeze(0).expand(population_size, -1)
        portfolio_returns = positions * period_returns
        
        # 性能指标计算
        mean_returns = torch.mean(portfolio_returns, dim=1)
        std_returns = torch.std(portfolio_returns, dim=1) + 1e-8
        sharpe_ratios = mean_returns / std_returns
        
        # 最大回撤（简化计算）
        cumulative_returns = torch.cumsum(portfolio_returns, dim=1)
        running_max = torch.cummax(cumulative_returns, dim=1)[0]
        drawdowns = running_max - cumulative_returns
        max_drawdowns = torch.max(drawdowns, dim=1)[0]
        
        return sharpe_ratios - 0.5 * max_drawdowns
    
    def vectorized_backtest_v2(self, signals: torch.Tensor, returns: torch.Tensor,
                              buy_thresholds: torch.Tensor, sell_thresholds: torch.Tensor,
                              max_positions: torch.Tensor) -> torch.Tensor:
        """
        版本2：精确向量化回测（平衡速度和精度）
        包含更真实的交易逻辑
        """
        population_size, n_samples = signals.shape
        
        # 生成交易信号
        buy_signals = (signals > buy_thresholds.unsqueeze(1)).float()
        sell_signals = (signals < sell_thresholds.unsqueeze(1)).float()
        
        # 使用卷积操作模拟状态转换
        positions = self._simulate_position_changes(buy_signals, sell_signals, max_positions)
        
        # 计算组合价值
        portfolio_values = self._calculate_portfolio_values(positions, returns)
        
        # 性能指标
        sharpe_ratios = self._calculate_sharpe_ratios(portfolio_values)
        max_drawdowns = self._calculate_max_drawdowns(portfolio_values)
        trade_frequencies = self._calculate_trade_frequencies(positions)
        
        # 综合评分
        fitness = 0.6 * sharpe_ratios - 0.3 * max_drawdowns + 0.1 * trade_frequencies
        
        return fitness
    
    def vectorized_backtest_v3(self, signals: torch.Tensor, returns: torch.Tensor,
                              buy_thresholds: torch.Tensor, sell_thresholds: torch.Tensor,
                              max_positions: torch.Tensor, stop_losses: torch.Tensor) -> torch.Tensor:
        """
        版本3：完整向量化回测（最精确）
        包含止损、仓位管理等完整功能
        """
        population_size, n_samples = signals.shape
        
        # 初始化状态张量
        positions = torch.zeros(population_size, n_samples + 1, device=self.device)
        cash = torch.ones(population_size, n_samples + 1, device=self.device)
        entry_prices = torch.zeros(population_size, n_samples + 1, device=self.device)
        
        # 扩展参数维度
        buy_thresh = buy_thresholds.unsqueeze(1)
        sell_thresh = sell_thresholds.unsqueeze(1)
        max_pos = max_positions.unsqueeze(1)
        stop_loss = stop_losses.unsqueeze(1)
        
        # 逐步模拟（仍然向量化）
        for t in range(n_samples):
            current_return = returns[t]
            current_signal = signals[:, t]
            
            # 当前状态
            current_pos = positions[:, t]
            current_cash = cash[:, t]
            current_entry = entry_prices[:, t]
            
            # 更新持仓价值
            position_value = current_pos * (1 + current_return)
            total_value = current_cash + position_value
            
            # 交易决策
            buy_condition = (current_signal > buy_thresh.squeeze(1)) & (current_pos == 0)
            sell_condition = (current_signal < sell_thresh.squeeze(1)) & (current_pos > 0)
            
            # 止损条件
            if torch.any(current_pos > 0):
                loss_ratio = (current_return - current_entry) / (current_entry + 1e-8)
                stop_loss_condition = (loss_ratio < -stop_loss.squeeze(1)) & (current_pos > 0)
                sell_condition = sell_condition | stop_loss_condition
            
            # 执行交易
            new_pos = current_pos.clone()
            new_cash = current_cash.clone()
            new_entry = current_entry.clone()
            
            # 买入
            buy_amount = max_pos.squeeze(1) * total_value * buy_condition.float()
            new_pos = torch.where(buy_condition, buy_amount / (1 + current_return), new_pos)
            new_cash = torch.where(buy_condition, total_value - buy_amount, new_cash)
            new_entry = torch.where(buy_condition, torch.zeros_like(current_entry), new_entry)
            
            # 卖出
            sell_value = new_pos * (1 + current_return)
            new_cash = torch.where(sell_condition, new_cash + sell_value, new_cash)
            new_pos = torch.where(sell_condition, torch.zeros_like(new_pos), new_pos)
            
            # 更新状态
            positions[:, t + 1] = new_pos
            cash[:, t + 1] = new_cash
            entry_prices[:, t + 1] = new_entry
        
        # 计算最终组合价值
        final_positions = positions[:, 1:]
        final_cash = cash[:, 1:]
        
        portfolio_values = final_cash + final_positions * torch.cumprod(1 + returns.unsqueeze(0).expand(population_size, -1), dim=1)
        
        # 性能指标
        returns_series = torch.diff(portfolio_values, dim=1) / portfolio_values[:, :-1]
        
        # 夏普比率
        mean_returns = torch.mean(returns_series, dim=1)
        std_returns = torch.std(returns_series, dim=1) + 1e-8
        sharpe_ratios = mean_returns / std_returns
        
        # 最大回撤
        running_max = torch.cummax(portfolio_values, dim=1)[0]
        drawdowns = (running_max - portfolio_values) / (running_max + 1e-8)
        max_drawdowns = torch.max(drawdowns, dim=1)[0]
        
        # 交易次数
        position_changes = torch.abs(torch.diff(positions, dim=1))
        trade_counts = torch.sum(position_changes > 1e-6, dim=1).float()
        normalized_trades = torch.clamp(trade_counts / n_samples, 0, 1)
        
        # 综合评分
        fitness = 0.5 * sharpe_ratios - 0.3 * max_drawdowns + 0.2 * normalized_trades
        
        return fitness
    
    def _simulate_position_changes(self, buy_signals: torch.Tensor, sell_signals: torch.Tensor,
                                 max_positions: torch.Tensor) -> torch.Tensor:
        """模拟仓位变化"""
        population_size, n_samples = buy_signals.shape
        positions = torch.zeros(population_size, n_samples, device=self.device)
        
        # 使用状态机逻辑
        current_pos = torch.zeros(population_size, device=self.device)
        
        for t in range(n_samples):
            # 买入信号且当前无仓位
            buy_mask = buy_signals[:, t] & (current_pos == 0)
            # 卖出信号且当前有仓位
            sell_mask = sell_signals[:, t] & (current_pos > 0)
            
            # 更新仓位
            current_pos = torch.where(buy_mask, max_positions, current_pos)
            current_pos = torch.where(sell_mask, torch.zeros_like(current_pos), current_pos)
            
            positions[:, t] = current_pos
        
        return positions
    
    def _calculate_portfolio_values(self, positions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """计算组合价值"""
        population_size, n_samples = positions.shape
        
        # 计算每期收益
        period_returns = returns.unsqueeze(0).expand(population_size, -1)
        portfolio_returns = positions * period_returns
        
        # 累积组合价值
        portfolio_values = torch.cumprod(1 + portfolio_returns, dim=1)
        
        return portfolio_values
    
    def _calculate_sharpe_ratios(self, portfolio_values: torch.Tensor) -> torch.Tensor:
        """计算夏普比率"""
        returns = torch.diff(portfolio_values, dim=1) / portfolio_values[:, :-1]
        mean_returns = torch.mean(returns, dim=1)
        std_returns = torch.std(returns, dim=1) + 1e-8
        return mean_returns / std_returns
    
    def _calculate_max_drawdowns(self, portfolio_values: torch.Tensor) -> torch.Tensor:
        """计算最大回撤"""
        running_max = torch.cummax(portfolio_values, dim=1)[0]
        drawdowns = (running_max - portfolio_values) / running_max
        return torch.max(drawdowns, dim=1)[0]
    
    def _calculate_trade_frequencies(self, positions: torch.Tensor) -> torch.Tensor:
        """计算交易频率"""
        position_changes = torch.abs(torch.diff(positions, dim=1))
        trade_counts = torch.sum(position_changes > 1e-6, dim=1).float()
        return torch.clamp(trade_counts / positions.shape[1], 0, 1)
    
    def benchmark_methods(self, signals: torch.Tensor, returns: torch.Tensor,
                         buy_thresholds: torch.Tensor, sell_thresholds: torch.Tensor,
                         max_positions: torch.Tensor, stop_losses: torch.Tensor) -> Dict[str, Any]:
        """基准测试不同回测方法的性能"""
        
        methods = {
            'v1_simple': self.vectorized_backtest_v1,
            'v2_balanced': self.vectorized_backtest_v2,
            'v3_complete': self.vectorized_backtest_v3
        }
        
        results = {}
        
        for name, method in methods.items():
            # 预热GPU
            if name == 'v3_complete':
                _ = method(signals[:10], returns, buy_thresholds[:10], sell_thresholds[:10], 
                          max_positions[:10], stop_losses[:10])
            else:
                _ = method(signals[:10], returns, buy_thresholds[:10], sell_thresholds[:10], 
                          max_positions[:10])
            
            torch.cuda.synchronize()
            
            # 计时
            start_time = time.time()
            
            if name == 'v3_complete':
                fitness = method(signals, returns, buy_thresholds, sell_thresholds, 
                               max_positions, stop_losses)
            else:
                fitness = method(signals, returns, buy_thresholds, sell_thresholds, max_positions)
            
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            
            results[name] = {
                'time': elapsed_time,
                'fitness_mean': torch.mean(fitness).item(),
                'fitness_std': torch.std(fitness).item(),
                'throughput': signals.shape[0] / elapsed_time  # 个体/秒
            }
        
        return results


def test_cuda_backtest_optimizer():
    """测试CUDA回测优化器"""
    print("=== CUDA回测优化器测试 ===")
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过测试")
        return
    
    device = torch.device('cuda:0')
    optimizer = CudaBacktestOptimizer(device)
    
    # 创建测试数据
    population_size = 500
    n_samples = 1000
    
    print(f"测试配置: 种群{population_size}, 样本{n_samples}")
    
    # 生成模拟数据
    signals = torch.rand(population_size, n_samples, device=device)
    returns = torch.randn(n_samples, device=device) * 0.01
    buy_thresholds = torch.rand(population_size, device=device) * 0.3 + 0.5
    sell_thresholds = torch.rand(population_size, device=device) * 0.3 + 0.2
    max_positions = torch.rand(population_size, device=device) * 0.8 + 0.2
    stop_losses = torch.rand(population_size, device=device) * 0.05 + 0.02
    
    # 运行基准测试
    results = optimizer.benchmark_methods(
        signals, returns, buy_thresholds, sell_thresholds, max_positions, stop_losses
    )
    
    # 显示结果
    print("\n基准测试结果:")
    print("-" * 60)
    print(f"{'方法':<15} {'时间(s)':<10} {'吞吐量':<15} {'适应度均值':<15}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name:<15} {result['time']:<10.4f} {result['throughput']:<15.1f} {result['fitness_mean']:<15.6f}")
    
    print("-" * 60)
    
    # 推荐最佳方法
    best_method = min(results.keys(), key=lambda x: results[x]['time'])
    print(f"\n推荐方法: {best_method} (最快)")
    print(f"性能提升: {results['v3_complete']['time'] / results[best_method]['time']:.2f}x")


if __name__ == "__main__":
    test_cuda_backtest_optimizer()