
"""
CUDA回测优化器
专门针对CUDA环境优化的高性能回测实现
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any
import time
import torch.jit

# JIT编译的、独立的完整回测函数
# 这是将for-loop JIT优化的正确且高效的模式
@torch.jit.script
def _run_jit_backtest(signals: torch.Tensor, returns: torch.Tensor,
                        buy_thresholds: torch.Tensor, sell_thresholds: torch.Tensor,
                        max_positions: torch.Tensor, stop_losses: torch.Tensor, 
                        max_drawdowns: torch.Tensor) -> torch.Tensor:
    
    population_size, n_samples = signals.shape
    device = signals.device

    # 模拟价格序列
    prices = torch.cat([torch.ones(1, device=device), 1 + returns]).cumprod(0)

    # 初始化状态张量
    positions = torch.zeros(population_size, n_samples + 1, device=device)
    portfolio_values = torch.ones(population_size, n_samples + 1, device=device)
    entry_prices = torch.zeros(population_size, device=device)
    
    # 扩展参数维度以进行向量化操作
    buy_thresh = buy_thresholds.unsqueeze(1)
    sell_thresh = sell_thresholds.unsqueeze(1)
    max_pos = max_positions.unsqueeze(1)
    stop_loss_param = stop_losses.unsqueeze(1)
    max_dd_param = max_drawdowns.unsqueeze(1)

    # JIT编译的for循环
    for t in range(n_samples):
        current_price = prices[t + 1]
        current_positions = positions[:, t]
        current_values = portfolio_values[:, t]
        
        # 1. 更新当前投资组合价值
        updated_values = current_values * (1 + current_positions * returns[t])
        
        # 2. 计算当前回撤
        # 此处简化处理：直接与历史所有时点比较，虽然效率稍低但JIT兼容
        running_max = torch.max(portfolio_values[:, :t+1], dim=1)[0]
        current_running_max = torch.maximum(running_max, updated_values)
        current_drawdown = (current_running_max - updated_values) / (current_running_max + 1e-8)

        # 3. 止损决策
        stop_loss_triggered = (current_positions > 0) & (entry_prices > 0) & (current_price < entry_prices * (1 - stop_loss_param.squeeze(1)))

        # 4. 交易决策
        can_buy = (current_positions == 0) & (signals[:, t] > buy_thresh.squeeze(1)) & (current_drawdown < max_dd_param.squeeze(1))
        can_sell = (current_positions > 0) & ((signals[:, t] < sell_thresh.squeeze(1)) | stop_loss_triggered | (current_drawdown > max_dd_param.squeeze(1)))
        
        # 5. 执行交易并更新状态
        new_positions = current_positions.clone()
        new_positions = torch.where(can_buy, max_pos.squeeze(1), new_positions)
        new_positions = torch.where(can_sell, torch.zeros_like(new_positions), new_positions)
        
        entry_prices = torch.where(can_buy, current_price, entry_prices)
        entry_prices = torch.where(can_sell, torch.zeros_like(entry_prices), entry_prices)
        
        positions[:, t + 1] = new_positions
        portfolio_values[:, t + 1] = updated_values

    # --- 性能指标计算 ---
    # 计算最终组合价值
    final_portfolio_values = portfolio_values
    
    # 收益序列
    returns_series = torch.diff(final_portfolio_values, dim=1) / (final_portfolio_values[:, :-1] + 1e-8)
    
    # 夏普比率
    mean_returns = torch.mean(returns_series, dim=1)
    std_returns = torch.std(returns_series, dim=1) + 1e-8
    sharpe_ratios = mean_returns / std_returns
    
    # 最大回撤
    running_max_final = torch.cummax(final_portfolio_values, dim=1)[0]
    drawdowns_final = (running_max_final - final_portfolio_values) / (running_max_final + 1e-8)
    max_drawdowns_calc = torch.max(drawdowns_final, dim=1)[0]
    
    # 交易频率
    position_changes = torch.abs(torch.diff(positions, dim=1))
    trade_counts = torch.sum(position_changes > 1e-6, dim=1).float()
    normalized_trades = torch.clamp(trade_counts / n_samples, 0, 1)
    
    # 综合适应度
    fitness = 0.5 * sharpe_ratios - 0.3 * max_drawdowns_calc - 0.2 * normalized_trades
    
    return torch.nan_to_num(fitness, nan=-10.0, posinf=0.0, neginf=-10.0)

class CudaBacktestOptimizer:
    """CUDA优化的回测引擎"""
    
    def __init__(self, device: torch.device):
        self.device = device

    def vectorized_backtest_v1(self, signals: torch.Tensor, returns: torch.Tensor,
                              buy_thresholds: torch.Tensor, sell_thresholds: torch.Tensor,
                              max_positions: torch.Tensor) -> torch.Tensor:
        population_size, n_samples = signals.shape
        buy_thresh = buy_thresholds.unsqueeze(1)
        sell_thresh = sell_thresholds.unsqueeze(1)
        max_pos = max_positions.unsqueeze(1)
        signal_strength = torch.sigmoid((signals - 0.5) * 6)
        positions = signal_strength * max_pos
        period_returns = returns.unsqueeze(0).expand(population_size, -1)
        portfolio_returns = positions * period_returns
        mean_returns = torch.mean(portfolio_returns, dim=1)
        std_returns = torch.std(portfolio_returns, dim=1) + 1e-8
        sharpe_ratios = mean_returns / std_returns
        cumulative_returns = torch.cumsum(portfolio_returns, dim=1)
        running_max = torch.cummax(cumulative_returns, dim=1)[0]
        drawdowns = running_max - cumulative_returns
        max_drawdowns = torch.max(drawdowns, dim=1)[0]
        return sharpe_ratios - 0.5 * max_drawdowns

    def _simulate_position_changes(self, buy_signals: torch.Tensor, sell_signals: torch.Tensor,
                                 max_positions: torch.Tensor) -> torch.Tensor:
        population_size, n_samples = buy_signals.shape
        positions = torch.zeros(population_size, n_samples, device=self.device)
        current_pos = torch.zeros(population_size, device=self.device)
        for t in range(n_samples):
            buy_mask = (buy_signals[:, t] == 1) & (current_pos == 0)
            sell_mask = (sell_signals[:, t] == 1) & (current_pos > 0)
            current_pos = torch.where(buy_mask, max_positions, current_pos)
            current_pos = torch.where(sell_mask, torch.zeros_like(current_pos), current_pos)
            positions[:, t] = current_pos
        return positions

    def _calculate_portfolio_values(self, positions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        period_returns = returns.unsqueeze(0).expand(positions.shape[0], -1)
        portfolio_returns = positions * period_returns
        return torch.cumprod(1 + portfolio_returns, dim=1)

    def _calculate_sharpe_ratios(self, portfolio_values: torch.Tensor) -> torch.Tensor:
        returns = torch.diff(portfolio_values, dim=1) / (portfolio_values[:, :-1] + 1e-8)
        mean_returns = torch.mean(returns, dim=1)
        std_returns = torch.std(returns, dim=1) + 1e-8
        return mean_returns / std_returns

    def _calculate_max_drawdowns(self, portfolio_values: torch.Tensor) -> torch.Tensor:
        running_max = torch.cummax(portfolio_values, dim=1)[0]
        drawdowns = (running_max - portfolio_values) / running_max
        return torch.max(drawdowns, dim=1)[0]

    def _calculate_trade_frequencies(self, positions: torch.Tensor) -> torch.Tensor:
        position_changes = torch.abs(torch.diff(positions, dim=1))
        trade_counts = torch.sum(position_changes > 1e-6, dim=1).float()
        return torch.clamp(trade_counts / positions.shape[1], 0, 1)

    def vectorized_backtest_v2(self, signals: torch.Tensor, returns: torch.Tensor,
                              buy_thresholds: torch.Tensor, sell_thresholds: torch.Tensor,
                              max_positions: torch.Tensor) -> torch.Tensor:
        buy_signals = (signals > buy_thresholds.unsqueeze(1)).float()
        sell_signals = (signals < sell_thresholds.unsqueeze(1)).float()
        positions = self._simulate_position_changes(buy_signals, sell_signals, max_positions)
        portfolio_values = self._calculate_portfolio_values(positions, returns)
        sharpe_ratios = self._calculate_sharpe_ratios(portfolio_values)
        max_drawdowns = self._calculate_max_drawdowns(portfolio_values)
        trade_frequencies = self._calculate_trade_frequencies(positions)
        return 0.6 * sharpe_ratios - 0.3 * max_drawdowns + 0.1 * trade_frequencies

    def vectorized_backtest_v3(self, *args, **kwargs):
        """版本3：完整回测 (JIT for-loop)."""
        return _run_jit_backtest(*args, **kwargs)

    def vectorized_backtest_v4_scan_style(self, *args, **kwargs):
        """版本4：完整回测 (Scan风格的JIT实现).
        注意: JIT的最佳实践使其内部实现与v3相同。
        """
        return _run_jit_backtest(*args, **kwargs)

    def benchmark_methods(self, signals: torch.Tensor, returns: torch.Tensor,
                         buy_thresholds: torch.Tensor, sell_thresholds: torch.Tensor,
                         max_positions: torch.Tensor, stop_losses: torch.Tensor,
                         max_drawdowns: torch.Tensor) -> Dict[str, Any]:
        """基准测试不同回测方法的性能"""
        methods = {
            'v1_simple': self.vectorized_backtest_v1,
            'v2_balanced': self.vectorized_backtest_v2,
            'v3_jit_loop': self.vectorized_backtest_v3,
            'v4_jit_scan': self.vectorized_backtest_v4_scan_style
        }
        results = {}
        print("\nRunning benchmark...")
        for name, method in methods.items():
            # Prepare arguments
            if name in ['v1_simple', 'v2_balanced']:
                warmup_args = (signals[:10], returns, buy_thresholds[:10], sell_thresholds[:10], max_positions[:10])
                full_args = (signals, returns, buy_thresholds, sell_thresholds, max_positions)
            else:
                warmup_args = (signals[:10], returns, buy_thresholds[:10], sell_thresholds[:10], max_positions[:10], stop_losses[:10], max_drawdowns[:10])
                full_args = (signals, returns, buy_thresholds, sell_thresholds, max_positions, stop_losses, max_drawdowns)
            
            # Warmup
            try:
                _ = method(*warmup_args)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"Warmup for {name} failed: {e}")
                continue

            # Benchmark
            start_time = time.time()
            try:
                fitness = method(*full_args)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                results[name] = {
                    'time': elapsed_time,
                    'fitness_mean': torch.mean(fitness).item(),
                    'fitness_std': torch.std(fitness).item(),
                    'throughput': signals.shape[0] / elapsed_time
                }
            except Exception as e:
                print(f"Benchmark for {name} failed: {e}")
                results[name] = {'time': -1, 'fitness_mean': -1, 'fitness_std': -1, 'throughput': -1}

        return results

def test_cuda_backtest_optimizer():
    """测试CUDA回测优化器"""
    print("=== CUDA回测优化器测试 ===")
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过测试")
        return
    device = torch.device('cuda:0')
    optimizer = CudaBacktestOptimizer(device)
    population_size = 500
    n_samples = 1000
    print(f"测试配置: 种群{population_size}, 样本{n_samples}")
    signals = torch.rand(population_size, n_samples, device=device)
    returns = torch.randn(n_samples, device=device) * 0.01
    buy_thresholds = torch.rand(population_size, device=device) * 0.3 + 0.5
    sell_thresholds = torch.rand(population_size, device=device) * 0.3 + 0.2
    max_positions = torch.rand(population_size, device=device) * 0.8 + 0.2
    stop_losses = torch.rand(population_size, device=device) * 0.05 + 0.02
    max_drawdowns = torch.rand(population_size, device=device) * 0.15 + 0.1
    results = optimizer.benchmark_methods(
        signals, returns, buy_thresholds, sell_thresholds, max_positions, stop_losses, max_drawdowns
    )
    print("\n基准测试结果:")
    print("-" * 70)
    print(f"{'方法':<15} {'时间(s)':<12} {'吞吐量 (个/s)':<20} {'适应度均值':<15}")
    print("-" * 70)
    for name, result in results.items():
        if result['time'] > 0:
            print(f"{name:<15} {result['time']:<12.4f} {result['throughput']:<20.1f} {result['fitness_mean']:<15.6f}")
        else:
            print(f"{name:<15} {'FAILED':<12}")
    print("-" * 70)

if __name__ == "__main__":
    test_cuda_backtest_optimizer()
