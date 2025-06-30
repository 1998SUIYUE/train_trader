"""
数据退火调度器 - 逐步增加训练数据复杂度
Data Annealing Scheduler - Gradually increase training data complexity
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

class AnnealingStrategy(Enum):
    """退火策略枚举"""
    TEMPORAL = "temporal"              # 时间复杂度退火
    VOLATILITY = "volatility"          # 波动率退火
    MARKET_REGIME = "market_regime"    # 市场状态退火
    FEATURE_COMPLEXITY = "feature_complexity"  # 特征复杂度退火
    PROGRESSIVE = "progressive"        # 渐进式综合退火

@dataclass
class AnnealingConfig:
    """数据退火配置"""
    strategy: AnnealingStrategy = AnnealingStrategy.PROGRESSIVE
    total_generations: int = 1000
    min_data_ratio: float = 0.3        # 最小数据使用比例
    max_data_ratio: float = 1.0        # 最大数据使用比例
    annealing_rate: float = 0.1        # 退火速度
    volatility_threshold: float = 0.02 # 波动率阈值
    warmup_generations: int = 50       # 预热代数
    
class DataAnnealingScheduler:
    """数据退火调度器"""
    
    def __init__(self, config: AnnealingConfig):
        """
        初始化数据退火调度器
        
        Args:
            config: 退火配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 内部状态
        self.current_generation = 0
        self.data_complexity_history = []
        self.volatility_cache = {}
        
        # 策略映射
        self.strategy_map = {
            AnnealingStrategy.TEMPORAL: self._temporal_annealing,
            AnnealingStrategy.VOLATILITY: self._volatility_annealing,
            AnnealingStrategy.MARKET_REGIME: self._regime_annealing,
            AnnealingStrategy.FEATURE_COMPLEXITY: self._feature_annealing,
            AnnealingStrategy.PROGRESSIVE: self._progressive_annealing,
        }
        
        self.logger.info(f"数据退火调度器初始化完成，策略: {config.strategy.value}")
    
    def get_annealed_data(self, generation: int, features: torch.Tensor, 
                         labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        获取退火后的训练数据
        
        Args:
            generation: 当前代数
            features: 完整特征数据 [n_samples, feature_dim]
            labels: 完整标签数据 [n_samples]
            
        Returns:
            退火后的特征数据, 退火后的标签数据, 退火信息
        """
        self.current_generation = generation
        
        # 选择退火策略
        strategy_func = self.strategy_map[self.config.strategy]
        annealed_features, annealed_labels, annealing_info = strategy_func(
            generation, features, labels
        )
        
        # 记录退火历史
        self.data_complexity_history.append({
            'generation': generation,
            'data_ratio': annealing_info.get('data_ratio', 1.0),
            'complexity_score': annealing_info.get('complexity_score', 1.0),
            'strategy': self.config.strategy.value
        })
        
        return annealed_features, annealed_labels, annealing_info
    
    def _calculate_annealing_progress(self, generation: int) -> float:
        """计算退火进度 [0, 1]"""
        if generation < self.config.warmup_generations:
            return 0.0
        
        effective_generation = generation - self.config.warmup_generations
        effective_total = self.config.total_generations - self.config.warmup_generations
        
        if effective_total <= 0:
            return 1.0
            
        progress = min(1.0, effective_generation / effective_total)
        
        # 应用退火速度调整
        if self.config.annealing_rate != 1.0:
            progress = progress ** (1.0 / self.config.annealing_rate)
            
        return progress
    
    def _temporal_annealing(self, generation: int, features: torch.Tensor, 
                           labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """时间复杂度退火：从短期数据开始，逐步增加历史数据长度"""
        progress = self._calculate_annealing_progress(generation)
        
        # 计算数据使用比例
        data_ratio = self.config.min_data_ratio + progress * (
            self.config.max_data_ratio - self.config.min_data_ratio
        )
        
        # 计算使用的数据量
        total_samples = features.shape[0]
        use_samples = int(total_samples * data_ratio)
        use_samples = max(1, min(use_samples, total_samples))
        
        # 从最新数据开始选择（时间序列的末尾是最新的）
        start_idx = total_samples - use_samples
        annealed_features = features[start_idx:]
        annealed_labels = labels[start_idx:]
        
        annealing_info = {
            'strategy': 'temporal',
            'data_ratio': data_ratio,
            'use_samples': use_samples,
            'total_samples': total_samples,
            'progress': progress,
            'complexity_score': data_ratio
        }
        
        return annealed_features, annealed_labels, annealing_info
    
    def _volatility_annealing(self, generation: int, features: torch.Tensor, 
                             labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """波动率退火：从低波动期开始，逐步加入高波动期数据"""
        progress = self._calculate_annealing_progress(generation)
        
        # 计算标签的滚动波动率
        if generation not in self.volatility_cache:
            window_size = min(20, len(labels) // 10)  # 动态窗口大小
            volatilities = self._calculate_rolling_volatility(labels, window_size)
            self.volatility_cache[generation] = volatilities
        else:
            volatilities = self.volatility_cache[generation]
        
        # 根据进度确定波动率阈值
        volatility_percentile = self.config.min_data_ratio + progress * (
            self.config.max_data_ratio - self.config.min_data_ratio
        )
        volatility_threshold = torch.quantile(volatilities, volatility_percentile)
        
        # 选择低于阈值的数据点
        mask = volatilities <= volatility_threshold
        
        # 确保至少有一些数据
        if mask.sum() < len(labels) * 0.1:
            # 如果筛选后数据太少，选择波动率最低的10%
            _, indices = torch.topk(volatilities, int(len(labels) * 0.1), largest=False)
            mask = torch.zeros_like(volatilities, dtype=torch.bool)
            mask[indices] = True
        
        annealed_features = features[mask]
        annealed_labels = labels[mask]
        
        annealing_info = {
            'strategy': 'volatility',
            'data_ratio': mask.sum().item() / len(labels),
            'volatility_threshold': volatility_threshold.item(),
            'avg_volatility': volatilities[mask].mean().item(),
            'progress': progress,
            'complexity_score': volatility_percentile
        }
        
        return annealed_features, annealed_labels, annealing_info
    
    def _regime_annealing(self, generation: int, features: torch.Tensor, 
                         labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """市场状态退火：从单一市场状态到多种市场状态混合"""
        progress = self._calculate_annealing_progress(generation)
        
        # 简单的市场状态识别：基于收益率的符号和幅度
        returns = labels
        
        # 定义市场状态
        bull_market = returns > 0.01    # 牛市：大幅上涨
        bear_market = returns < -0.01   # 熊市：大幅下跌
        sideways = torch.abs(returns) <= 0.01  # 横盘：小幅波动
        
        # 根据进度选择市场状态
        if progress < 0.33:
            # 早期：只选择横盘市场（最简单）
            mask = sideways
            regime_name = "sideways"
        elif progress < 0.66:
            # 中期：横盘 + 牛市
            mask = sideways | bull_market
            regime_name = "sideways_bull"
        else:
            # 后期：所有市场状态
            mask = torch.ones_like(returns, dtype=torch.bool)
            regime_name = "all_regimes"
        
        # 确保有足够数据
        if mask.sum() < len(labels) * 0.1:
            mask = torch.ones_like(returns, dtype=torch.bool)
            regime_name = "fallback_all"
        
        annealed_features = features[mask]
        annealed_labels = labels[mask]
        
        annealing_info = {
            'strategy': 'market_regime',
            'data_ratio': mask.sum().item() / len(labels),
            'regime': regime_name,
            'bull_ratio': bull_market[mask].sum().item() / mask.sum().item(),
            'bear_ratio': bear_market[mask].sum().item() / mask.sum().item(),
            'sideways_ratio': sideways[mask].sum().item() / mask.sum().item(),
            'progress': progress,
            'complexity_score': progress
        }
        
        return annealed_features, annealed_labels, annealing_info
    
    def _feature_annealing(self, generation: int, features: torch.Tensor, 
                          labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """特征复杂度退火：从基础特征到复杂特征组合"""
        progress = self._calculate_annealing_progress(generation)
        
        # 计算要使用的特征数量
        total_features = features.shape[1]
        use_features = int(self.config.min_data_ratio * total_features + 
                          progress * (self.config.max_data_ratio - self.config.min_data_ratio) * total_features)
        use_features = max(1, min(use_features, total_features))
        
        # 选择特征：假设前面的特征是基础的，后面的是复杂的
        # 在实际应用中，可以根据特征的重要性或复杂度来排序
        selected_features = torch.arange(use_features)
        
        annealed_features = features[:, selected_features]
        annealed_labels = labels  # 标签不变
        
        annealing_info = {
            'strategy': 'feature_complexity',
            'data_ratio': 1.0,  # 数据量不变
            'feature_ratio': use_features / total_features,
            'use_features': use_features,
            'total_features': total_features,
            'progress': progress,
            'complexity_score': use_features / total_features
        }
        
        return annealed_features, annealed_labels, annealing_info
    
    def _progressive_annealing(self, generation: int, features: torch.Tensor, 
                              labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """渐进式综合退火：结合多种策略的综合方法"""
        progress = self._calculate_annealing_progress(generation)
        
        # 阶段性策略切换
        if progress < 0.25:
            # 第一阶段：时间退火（从短期开始）
            return self._temporal_annealing(generation, features, labels)
        elif progress < 0.5:
            # 第二阶段：波动率退火（加入低波动数据）
            return self._volatility_annealing(generation, features, labels)
        elif progress < 0.75:
            # 第三阶段：市场状态退火（多种市场状态）
            return self._regime_annealing(generation, features, labels)
        else:
            # 第四阶段：使用全部数据，但可能进行特征选择
            data_ratio = self.config.min_data_ratio + progress * (
                self.config.max_data_ratio - self.config.min_data_ratio
            )
            
            # 使用全部数据
            annealing_info = {
                'strategy': 'progressive_full',
                'data_ratio': 1.0,
                'feature_ratio': 1.0,
                'progress': progress,
                'complexity_score': 1.0,
                'stage': 'full_complexity'
            }
            
            return features, labels, annealing_info
    
    def _calculate_rolling_volatility(self, returns: torch.Tensor, window_size: int) -> torch.Tensor:
        """计算滚动波动率"""
        if len(returns) < window_size:
            return torch.full_like(returns, returns.std().item())
        
        volatilities = torch.zeros_like(returns)
        
        # 前面的数据使用扩展窗口
        for i in range(window_size):
            volatilities[i] = returns[:i+1].std()
        
        # 后面的数据使用滚动窗口
        for i in range(window_size, len(returns)):
            volatilities[i] = returns[i-window_size+1:i+1].std()
        
        return volatilities
    
    def get_annealing_progress(self) -> Dict:
        """获取退火进度信息（带错误保护）"""
        try:
            if not self.data_complexity_history:
                return {
                    'current_generation': self.current_generation,
                    'strategy': self.config.strategy.value,
                    'progress': 0.0,
                    'data_ratio': 1.0,
                    'complexity_score': 1.0,
                    'total_generations': self.config.total_generations,
                    'warmup_generations': self.config.warmup_generations
                }
            
            latest = self.data_complexity_history[-1]
            progress = self._calculate_annealing_progress(self.current_generation)
            
            return {
                'current_generation': self.current_generation,
                'strategy': self.config.strategy.value,
                'progress': progress,
                'data_ratio': latest.get('data_ratio', 1.0),
                'complexity_score': latest.get('complexity_score', 1.0),
                'total_generations': self.config.total_generations,
                'warmup_generations': self.config.warmup_generations
            }
        except Exception as e:
            self.logger.warning(f"获取退火进度失败: {e}")
            return {
                'current_generation': self.current_generation,
                'strategy': 'error',
                'progress': 0.0,
                'data_ratio': 1.0,
                'complexity_score': 1.0,
                'total_generations': self.config.total_generations,
                'warmup_generations': self.config.warmup_generations
            }
    
    def get_complexity_history(self) -> List[Dict]:
        """获取复杂度历史"""
        return self.data_complexity_history.copy()
    
    def reset(self):
        """重置调度器状态"""
        self.current_generation = 0
        self.data_complexity_history.clear()
        self.volatility_cache.clear()
        self.logger.info("数据退火调度器已重置")


if __name__ == "__main__":
    # 测试数据退火调度器
    print("=== 数据退火调度器测试 ===")
    
    # 创建测试数据
    n_samples = 1000
    n_features = 100
    
    # 模拟特征数据
    features = torch.randn(n_samples, n_features)
    
    # 模拟标签数据（价格变化）
    # 创建一些有趣的模式：前期低波动，中期高波动，后期混合
    labels = torch.zeros(n_samples)
    labels[:300] = torch.randn(300) * 0.005  # 低波动期
    labels[300:700] = torch.randn(400) * 0.02  # 高波动期
    labels[700:] = torch.randn(300) * 0.01   # 中等波动期
    
    # 测试不同的退火策略
    strategies = [
        AnnealingStrategy.TEMPORAL,
        AnnealingStrategy.VOLATILITY,
        AnnealingStrategy.MARKET_REGIME,
        AnnealingStrategy.FEATURE_COMPLEXITY,
        AnnealingStrategy.PROGRESSIVE
    ]
    
    for strategy in strategies:
        print(f"\n--- 测试策略: {strategy.value} ---")
        
        config = AnnealingConfig(
            strategy=strategy,
            total_generations=100,
            min_data_ratio=0.2,
            max_data_ratio=1.0,
            warmup_generations=10
        )
        
        scheduler = DataAnnealingScheduler(config)
        
        # 测试几个关键代数
        test_generations = [0, 10, 25, 50, 75, 99]
        
        for gen in test_generations:
            annealed_features, annealed_labels, info = scheduler.get_annealed_data(
                gen, features, labels
            )
            
            print(f"  代数 {gen:2d}: 数据比例={info.get('data_ratio', 1.0):.3f}, "
                  f"复杂度={info.get('complexity_score', 1.0):.3f}, "
                  f"样本数={annealed_features.shape[0]}, "
                  f"特征数={annealed_features.shape[1]}")
    
    print("\n=== 测试完成 ===")