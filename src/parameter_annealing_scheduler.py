"""
参数退火调度器 - 动态调整遗传算法超参数
Parameter Annealing Scheduler - Dynamically adjust genetic algorithm hyperparameters
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import math

class ParameterAnnealingStrategy(Enum):
    """参数退火策略枚举"""
    LINEAR = "linear"                    # 线性退火
    EXPONENTIAL = "exponential"          # 指数退火
    COSINE = "cosine"                   # 余弦退火
    STEP = "step"                       # 阶梯退火
    ADAPTIVE = "adaptive"               # 自适应退火（基于性能）
    CYCLIC = "cyclic"                   # 周期性退火

@dataclass
class ParameterRange:
    """参数范围配置"""
    initial_value: float      # 初始值
    final_value: float        # 最终值
    min_value: float = None   # 最小值限制
    max_value: float = None   # 最大值限制
    
    def __post_init__(self):
        if self.min_value is None:
            self.min_value = min(self.initial_value, self.final_value)
        if self.max_value is None:
            self.max_value = max(self.initial_value, self.final_value)

@dataclass
class ParameterAnnealingConfig:
    """参数退火配置"""
    strategy: ParameterAnnealingStrategy = ParameterAnnealingStrategy.ADAPTIVE
    total_generations: int = 1000
    warmup_generations: int = 50
    
    # 变异率退火配置
    mutation_rate_range: ParameterRange = None
    
    # 交叉率退火配置  
    crossover_rate_range: ParameterRange = None
    
    # 精英比例退火配置
    elite_ratio_range: ParameterRange = None
    
    # 学习率退火配置（如果有的话）
    learning_rate_range: ParameterRange = None
    
    # 自适应退火参数
    performance_window: int = 10         # 性能评估窗口
    improvement_threshold: float = 0.001 # 改进阈值
    adaptation_rate: float = 0.1         # 自适应速度
    
    # 周期性退火参数
    cycle_length: int = 100              # 周期长度
    cycle_amplitude: float = 0.5         # 周期幅度
    
    def __post_init__(self):
        """设置默认参数范围"""
        if self.mutation_rate_range is None:
            self.mutation_rate_range = ParameterRange(
                initial_value=0.02,      # 开始时较高变异率，增加探索
                final_value=0.005,       # 结束时较低变异率，精细调优
                min_value=0.001,
                max_value=0.1
            )
        
        if self.crossover_rate_range is None:
            self.crossover_rate_range = ParameterRange(
                initial_value=0.6,       # 开始时较低交叉率
                final_value=0.9,         # 结束时较高交叉率，增加信息交换
                min_value=0.3,
                max_value=0.95
            )
        
        if self.elite_ratio_range is None:
            self.elite_ratio_range = ParameterRange(
                initial_value=0.02,      # 开始时较少精英，增加多样性
                final_value=0.1,         # 结束时较多精英，保持优秀基因
                min_value=0.01,
                max_value=0.2
            )
        
        if self.learning_rate_range is None:
            self.learning_rate_range = ParameterRange(
                initial_value=0.01,      # 开始时较高学习率
                final_value=0.001,       # 结束时较低学习率
                min_value=0.0001,
                max_value=0.1
            )

class ParameterAnnealingScheduler:
    """参数退火调度器"""
    
    def __init__(self, config: ParameterAnnealingConfig):
        """
        初始化参数退火调度器
        
        Args:
            config: 退火配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 内部状态
        self.current_generation = 0
        self.parameter_history = []
        self.performance_history = []
        
        # 自适应状态
        self.last_improvement_generation = 0
        self.adaptation_direction = {}  # 记录每个参数的调整方向
        
        # 周期性状态
        self.cycle_phase = 0.0
        
        # 策略映射
        self.strategy_map = {
            ParameterAnnealingStrategy.LINEAR: self._linear_annealing,
            ParameterAnnealingStrategy.EXPONENTIAL: self._exponential_annealing,
            ParameterAnnealingStrategy.COSINE: self._cosine_annealing,
            ParameterAnnealingStrategy.STEP: self._step_annealing,
            ParameterAnnealingStrategy.ADAPTIVE: self._adaptive_annealing,
            ParameterAnnealingStrategy.CYCLIC: self._cyclic_annealing,
        }
        
        self.logger.info(f"参数退火调度器初始化完成，策略: {config.strategy.value}")
    
    def get_annealed_parameters(self, generation: int, current_fitness: float = None,
                               fitness_history: List[float] = None) -> Dict[str, float]:
        """
        获取退火后的参数
        
        Args:
            generation: 当前代数
            current_fitness: 当前适应度
            fitness_history: 适应度历史
            
        Returns:
            退火后的参数字典
        """
        self.current_generation = generation
        
        # 更新性能历史
        if current_fitness is not None:
            self.performance_history.append(current_fitness)
            # 保持窗口大小
            if len(self.performance_history) > self.config.performance_window * 2:
                self.performance_history = self.performance_history[-self.config.performance_window:]
        
        # 选择退火策略
        strategy_func = self.strategy_map[self.config.strategy]
        annealed_params = strategy_func(generation, current_fitness, fitness_history)
        
        # 应用参数限制
        annealed_params = self._apply_parameter_constraints(annealed_params)
        
        # 记录参数历史
        param_record = {
            'generation': generation,
            'current_fitness': current_fitness,
            **annealed_params
        }
        self.parameter_history.append(param_record)
        
        return annealed_params
    
    def _calculate_annealing_progress(self, generation: int) -> float:
        """计算退火进度 [0, 1]"""
        if generation < self.config.warmup_generations:
            return 0.0
        
        effective_generation = generation - self.config.warmup_generations
        effective_total = self.config.total_generations - self.config.warmup_generations
        
        if effective_total <= 0:
            return 1.0
            
        return min(1.0, effective_generation / effective_total)
    
    def _linear_annealing(self, generation: int, current_fitness: float = None,
                         fitness_history: List[float] = None) -> Dict[str, float]:
        """线性退火"""
        progress = self._calculate_annealing_progress(generation)
        
        return {
            'mutation_rate': self._interpolate_parameter(
                self.config.mutation_rate_range, progress
            ),
            'crossover_rate': self._interpolate_parameter(
                self.config.crossover_rate_range, progress
            ),
            'elite_ratio': self._interpolate_parameter(
                self.config.elite_ratio_range, progress
            ),
            'learning_rate': self._interpolate_parameter(
                self.config.learning_rate_range, progress
            ),
        }
    
    def _exponential_annealing(self, generation: int, current_fitness: float = None,
                              fitness_history: List[float] = None) -> Dict[str, float]:
        """指数退火"""
        progress = self._calculate_annealing_progress(generation)
        
        # 使用指数函数进行退火
        exp_progress = 1.0 - math.exp(-3.0 * progress)  # 3.0控制指数衰减速度
        
        return {
            'mutation_rate': self._interpolate_parameter(
                self.config.mutation_rate_range, exp_progress
            ),
            'crossover_rate': self._interpolate_parameter(
                self.config.crossover_rate_range, exp_progress
            ),
            'elite_ratio': self._interpolate_parameter(
                self.config.elite_ratio_range, exp_progress
            ),
            'learning_rate': self._interpolate_parameter(
                self.config.learning_rate_range, exp_progress
            ),
        }
    
    def _cosine_annealing(self, generation: int, current_fitness: float = None,
                         fitness_history: List[float] = None) -> Dict[str, float]:
        """余弦退火"""
        progress = self._calculate_annealing_progress(generation)
        
        # 使用余弦函数进行退火
        cos_progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        
        return {
            'mutation_rate': self._interpolate_parameter(
                self.config.mutation_rate_range, cos_progress
            ),
            'crossover_rate': self._interpolate_parameter(
                self.config.crossover_rate_range, cos_progress
            ),
            'elite_ratio': self._interpolate_parameter(
                self.config.elite_ratio_range, cos_progress
            ),
            'learning_rate': self._interpolate_parameter(
                self.config.learning_rate_range, cos_progress
            ),
        }
    
    def _step_annealing(self, generation: int, current_fitness: float = None,
                       fitness_history: List[float] = None) -> Dict[str, float]:
        """阶梯退火"""
        progress = self._calculate_annealing_progress(generation)
        
        # 分为4个阶段
        if progress < 0.25:
            step_progress = 0.0
        elif progress < 0.5:
            step_progress = 0.33
        elif progress < 0.75:
            step_progress = 0.66
        else:
            step_progress = 1.0
        
        return {
            'mutation_rate': self._interpolate_parameter(
                self.config.mutation_rate_range, step_progress
            ),
            'crossover_rate': self._interpolate_parameter(
                self.config.crossover_rate_range, step_progress
            ),
            'elite_ratio': self._interpolate_parameter(
                self.config.elite_ratio_range, step_progress
            ),
            'learning_rate': self._interpolate_parameter(
                self.config.learning_rate_range, step_progress
            ),
        }
    
    def _adaptive_annealing(self, generation: int, current_fitness: float = None,
                           fitness_history: List[float] = None) -> Dict[str, float]:
        """自适应退火：基于性能动态调整"""
        base_progress = self._calculate_annealing_progress(generation)
        
        # 如果没有性能数据，使用线性退火
        if len(self.performance_history) < self.config.performance_window:
            return self._linear_annealing(generation, current_fitness, fitness_history)
        
        # 计算最近的性能改进
        recent_performance = self.performance_history[-self.config.performance_window:]
        performance_trend = self._calculate_performance_trend(recent_performance)
        
        # 根据性能趋势调整参数
        adaptation_factor = self._calculate_adaptation_factor(performance_trend)
        
        # 基础参数值
        base_params = self._linear_annealing(generation, current_fitness, fitness_history)
        
        # 自适应调整
        adapted_params = {}
        for param_name, base_value in base_params.items():
            if param_name == 'mutation_rate':
                # 性能停滞时增加变异率，性能改善时减少变异率
                adapted_params[param_name] = base_value * (1.0 + adaptation_factor * 0.5)
            elif param_name == 'crossover_rate':
                # 性能停滞时减少交叉率，性能改善时增加交叉率
                adapted_params[param_name] = base_value * (1.0 - adaptation_factor * 0.2)
            elif param_name == 'elite_ratio':
                # 性能停滞时减少精英比例，性能改善时增加精英比例
                adapted_params[param_name] = base_value * (1.0 - adaptation_factor * 0.3)
            else:
                adapted_params[param_name] = base_value
        
        return adapted_params
    
    def _cyclic_annealing(self, generation: int, current_fitness: float = None,
                         fitness_history: List[float] = None) -> Dict[str, float]:
        """周期性退火"""
        base_progress = self._calculate_annealing_progress(generation)
        
        # 计算周期内的位置
        cycle_position = (generation % self.config.cycle_length) / self.config.cycle_length
        cycle_factor = math.sin(2 * math.pi * cycle_position) * self.config.cycle_amplitude
        
        # 基础参数值
        base_params = self._linear_annealing(generation, current_fitness, fitness_history)
        
        # 添加周期性变化
        cyclic_params = {}
        for param_name, base_value in base_params.items():
            if param_name == 'mutation_rate':
                # 变异率周期性变化
                cyclic_params[param_name] = base_value * (1.0 + cycle_factor)
            elif param_name == 'crossover_rate':
                # 交叉率反向周期性变化
                cyclic_params[param_name] = base_value * (1.0 - cycle_factor * 0.5)
            else:
                cyclic_params[param_name] = base_value
        
        return cyclic_params
    
    def _interpolate_parameter(self, param_range: ParameterRange, progress: float) -> float:
        """插值计算参数值"""
        return param_range.initial_value + progress * (
            param_range.final_value - param_range.initial_value
        )
    
    def _apply_parameter_constraints(self, params: Dict[str, float]) -> Dict[str, float]:
        """应用参数约束"""
        constrained_params = {}
        
        param_ranges = {
            'mutation_rate': self.config.mutation_rate_range,
            'crossover_rate': self.config.crossover_rate_range,
            'elite_ratio': self.config.elite_ratio_range,
            'learning_rate': self.config.learning_rate_range,
        }
        
        for param_name, value in params.items():
            if param_name in param_ranges:
                param_range = param_ranges[param_name]
                constrained_value = max(param_range.min_value, 
                                      min(param_range.max_value, value))
                constrained_params[param_name] = constrained_value
            else:
                constrained_params[param_name] = value
        
        return constrained_params
    
    def _calculate_performance_trend(self, recent_performance: List[float]) -> float:
        """计算性能趋势"""
        if len(recent_performance) < 2:
            return 0.0
        
        # 计算线性趋势
        x = np.arange(len(recent_performance))
        y = np.array(recent_performance)
        
        # 简单线性回归
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # 归一化斜率
        y_range = max(y) - min(y)
        if y_range > 0:
            normalized_slope = slope / y_range
        else:
            normalized_slope = 0.0
        
        return np.clip(normalized_slope, -1.0, 1.0)
    
    def _calculate_adaptation_factor(self, performance_trend: float) -> float:
        """计算自适应因子"""
        # 性能趋势为负时（性能下降），返回正的适应因子
        # 性能趋势为正时（性能改善），返回负的适应因子
        adaptation_factor = -performance_trend * self.config.adaptation_rate
        
        return np.clip(adaptation_factor, -0.5, 0.5)
    
    def get_parameter_history(self) -> List[Dict]:
        """获取参数历史"""
        return self.parameter_history.copy()
    
    def get_current_parameters(self) -> Dict[str, float]:
        """获取当前参数"""
        if not self.parameter_history:
            return self.get_annealed_parameters(0)
        
        latest = self.parameter_history[-1]
        return {k: v for k, v in latest.items() if k != 'generation' and k != 'current_fitness'}
    
    def reset(self):
        """重置调度器状态"""
        self.current_generation = 0
        self.parameter_history.clear()
        self.performance_history.clear()
        self.last_improvement_generation = 0
        self.adaptation_direction.clear()
        self.cycle_phase = 0.0
        self.logger.info("参数退火调度器已重置")


if __name__ == "__main__":
    # 测试参数退火调度器
    print("=== 参数退火调度器测试 ===")
    
    # 创建测试配置
    config = ParameterAnnealingConfig(
        strategy=ParameterAnnealingStrategy.ADAPTIVE,
        total_generations=200,
        warmup_generations=20,
    )
    
    scheduler = ParameterAnnealingScheduler(config)
    
    # 模拟训练过程
    print("\n--- 模拟训练过程 ---")
    fitness_values = []
    
    for generation in range(100):
        # 模拟适应度变化（前期快速改善，后期缓慢改善）
        if generation < 30:
            fitness = 0.1 + generation * 0.02 + np.random.normal(0, 0.01)
        elif generation < 60:
            fitness = 0.7 + (generation - 30) * 0.005 + np.random.normal(0, 0.005)
        else:
            fitness = 0.85 + (generation - 60) * 0.001 + np.random.normal(0, 0.002)
        
        fitness_values.append(fitness)
        
        # 获取退火后的参数
        params = scheduler.get_annealed_parameters(generation, fitness, fitness_values)
        
        # 每10代打印一次
        if generation % 10 == 0:
            print(f"代数 {generation:3d}: 适应度={fitness:.4f}, "
                  f"变异率={params['mutation_rate']:.4f}, "
                  f"交叉率={params['crossover_rate']:.4f}, "
                  f"精英比例={params['elite_ratio']:.4f}")
    
    # 测试不同策略
    print("\n--- 测试不同退火策略 ---")
    strategies = [
        ParameterAnnealingStrategy.LINEAR,
        ParameterAnnealingStrategy.EXPONENTIAL,
        ParameterAnnealingStrategy.COSINE,
        ParameterAnnealingStrategy.STEP,
        ParameterAnnealingStrategy.CYCLIC,
    ]
    
    for strategy in strategies:
        print(f"\n策略: {strategy.value}")
        config.strategy = strategy
        scheduler = ParameterAnnealingScheduler(config)
        
        # 测试几个关键代数
        test_generations = [0, 25, 50, 75, 100]
        for gen in test_generations:
            params = scheduler.get_annealed_parameters(gen, 0.5)
            print(f"  代数 {gen:2d}: 变异率={params['mutation_rate']:.4f}, "
                  f"交叉率={params['crossover_rate']:.4f}, "
                  f"精英比例={params['elite_ratio']:.4f}")
    
    print("\n=== 测试完成 ===")