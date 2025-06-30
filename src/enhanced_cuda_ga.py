"""
增强版CUDA遗传算法 - 第一阶段实现
Enhanced CUDA Genetic Algorithm - Phase 1 Implementation

集成功能：
1. 数据退火机制
2. 多目标优化
3. 增强监控系统
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import time
import json
from pathlib import Path
import logging

# 导入基础遗传算法
from cuda_accelerated_ga import CudaGPUAcceleratedGA, CudaGAConfig

# 导入新增功能模块
from data_annealing_scheduler import DataAnnealingScheduler, AnnealingConfig, AnnealingStrategy
from multi_objective_optimizer import MultiObjectiveOptimizer, MultiObjectiveConfig, ObjectiveConfig, ObjectiveType
from enhanced_monitoring import EnhancedMonitor, MonitoringConfig, PerformanceMetrics
from parameter_annealing_scheduler import ParameterAnnealingScheduler, ParameterAnnealingConfig, ParameterAnnealingStrategy, ParameterRange

try:
    from performance_profiler import timer
    PERFORMANCE_PROFILER_AVAILABLE = True
except ImportError:
    PERFORMANCE_PROFILER_AVAILABLE = False
    class timer:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

@dataclass
class EnhancedGAConfig(CudaGAConfig):
    """增强版遗传算法配置"""
    
    # 数据退火配置
    enable_data_annealing: bool = True
    annealing_strategy: AnnealingStrategy = AnnealingStrategy.PROGRESSIVE
    annealing_rate: float = 0.1
    min_data_ratio: float = 0.3
    max_data_ratio: float = 1.0
    warmup_generations: int = 50
    
    # 多目标优化配置
    enable_multi_objective: bool = True
    pareto_front_size: int = 100
    crowding_distance_weight: float = 0.5
    enable_hypervolume: bool = True
    
    # 目标权重配置（保持向后兼容）
    objective_weights: Dict[str, float] = None
    
    # 增强监控配置
    enable_enhanced_monitoring: bool = True
    monitoring_save_interval: int = 10
    detailed_logging: bool = True
    track_diversity: bool = True
    track_convergence: bool = True
    export_format: str = "json"
    
    # 参数退火配置
    enable_parameter_annealing: bool = True
    parameter_annealing_strategy: ParameterAnnealingStrategy = ParameterAnnealingStrategy.ADAPTIVE
    mutation_rate_range: ParameterRange = None
    crossover_rate_range: ParameterRange = None
    elite_ratio_range: ParameterRange = None
    
    def __post_init__(self):
        """后初始化处理"""
        super().__post_init__()
        
        # 设置默认目标权重
        if self.objective_weights is None:
            self.objective_weights = {
                'sharpe_ratio': 0.3,
                'max_drawdown': 0.2,  # 最小化目标
                'total_return': 0.25,
                'win_rate': 0.15,
                'volatility': 0.1,    # 最小化目标
            }
        
        # 验证权重和为1
        total_weight = sum(abs(w) for w in self.objective_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            # 自动归一化
            for key in self.objective_weights:
                self.objective_weights[key] /= total_weight

class EnhancedCudaGA(CudaGPUAcceleratedGA):
    """增强版CUDA遗传算法"""
    
    def __init__(self, config: EnhancedGAConfig, gpu_manager):
        """
        初始化增强版遗传算法
        
        Args:
            config: 增强配置
            gpu_manager: GPU管理器
        """
        # 初始化基础遗传算法
        super().__init__(config, gpu_manager)
        
        self.enhanced_config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化数据退火调度器
        if config.enable_data_annealing:
            annealing_config = AnnealingConfig(
                strategy=config.annealing_strategy,
                total_generations=config.max_generations,
                min_data_ratio=config.min_data_ratio,
                max_data_ratio=config.max_data_ratio,
                annealing_rate=config.annealing_rate,
                warmup_generations=config.warmup_generations
            )
            self.data_annealer = DataAnnealingScheduler(annealing_config)
            self.logger.info("数据退火调度器已启用")
        else:
            self.data_annealer = None
        
        # 初始化多目标优化器
        if config.enable_multi_objective:
            objectives_config = self._create_objectives_config()
            multi_obj_config = MultiObjectiveConfig(
                objectives=objectives_config,
                pareto_front_size=config.pareto_front_size,
                crowding_distance_weight=config.crowding_distance_weight,
                enable_hypervolume=config.enable_hypervolume
            )
            self.multi_objective_optimizer = MultiObjectiveOptimizer(multi_obj_config)
            self.logger.info("多目标优化器已启用")
        else:
            self.multi_objective_optimizer = None
        
        # 初始化增强监控系统
        if config.enable_enhanced_monitoring:
            monitoring_config = MonitoringConfig(
                save_interval=config.monitoring_save_interval,
                detailed_logging=config.detailed_logging,
                track_diversity=config.track_diversity,
                track_convergence=config.track_convergence,
                export_format=config.export_format
            )
            self.enhanced_monitor = EnhancedMonitor(monitoring_config)
            self.logger.info("增强监控系统已启用")
        else:
            self.enhanced_monitor = None
        
        # 初始化参数退火调度器
        if config.enable_parameter_annealing:
            param_annealing_config = ParameterAnnealingConfig(
                strategy=config.parameter_annealing_strategy,
                total_generations=config.max_generations,
                warmup_generations=config.warmup_generations,
                mutation_rate_range=config.mutation_rate_range,
                crossover_rate_range=config.crossover_rate_range,
                elite_ratio_range=config.elite_ratio_range
            )
            self.parameter_annealer = ParameterAnnealingScheduler(param_annealing_config)
            self.logger.info("参数退火调度器已启用")
        else:
            self.parameter_annealer = None
        
        self.logger.info("增强版CUDA遗传算法初始化完成")
    
    def _create_objectives_config(self) -> list:
        """创建目标配置列表"""
        objectives = []
        
        for obj_name, weight in self.enhanced_config.objective_weights.items():
            if obj_name in ['max_drawdown', 'volatility']:
                obj_type = ObjectiveType.MINIMIZE
            else:
                obj_type = ObjectiveType.MAXIMIZE
            
            objectives.append(ObjectiveConfig(
                name=obj_name,
                objective_type=obj_type,
                weight=abs(weight),
                normalize=True
            ))
        
        return objectives
    
    def evolve_one_generation_enhanced(self, features: torch.Tensor, labels: torch.Tensor, 
                                     output_dir: Optional[Path] = None) -> Dict[str, float]:
        """
        增强版单代进化
        
        Args:
            features: 特征数据
            labels: 标签数据
            output_dir: 输出目录
            
        Returns:
            增强的统计信息
        """
        with timer("evolve_one_generation_enhanced", "ga"):
            start_time = time.time()
            
            # 1. 参数退火处理
            parameter_annealing_info = {}
            if self.parameter_annealer:
                with timer("parameter_annealing", "ga"):
                    # 计算当前平均适应度作为性能指标
                    current_avg_fitness = torch.mean(self.fitness_scores).item() if hasattr(self, 'fitness_scores') else None
                    fitness_history = [h.get('avg_fitness', 0.0) for h in self.fitness_history[-20:]]  # 最近20代的历史
                    
                    annealed_params = self.parameter_annealer.get_annealed_parameters(
                        self.generation, current_avg_fitness, fitness_history
                    )
                    
                    # 更新遗传算法参数
                    self.config.mutation_rate = annealed_params['mutation_rate']
                    self.config.crossover_rate = annealed_params['crossover_rate'] 
                    self.config.elite_ratio = annealed_params['elite_ratio']
                    
                    parameter_annealing_info = {
                        'mutation_rate': annealed_params['mutation_rate'],
                        'crossover_rate': annealed_params['crossover_rate'],
                        'elite_ratio': annealed_params['elite_ratio'],
                        'learning_rate': annealed_params.get('learning_rate', 0.001),
                    }
                    
                    self.logger.debug(f"参数退火: 变异率={annealed_params['mutation_rate']:.4f}, "
                                    f"交叉率={annealed_params['crossover_rate']:.4f}, "
                                    f"精英比例={annealed_params['elite_ratio']:.4f}")
            
            # 2. 数据退火处理
            annealing_info = {}
            if self.data_annealer:
                with timer("data_annealing", "ga"):
                    annealed_features, annealed_labels, annealing_info = self.data_annealer.get_annealed_data(
                        self.generation, features, labels
                    )
                    self.logger.debug(f"数据退火: 使用比例={annealing_info.get('data_ratio', 1.0):.3f}")
            else:
                annealed_features, annealed_labels = features, labels
            
            # 3. 评估适应度（包括多目标）
            multi_objective_stats = {}
            if self.multi_objective_optimizer:
                try:
                    with timer("multi_objective_evaluation", "ga"):
                        # 提取个体参数
                        weights = self.population[:, :self.config.feature_dim]
                        biases = self.population[:, self.config.feature_dim]
                        buy_thresholds = self.population[:, self.config.feature_dim + 1]
                        sell_thresholds = self.population[:, self.config.feature_dim + 2]
                        stop_losses = self.population[:, self.config.feature_dim + 3]
                        max_positions = self.population[:, self.config.feature_dim + 4]
                        max_drawdowns = self.population[:, self.config.feature_dim + 5]
                        trade_positions = self.population[:, self.config.feature_dim + 6]
                        
                        # 计算预测信号
                        signals = torch.sigmoid(torch.matmul(weights, annealed_features.T) + biases.unsqueeze(1))
                        
                        # 多目标评估
                        objectives = self.multi_objective_optimizer.evaluate_all_objectives(
                            signals, annealed_labels, buy_thresholds, sell_thresholds,
                            stop_losses, max_positions, max_drawdowns, trade_positions
                        )
                        
                        # 计算帕累托前沿（大幅减少计算频率，避免卡死）
                        if self.generation % 50 == 0:  # 每50代计算一次完整的帕累托前沿
                            try:
                                pareto_front, domination_counts = self.multi_objective_optimizer.calculate_pareto_front(objectives)
                            except Exception as e:
                                self.logger.warning(f"帕累托前沿计算失败: {e}")
                                pareto_front, domination_counts = [], []
                        else:
                            pareto_front, domination_counts = [], []
                        
                        # 获取优化总结
                        multi_objective_stats = self.multi_objective_optimizer.get_optimization_summary(objectives)
                        
                        # 保存多目标统计数据供日志使用
                        self._last_multi_objective_stats = multi_objective_stats
                        
                        # 计算综合适应度（用于传统遗传算法操作）
                        self.fitness_scores = self._calculate_weighted_fitness(objectives)
                        
                except Exception as e:
                    self.logger.warning(f"多目标评估失败，回退到单目标: {e}")
                    # 回退到原始适应度评估
                    self.fitness_scores, sharpe_ratios, max_drawdowns_calc, normalized_trades = self.evaluate_fitness_batch(
                        annealed_features, annealed_labels
                    )
                    multi_objective_stats = {
                        'pareto_front_size': 0,
                        'hypervolume': 0.0,
                        'pareto_ratio': 0.0,
                        'objective_stats': {
                            'sharpe_ratio': {'mean': torch.mean(sharpe_ratios).item()},
                            'max_drawdown': {'mean': torch.mean(max_drawdowns_calc).item()},
                            'trade_frequency': {'mean': torch.mean(normalized_trades).item()},
                        }
                    }
            else:
                # 使用原始适应度评估
                self.fitness_scores, sharpe_ratios, max_drawdowns_calc, normalized_trades = self.evaluate_fitness_batch(
                    annealed_features, annealed_labels
                )
                
                # 构建兼容的多目标统计
                multi_objective_stats = {
                    'pareto_front_size': 0,
                    'hypervolume': 0.0,
                    'pareto_ratio': 0.0,
                    'objective_stats': {
                        'sharpe_ratio': {'mean': torch.mean(sharpe_ratios).item()},
                        'max_drawdown': {'mean': torch.mean(max_drawdowns_calc).item()},
                        'trade_frequency': {'mean': torch.mean(normalized_trades).item()},
                    }
                }
            
            # 4. 更新最佳个体
            with timer("update_best_individual", "ga"):
                current_avg_fitness = torch.mean(self.fitness_scores).item()
                best_idx = torch.argmax(self.fitness_scores)
                current_best_fitness = self.fitness_scores[best_idx].item()

                if current_avg_fitness > self.best_avg_fitness:
                    self.best_avg_fitness = current_avg_fitness
                    self.no_improvement_count = 0
                    
                    if output_dir and self.best_individual is not None:
                        best_path = output_dir / "best_individual.npy"
                        np.save(best_path, self.best_individual)
                        self.logger.debug(f"新的最佳个体已保存: 平均适应度={self.best_avg_fitness:.4f}")
                else:
                    self.no_improvement_count += 1

                if current_best_fitness > self.best_fitness:
                    self.best_fitness = current_best_fitness
                    self.best_individual = self.gpu_manager.to_cpu(self.population[best_idx])
            
            # 5. 遗传算法操作
            with timer("genetic_operations", "ga"):
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
            
            # 6. 构建增强统计信息
            enhanced_stats = {
                'generation': self.generation,
                'best_fitness': current_best_fitness,
                'avg_fitness': current_avg_fitness,
                'std_fitness': torch.std(self.fitness_scores).item(),
                'generation_time': generation_time,
                'no_improvement_count': self.no_improvement_count,
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate,
                'elite_ratio': self.config.elite_ratio,
            }
            
            # 7. 更新增强监控
            if self.enhanced_monitor:
                try:
                    with timer("enhanced_monitoring", "ga"):
                        # 适度减少种群多样性计算频率以提高性能
                        population_for_diversity = self.population if self.generation % 10 == 0 else None
                        
                        metrics = self.enhanced_monitor.update_metrics(
                            self.generation,
                            enhanced_stats,
                            multi_objective_stats,
                            annealing_info,
                            population_for_diversity
                        )
                        
                        # 保存种群多样性数据供日志使用
                        if hasattr(metrics, 'population_diversity'):
                            self._last_population_diversity = metrics.population_diversity
                        elif not hasattr(self, '_last_population_diversity'):
                            self._last_population_diversity = 0.5  # 默认值
                except Exception as e:
                    self.logger.warning(f"增强监控更新失败: {e}")
                    # 继续执行，不中断训练
                    if not hasattr(self, '_last_population_diversity'):
                        self._last_population_diversity = 0.5
            
            # 记录历史
            self.fitness_history.append(enhanced_stats)
            
            return enhanced_stats
    
    def _calculate_weighted_fitness(self, objectives: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算加权综合适应度"""
        weighted_fitness = torch.zeros(self.config.population_size, device=self.device)
        
        for obj_name, obj_values in objectives.items():
            if obj_name in self.enhanced_config.objective_weights:
                weight = self.enhanced_config.objective_weights[obj_name]
                weighted_fitness += weight * obj_values
        
        return weighted_fitness
    
    def _save_generation_log(self, stats: Dict[str, Any], log_file: Path):
        """简化版代数日志保存（确保JSON文件生成）"""
        try:
            # 确保目录存在
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 准备基础日志数据
            log_data = {
                'generation': stats.get('generation', 0),
                'best_fitness': stats.get('best_fitness', 0.0),
                'avg_fitness': stats.get('avg_fitness', 0.0),
                'std_fitness': stats.get('std_fitness', 0.0),
                'generation_time': stats.get('generation_time', 0.0),
                'no_improvement_count': stats.get('no_improvement_count', 0),
                'gpu_memory_allocated': stats.get('gpu_memory_allocated', 0.0),
                'gpu_memory_reserved': stats.get('gpu_memory_reserved', 0.0),
                'system_memory_gb': stats.get('system_memory_gb', 0.0),
            }
            
            # 安全地添加数据退火信息
            if self.data_annealer:
                try:
                    annealing_progress = self.data_annealer.get_annealing_progress()
                    log_data.update({
                        'data_ratio': annealing_progress.get('data_ratio', 1.0),
                        'complexity_score': annealing_progress.get('complexity_score', 1.0),
                        'annealing_strategy': annealing_progress.get('strategy', 'none'),
                        'annealing_progress': annealing_progress.get('progress', 0.0),
                    })
                except Exception as e:
                    self.logger.debug(f"获取退火进度失败: {e}")
                    log_data.update({
                        'data_ratio': 1.0,
                        'complexity_score': 1.0,
                        'annealing_strategy': 'error',
                        'annealing_progress': 0.0,
                    })
            
            # 安全地添加参数退火信息
            if self.parameter_annealer:
                try:
                    current_params = self.parameter_annealer.get_current_parameters()
                    log_data.update({
                        'current_mutation_rate': current_params.get('mutation_rate', self.config.mutation_rate),
                        'current_crossover_rate': current_params.get('crossover_rate', self.config.crossover_rate),
                        'current_elite_ratio': current_params.get('elite_ratio', self.config.elite_ratio),
                        'current_learning_rate': current_params.get('learning_rate', 0.001),
                        'parameter_annealing_strategy': self.parameter_annealer.config.strategy.value,
                    })
                except Exception as e:
                    self.logger.debug(f"获取参数退火信息失败: {e}")
                    log_data.update({
                        'current_mutation_rate': self.config.mutation_rate,
                        'current_crossover_rate': self.config.crossover_rate,
                        'current_elite_ratio': self.config.elite_ratio,
                        'current_learning_rate': 0.001,
                        'parameter_annealing_strategy': 'error',
                    })
            
            # 安全地添加多目标优化数据
            if hasattr(self, '_last_multi_objective_stats') and self._last_multi_objective_stats:
                try:
                    log_data.update({
                        'pareto_front_size': self._last_multi_objective_stats.get('pareto_front_size', 0),
                        'hypervolume': self._last_multi_objective_stats.get('hypervolume', 0.0),
                        'pareto_ratio': self._last_multi_objective_stats.get('pareto_ratio', 0.0),
                    })
                    
                    # 添加交易性能指标
                    obj_stats = self._last_multi_objective_stats.get('objective_stats', {})
                    log_data.update({
                        'avg_sharpe_ratio': obj_stats.get('sharpe_ratio', {}).get('mean', 0.0),
                        'avg_max_drawdown': obj_stats.get('max_drawdown', {}).get('mean', 0.0),
                        'avg_total_return': obj_stats.get('total_return', {}).get('mean', 0.0),
                        'avg_win_rate': obj_stats.get('win_rate', {}).get('mean', 0.0),
                        'avg_trade_frequency': obj_stats.get('trade_frequency', {}).get('mean', 0.0),
                        'avg_volatility': obj_stats.get('volatility', {}).get('mean', 0.0),
                        'avg_profit_factor': obj_stats.get('profit_factor', {}).get('mean', 0.0),
                    })
                except Exception as e:
                    self.logger.debug(f"获取多目标统计失败: {e}")
            
            # 添加种群多样性（如果可用）
            if hasattr(self, '_last_population_diversity'):
                log_data['population_diversity'] = self._last_population_diversity
            
            # 直接写入主文件
            try:
                with open(log_file, 'a', encoding='utf-8', buffering=1) as f:
                    json.dump(log_data, f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()
                    
                # 验证文件是否成功写入
                if log_file.exists() and log_file.stat().st_size > 0:
                    self.logger.debug(f"成功写入日志: 代数 {log_data['generation']}")
                else:
                    raise Exception("文件写入验证失败")
                    
            except Exception as e1:
                self.logger.warning(f"主文件写入失败: {e1}")
                
                # 尝试备份文件
                backup_file = log_file.with_suffix('.jsonl.backup')
                try:
                    with open(backup_file, 'a', encoding='utf-8', buffering=1) as f:
                        json.dump(log_data, f, ensure_ascii=False)
                        f.write('\n')
                        f.flush()
                    self.logger.info(f"已写入备份文件: {backup_file}")
                except Exception as e2:
                    self.logger.error(f"备份文件写入也失败: {e2}")
                    
                    # 最后的应急措施：简单文本文件
                    try:
                        simple_file = log_file.with_suffix('.simple.log')
                        with open(simple_file, 'a', encoding='utf-8') as f:
                            f.write(f"Gen {log_data['generation']}: fitness={log_data['best_fitness']:.6f}\n")
                            f.flush()
                        self.logger.info(f"已写入简单日志: {simple_file}")
                    except:
                        pass
                        
        except Exception as e:
            self.logger.error(f"日志保存完全失败: {e}")
            # 确保至少有一个文件被创建
            try:
                emergency_file = log_file.parent / "emergency_log.txt"
                with open(emergency_file, 'a', encoding='utf-8') as f:
                    f.write(f"Gen {stats.get('generation', 0)}: {stats.get('best_fitness', 0.0):.6f}\n")
                    f.flush()
            except:
                pass
    
    def _save_generation_log_simple(self, stats: Dict[str, Any], log_file: Path):
        """超简化版代数日志保存（专注于确保文件生成）"""
        try:
            # 确保目录存在
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 准备最基础的数据
            log_data = {
                'generation': stats.get('generation', 0),
                'best_fitness': float(stats.get('best_fitness', 0.0)),
                'avg_fitness': float(stats.get('avg_fitness', 0.0)),
                'generation_time': float(stats.get('generation_time', 0.0)),
                'no_improvement_count': int(stats.get('no_improvement_count', 0)),
                'timestamp': time.time(),
            }
            
            # 安全地添加额外数据
            try:
                if 'gpu_memory_allocated' in stats:
                    log_data['gpu_memory_allocated'] = float(stats['gpu_memory_allocated'])
                if 'system_memory_gb' in stats:
                    log_data['system_memory_gb'] = float(stats['system_memory_gb'])
            except:
                pass
            
            # 尝试写入主文件
            success = False
            try:
                with open(log_file, 'a', encoding='utf-8', buffering=1) as f:
                    json.dump(log_data, f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()
                success = True
                
                # 每10代验证一次文件
                if log_data['generation'] % 10 == 0:
                    if log_file.exists() and log_file.stat().st_size > 0:
                        self.logger.debug(f"JSON文件验证成功: {log_file}")
                    else:
                        success = False
                        
            except Exception as e:
                self.logger.debug(f"主文件写入失败: {e}")
                success = False
            
            # 如果主文件失败，尝试备份
            if not success:
                try:
                    backup_file = log_file.with_suffix('.jsonl.backup')
                    with open(backup_file, 'a', encoding='utf-8', buffering=1) as f:
                        json.dump(log_data, f, ensure_ascii=False)
                        f.write('\n')
                        f.flush()
                    self.logger.info(f"已写入备份文件: {backup_file}")
                except Exception as e:
                    self.logger.debug(f"备份文件写入失败: {e}")
                    
                    # 最后的应急措施
                    try:
                        emergency_file = log_file.parent / "emergency_training_log.txt"
                        with open(emergency_file, 'a', encoding='utf-8') as f:
                            f.write(f"Gen {log_data['generation']}: fitness={log_data['best_fitness']:.6f}\n")
                            f.flush()
                    except:
                        pass  # 如果连这个都失败，就放弃
                        
        except Exception as e:
            self.logger.error(f"简化日志保存完全失败: {e}")
    
    def evolve_enhanced(self, features: torch.Tensor, labels: torch.Tensor,
                       save_checkpoints: bool = True,
                       checkpoint_dir: Optional[Path] = None,
                       save_generation_results: bool = True,
                       generation_log_file: Optional[Path] = None,
                       generation_log_interval: int = 1,
                       auto_save_best: bool = True,
                       output_dir: Optional[Path] = None,
                       show_detailed_progress: bool = True,
                       progress_update_interval: float = 1.0) -> Dict[str, Any]:
        """
        增强版主进化循环
        
        Args:
            features: 训练特征
            labels: 训练标签
            save_checkpoints: 是否保存检查点
            checkpoint_dir: 检查点目录
            save_generation_results: 是否保存每代结果
            generation_log_file: 日志文件路径
            generation_log_interval: 日志记录间隔
            auto_save_best: 是否自动保存最佳个体
            output_dir: 输出目录
            show_detailed_progress: 是否显示详细进度
            progress_update_interval: 进度更新间隔
            
        Returns:
            增强的训练结果
        """
        # 设置监控日志文件
        if self.enhanced_monitor and generation_log_file:
            self.enhanced_monitor.config.log_file = generation_log_file
        
        # 启动增强监控
        if self.enhanced_monitor:
            self.enhanced_monitor.start_monitoring(self.config.max_generations)
        
        start_time = time.time()
        
        # 确保数据在GPU上
        features = self.gpu_manager.to_gpu(features)
        labels = self.gpu_manager.to_gpu(labels)
        
        self.logger.info("开始增强版进化过程...")
        
        try:
            while True:
                # 检查停止条件
                if self.config.max_generations > 0 and self.generation >= self.config.max_generations:
                    self.logger.info(f"达到最大代数 {self.config.max_generations}，停止训练")
                    break
                
                if self.no_improvement_count >= self.config.early_stop_patience:
                    self.logger.info(f"连续 {self.config.early_stop_patience} 代无改进，早停")
                    break
                
                # 增强版进化一代
                stats = self.evolve_one_generation_enhanced(features, labels, output_dir)
                
                # 添加GPU内存信息
                if torch.cuda.is_available():
                    stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1e9
                    stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1e9
                
                # 添加系统内存信息
                import psutil
                stats['system_memory_gb'] = psutil.virtual_memory().used / 1e9
                
                # 直接保存到JSONL文件（确保实时写入）
                if save_generation_results and generation_log_file and self.generation % generation_log_interval == 0:
                    # 使用简化的保存方法，确保JSON文件生成
                    self._save_generation_log_simple(stats, generation_log_file)
                
                # 显示进度（简化版，减少频率）
                if self.generation % 5 == 0 or self.generation < 5:
                    progress_info = f"代数 {self.generation:4d}: "
                    progress_info += f"最佳适应度={stats['best_fitness']:.6f}, "
                    progress_info += f"平均适应度={stats['avg_fitness']:.6f}, "
                    progress_info += f"无改进次数={stats['no_improvement_count']}"
                    
                    if self.data_annealer:
                        try:
                            annealing_progress = self.data_annealer.get_annealing_progress()
                            progress_info += f", 数据比例={annealing_progress.get('data_ratio', 1.0):.3f}"
                        except:
                            pass  # 忽略退火进度获取错误
                    
                    print(progress_info)
                
                # 定期保存检查点
                if save_checkpoints and checkpoint_dir and self.generation % 50 == 0:
                    checkpoint_path = checkpoint_dir / f"checkpoint_gen_{self.generation}.pt"
                    self.save_checkpoint(str(checkpoint_path))
                
                # 定期清理GPU缓存和重试保存
                if self.generation % 10 == 0:
                    self.gpu_manager.clear_cache()
                    
                    # 重试保存失败的监控数据
                    if self.enhanced_monitor and hasattr(self.enhanced_monitor, '_retry_failed_saves'):
                        self.enhanced_monitor._retry_failed_saves()
        
        except KeyboardInterrupt:
            self.logger.info("\n训练被用户中断")
        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {e}")
            raise
        
        total_time = time.time() - start_time
        
        # 保存最终检查点
        if save_checkpoints and checkpoint_dir:
            final_checkpoint_path = checkpoint_dir / "final_checkpoint_enhanced.pt"
            self.save_checkpoint_enhanced(str(final_checkpoint_path))
            self.logger.info(f"最终增强检查点已保存: {final_checkpoint_path.name}")
        
        # 保存最终最佳个体
        if auto_save_best and output_dir and self.best_individual is not None:
            final_best_path = output_dir / "best_individual_enhanced.npy"
            np.save(final_best_path, self.best_individual)
            self.logger.info(f"最终最佳个体已保存: {final_best_path.name}")
        
        # 生成增强训练报告
        enhanced_results = self._generate_enhanced_results(total_time, output_dir)
        
        # 显示最终总结
        self._display_enhanced_summary(enhanced_results)
        
        return enhanced_results
    
    def save_checkpoint_enhanced(self, filepath: str) -> None:
        """保存增强检查点"""
        checkpoint = {
            'generation': self.generation,
            'population': self.gpu_manager.to_cpu(self.population),
            'fitness_scores': self.gpu_manager.to_cpu(self.fitness_scores),
            'best_fitness': self.best_fitness,
            'best_avg_fitness': self.best_avg_fitness,
            'best_individual': self.best_individual,
            'fitness_history': self.fitness_history,
            'no_improvement_count': self.no_improvement_count,
            'config': self.enhanced_config,
            
            # 增强状态
            'data_annealer_state': self.data_annealer.get_complexity_history() if self.data_annealer else None,
            'enhanced_monitor_state': self.enhanced_monitor.metrics_history if self.enhanced_monitor else None,
            'parameter_annealer_state': self.parameter_annealer.get_parameter_history() if self.parameter_annealer else None,
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"增强检查点已保存: {filepath}")
    
    def load_checkpoint_enhanced(self, filepath: str) -> None:
        """加载增强检查点"""
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # 加载基础状态
        self.generation = checkpoint['generation']
        self.population = self.gpu_manager.to_gpu(checkpoint['population'])
        self.fitness_scores = self.gpu_manager.to_gpu(checkpoint['fitness_scores'])
        self.best_fitness = checkpoint['best_fitness']
        self.best_avg_fitness = checkpoint['best_avg_fitness']
        self.best_individual = checkpoint['best_individual']
        self.fitness_history = checkpoint['fitness_history']
        self.no_improvement_count = checkpoint['no_improvement_count']
        
        # 恢复增强状态
        if 'data_annealer_state' in checkpoint and self.data_annealer:
            self.data_annealer.data_complexity_history = checkpoint['data_annealer_state'] or []
        
        if 'enhanced_monitor_state' in checkpoint and self.enhanced_monitor:
            self.enhanced_monitor.metrics_history = checkpoint['enhanced_monitor_state'] or []
        
        if 'parameter_annealer_state' in checkpoint and self.parameter_annealer:
            self.parameter_annealer.parameter_history = checkpoint['parameter_annealer_state'] or []
        
        self.logger.info(f"增强检查点已加载: {filepath}")
        self.logger.info(f"恢复到第 {self.generation} 代，最佳适应度: {self.best_fitness:.4f}")
    
    def _generate_enhanced_results(self, total_time: float, output_dir: Optional[Path]) -> Dict[str, Any]:
        """生成增强训练结果"""
        results = {
            'best_fitness': self.best_fitness,
            'best_individual': self.best_individual,
            'final_generation': self.generation,
            'total_time': total_time,
            'fitness_history': self.fitness_history
        }
        
        # 添加增强监控结果
        if self.enhanced_monitor:
            results['training_summary'] = self.enhanced_monitor.get_training_summary()
            results['metrics_history'] = [m.__dict__ for m in self.enhanced_monitor.metrics_history]
            
            # 导出详细报告
            if output_dir:
                report_path = output_dir / "enhanced_training_report.json"
                self.enhanced_monitor.export_detailed_report(report_path)
                results['detailed_report_path'] = str(report_path)
                
                # 生成训练进度图
                plot_path = output_dir / "training_progress.png"
                plot_result = self.enhanced_monitor.plot_training_progress(plot_path)
                if plot_result:
                    results['progress_plot_path'] = str(plot_path)
        
        # 添加数据退火结果
        if self.data_annealer:
            results['annealing_history'] = self.data_annealer.get_complexity_history()
            results['final_annealing_progress'] = self.data_annealer.get_annealing_progress()
        
        # 添加多目标优化结果
        if self.multi_objective_optimizer:
            results['multi_objective_config'] = {
                'objectives': [obj.name for obj in self.multi_objective_optimizer.config.objectives],
                'pareto_front_size': self.multi_objective_optimizer.config.pareto_front_size,
                'hypervolume_enabled': self.multi_objective_optimizer.config.enable_hypervolume
            }
        
        return results
    
    def _display_enhanced_summary(self, results: Dict[str, Any]):
        """显示增强训练总结"""
        print("=" * 80)
        print("              增强版CUDA遗传算法训练完成")
        print("=" * 80)
        print(f"  - 使用GPU:           {self.gpu_manager.device}")
        if self.gpu_manager.device.type == 'cuda':
            print(f"  - GPU名称:           {torch.cuda.get_device_name(self.gpu_manager.device.index)}")
        print(f"  - 最佳适应度:         {results['best_fitness']:.6f}")
        print(f"  - 总训练时间:         {results['total_time']:.2f}秒")
        print(f"  - 最终代数:           {results['final_generation']}")
        print(f"  - 种群大小:           {self.enhanced_config.population_size}")
        
        # 增强功能总结
        if self.data_annealer:
            annealing_progress = results.get('final_annealing_progress', {})
            print(f"  - 数据退火策略:       {annealing_progress.get('strategy', 'N/A')}")
            print(f"  - 最终数据复杂度:     {annealing_progress.get('complexity_score', 0.0):.3f}")
        
        if self.multi_objective_optimizer:
            mo_config = results.get('multi_objective_config', {})
            print(f"  - 多目标优化:         已启用 ({len(mo_config.get('objectives', []))}个目标)")
            print(f"  - 帕累托前沿大小:     {mo_config.get('pareto_front_size', 0)}")
        
        if self.enhanced_monitor:
            training_summary = results.get('training_summary', {})
            print(f"  - 收敛状态:           {'已收敛' if training_summary.get('convergence_achieved', False) else '未收敛'}")
            print(f"  - 平均代数时间:       {training_summary.get('avg_generation_time', 0.0):.2f}秒")
        
        # 文件路径
        if 'detailed_report_path' in results:
            print(f"  - 详细报告:           {results['detailed_report_path']}")
        if 'progress_plot_path' in results:
            print(f"  - 进度图表:           {results['progress_plot_path']}")
        
        print("=" * 80)
        
        # 显示GPU内存使用情况
        gpu_alloc, gpu_total, sys_used, sys_total = self.gpu_manager.get_memory_usage()
        print(f"最终GPU内存使用: {gpu_alloc:.2f}GB / {gpu_total:.2f}GB")
        print(f"最终系统内存使用: {sys_used:.2f}GB / {sys_total:.2f}GB")


if __name__ == "__main__":
    # 测试增强版CUDA遗传算法
    print("=== 增强版CUDA遗传算法测试 ===")
    
    from cuda_gpu_utils import get_cuda_gpu_manager
    from pathlib import Path
    
    # 初始化GPU管理器
    gpu_manager = get_cuda_gpu_manager()
    
    # 创建测试配置
    config = EnhancedGAConfig(
        population_size=200,
        max_generations=50,
        feature_dim=100,
        batch_size=500,
        
        # 启用所有增强功能
        enable_data_annealing=True,
        annealing_strategy=AnnealingStrategy.PROGRESSIVE,
        min_data_ratio=0.3,
        warmup_generations=10,
        
        enable_multi_objective=True,
        pareto_front_size=50,
        
        enable_enhanced_monitoring=True,
        monitoring_save_interval=5,
        detailed_logging=True,
    )
    
    # 创建输出目录
    output_dir = Path("test_enhanced_results")
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 创建测试数据
    n_samples = 1000
    features = np.random.randn(n_samples, config.feature_dim).astype(np.float32)
    labels = np.random.randn(n_samples).astype(np.float32) * 0.01
    
    print(f"测试数据: features {features.shape}, labels {labels.shape}")
    
    # 初始化增强版遗传算法
    ga = EnhancedCudaGA(config, gpu_manager)
    ga.initialize_population(seed=42)
    
    # 运行增强版训练
    start_time = time.time()
    results = ga.evolve_enhanced(
        features, labels,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        generation_log_file=output_dir / "enhanced_training.log"
    )
    test_time = time.time() - start_time
    
    print(f"\n增强版测试完成!")
    print(f"最佳适应度: {results['best_fitness']:.4f}")
    print(f"总代数: {results['final_generation']}")
    print(f"测试时间: {test_time:.2f}秒")
    
    # 显示增强功能效果
    if 'training_summary' in results:
        summary = results['training_summary']
        print(f"收敛状态: {summary.get('convergence_achieved', False)}")
        print(f"帕累托前沿大小: {summary.get('final_pareto_front_size', 0)}")
    
    print("增强版CUDA遗传算法测试完成！")