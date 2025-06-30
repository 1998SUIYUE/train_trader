"""
增强监控系统 - 全方位性能追踪和分析
Enhanced Monitoring System - Comprehensive Performance Tracking and Analysis
"""

import torch
import numpy as np
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from collections import deque
import psutil

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    # 基础指标
    generation: int
    best_fitness: float
    avg_fitness: float
    std_fitness: float
    
    # 多目标指标
    pareto_front_size: int = 0
    hypervolume: float = 0.0
    pareto_ratio: float = 0.0
    
    # 数据退火指标
    data_ratio: float = 1.0
    complexity_score: float = 1.0
    annealing_strategy: str = "none"
    annealing_progress: float = 0.0
    
    # 交易性能指标
    avg_sharpe_ratio: float = 0.0
    avg_max_drawdown: float = 0.0
    avg_total_return: float = 0.0
    avg_win_rate: float = 0.0
    avg_trade_frequency: float = 0.0
    avg_volatility: float = 0.0
    avg_profit_factor: float = 0.0
    
    # 系统性能指标
    generation_time: float = 0.0
    gpu_memory_allocated: float = 0.0
    gpu_memory_reserved: float = 0.0
    system_memory_gb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # 算法状态指标
    no_improvement_count: int = 0
    mutation_rate: float = 0.0
    crossover_rate: float = 0.0
    elite_ratio: float = 0.0
    
    # 收敛性指标
    fitness_improvement_rate: float = 0.0
    population_diversity: float = 0.0
    convergence_speed: float = 0.0

@dataclass
class MonitoringConfig:
    """监控配置"""
    log_file: Optional[Path] = None
    save_interval: int = 10
    detailed_logging: bool = True
    track_diversity: bool = True
    track_convergence: bool = True
    memory_monitoring: bool = True
    export_format: str = "json"  # json, csv, both

class EnhancedMonitor:
    """增强监控系统"""
    
    def __init__(self, config: MonitoringConfig):
        """
        初始化增强监控系统
        
        Args:
            config: 监控配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 监控数据存储
        self.metrics_history: List[PerformanceMetrics] = []
        self.fitness_history: deque = deque(maxlen=100)  # 最近100代的适应度历史
        self.diversity_history: deque = deque(maxlen=50)  # 多样性历史
        
        # 性能统计
        self.start_time = None
        self.best_fitness_ever = -float('inf')
        self.best_generation = 0
        self.total_generations = 0
        
        # 收敛性分析
        self.convergence_window = 20  # 收敛性分析窗口
        self.stagnation_threshold = 1e-6  # 停滞阈值
        
        self.logger.info("增强监控系统初始化完成")
    
    def start_monitoring(self, total_generations: int = -1):
        """开始监控"""
        self.start_time = time.time()
        self.total_generations = total_generations
        self.logger.info(f"开始监控，预计总代数: {total_generations}")
    
    def update_metrics(self, generation: int, basic_stats: Dict[str, Any],
                      multi_objective_stats: Optional[Dict[str, Any]] = None,
                      annealing_stats: Optional[Dict[str, Any]] = None,
                      population: Optional[torch.Tensor] = None) -> PerformanceMetrics:
        """
        更新监控指标
        
        Args:
            generation: 当前代数
            basic_stats: 基础统计信息
            multi_objective_stats: 多目标优化统计
            annealing_stats: 数据退火统计
            population: 当前种群（用于多样性计算）
            
        Returns:
            当前代的性能指标
        """
        # 创建基础指标
        metrics = PerformanceMetrics(
            generation=generation,
            best_fitness=basic_stats.get('best_fitness', 0.0),
            avg_fitness=basic_stats.get('avg_fitness', 0.0),
            std_fitness=basic_stats.get('std_fitness', 0.0),
            generation_time=basic_stats.get('generation_time', 0.0),
            no_improvement_count=basic_stats.get('no_improvement_count', 0),
        )
        
        # 更新多目标指标
        if multi_objective_stats:
            metrics.pareto_front_size = multi_objective_stats.get('pareto_front_size', 0)
            metrics.hypervolume = multi_objective_stats.get('hypervolume', 0.0)
            metrics.pareto_ratio = multi_objective_stats.get('pareto_ratio', 0.0)
            
            # 更新交易性能指标
            obj_stats = multi_objective_stats.get('objective_stats', {})
            metrics.avg_sharpe_ratio = obj_stats.get('sharpe_ratio', {}).get('mean', 0.0)
            metrics.avg_max_drawdown = obj_stats.get('max_drawdown', {}).get('mean', 0.0)
            metrics.avg_total_return = obj_stats.get('total_return', {}).get('mean', 0.0)
            metrics.avg_win_rate = obj_stats.get('win_rate', {}).get('mean', 0.0)
            metrics.avg_trade_frequency = obj_stats.get('trade_frequency', {}).get('mean', 0.0)
            metrics.avg_volatility = obj_stats.get('volatility', {}).get('mean', 0.0)
            metrics.avg_profit_factor = obj_stats.get('profit_factor', {}).get('mean', 0.0)
        
        # 更新数据退火指标
        if annealing_stats:
            metrics.data_ratio = annealing_stats.get('data_ratio', 1.0)
            metrics.complexity_score = annealing_stats.get('complexity_score', 1.0)
            metrics.annealing_strategy = annealing_stats.get('strategy', 'none')
            metrics.annealing_progress = annealing_stats.get('progress', 0.0)
        
        # 更新系统性能指标
        self._update_system_metrics(metrics)
        
        # 更新算法状态指标
        self._update_algorithm_metrics(metrics, basic_stats)
        
        # 计算收敛性指标
        if self.config.track_convergence:
            self._update_convergence_metrics(metrics)
        
        # 计算种群多样性
        if self.config.track_diversity and population is not None:
            diversity = self._calculate_population_diversity(population)
            metrics.population_diversity = diversity
            self.diversity_history.append(diversity)
        
        # 更新历史记录
        self.fitness_history.append(metrics.best_fitness)
        self.metrics_history.append(metrics)
        
        # 更新最佳记录
        if metrics.best_fitness > self.best_fitness_ever:
            self.best_fitness_ever = metrics.best_fitness
            self.best_generation = generation
        
        # 定期保存（确保JSON文件生成）
        if self.config.log_file:
            # 每代都保存，确保实时更新
            self._save_metrics(metrics)
        
        return metrics
    
    def _update_system_metrics(self, metrics: PerformanceMetrics):
        """更新系统性能指标"""
        if self.config.memory_monitoring:
            # GPU内存
            if torch.cuda.is_available():
                metrics.gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
                metrics.gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9
            
            # 系统内存和CPU
            memory = psutil.virtual_memory()
            metrics.system_memory_gb = memory.used / 1e9
            metrics.cpu_usage_percent = psutil.cpu_percent(interval=None)
    
    def _update_algorithm_metrics(self, metrics: PerformanceMetrics, basic_stats: Dict[str, Any]):
        """更新算法状态指标"""
        # 这些通常从遗传算法配置中获取
        metrics.mutation_rate = basic_stats.get('mutation_rate', 0.0)
        metrics.crossover_rate = basic_stats.get('crossover_rate', 0.0)
        metrics.elite_ratio = basic_stats.get('elite_ratio', 0.0)
    
    def _update_convergence_metrics(self, metrics: PerformanceMetrics):
        """更新收敛性指标"""
        if len(self.fitness_history) >= 2:
            # 适应度改进率
            recent_fitness = list(self.fitness_history)
            if len(recent_fitness) >= 2:
                improvement = recent_fitness[-1] - recent_fitness[-2]
                metrics.fitness_improvement_rate = improvement
            
            # 收敛速度（基于最近几代的适应度变化）
            if len(recent_fitness) >= self.convergence_window:
                window_fitness = recent_fitness[-self.convergence_window:]
                fitness_range = max(window_fitness) - min(window_fitness)
                metrics.convergence_speed = 1.0 / (fitness_range + 1e-8)
    
    def _calculate_population_diversity(self, population: torch.Tensor) -> float:
        """计算种群多样性（优化版）"""
        if population.shape[0] < 2:
            return 0.0
        
        try:
            # 使用更简单高效的方法：计算种群标准差的平均值
            population_flat = population.view(population.shape[0], -1)
            
            # 方法1：快速标准差方法
            if population.shape[0] > 100:
                # 对于大种群，使用标准差作为多样性指标
                std_values = torch.std(population_flat, dim=0)
                diversity = torch.mean(std_values).item()
                return diversity
            
            # 方法2：小种群使用采样距离方法
            sample_size = min(15, population.shape[0])
            if population.shape[0] > sample_size:
                indices = torch.randperm(population.shape[0])[:sample_size]
                sample_pop = population_flat[indices]
            else:
                sample_pop = population_flat
            
            # 计算少量个体对的距离
            n_pairs = min(20, sample_size * (sample_size - 1) // 2)
            
            if n_pairs > 0:
                # 随机选择个体对
                indices_i = torch.randint(0, sample_size, (n_pairs,))
                indices_j = torch.randint(0, sample_size, (n_pairs,))
                # 确保不是同一个个体
                mask = indices_i != indices_j
                if mask.sum() > 0:
                    indices_i = indices_i[mask]
                    indices_j = indices_j[mask]
                    
                    # 计算L2距离
                    distances = torch.norm(sample_pop[indices_i] - sample_pop[indices_j], dim=1)
                    avg_distance = distances.mean().item()
                    return avg_distance
            
            return 0.0
            
        except Exception as e:
            self.logger.debug(f"计算种群多样性失败: {e}")
            return 0.0
    
    def _serialize_config(self) -> Dict[str, Any]:
        """序列化配置，处理Path对象"""
        config_dict = asdict(self.config)
        
        # 将Path对象转换为字符串
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        return config_dict
    
    def _save_metrics(self, metrics: PerformanceMetrics):
        """简化版指标保存（确保JSON文件生成）"""
        if not self.config.log_file:
            return
        
        try:
            # 确保目录存在
            self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 准备数据
            metrics_dict = asdict(metrics)
            
            # 主要保存JSON格式
            if self.config.export_format in ["json", "both"]:
                json_file = self.config.log_file.with_suffix('.jsonl')
                
                try:
                    # 直接写入，不使用fsync避免阻塞
                    with open(json_file, 'a', encoding='utf-8', buffering=1) as f:
                        json.dump(metrics_dict, f, ensure_ascii=False)
                        f.write('\n')
                        f.flush()
                    
                    # 验证文件写入
                    if json_file.exists() and json_file.stat().st_size > 0:
                        return  # 成功写入，直接返回
                    
                except Exception as e1:
                    self.logger.debug(f"主JSON文件写入失败: {e1}")
                    
                    # 尝试备份文件
                    try:
                        backup_file = json_file.with_suffix('.jsonl.backup')
                        with open(backup_file, 'a', encoding='utf-8', buffering=1) as f:
                            json.dump(metrics_dict, f, ensure_ascii=False)
                            f.write('\n')
                            f.flush()
                        self.logger.debug(f"已写入备份文件: {backup_file}")
                        return
                    except Exception as e2:
                        self.logger.debug(f"备份文件写入失败: {e2}")
            
            # 如果JSON失败，尝试CSV
            if self.config.export_format in ["csv", "both"]:
                try:
                    csv_file = self.config.log_file.with_suffix('.csv')
                    
                    # 简化的CSV写入
                    import csv
                    file_exists = csv_file.exists()
                    
                    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(metrics_dict)
                        f.flush()
                    
                    self.logger.debug(f"已写入CSV文件: {csv_file}")
                    return
                    
                except Exception as e:
                    self.logger.debug(f"CSV保存失败: {e}")
            
            # 最后的应急措施：简单文本文件
            try:
                simple_file = self.config.log_file.with_suffix('.simple.log')
                with open(simple_file, 'a', encoding='utf-8') as f:
                    f.write(f"Gen {metrics.generation}: fitness={metrics.best_fitness:.6f}\n")
                    f.flush()
                self.logger.debug(f"已写入简单日志: {simple_file}")
            except Exception as e:
                self.logger.error(f"所有保存方式都失败: {e}")
                
        except Exception as e:
            self.logger.error(f"保存指标完全失败: {e}")
    
    def _retry_failed_saves(self):
        """重试保存失败的指标"""
        if not hasattr(self, '_failed_saves') or not self._failed_saves:
            return
        
        retry_count = 0
        while self._failed_saves and retry_count < 3:
            retry_count += 1
            failed_metrics = self._failed_saves.copy()
            self._failed_saves.clear()
            
            for metrics in failed_metrics:
                try:
                    self._save_metrics(metrics)
                except Exception as e:
                    self.logger.warning(f"重试保存失败 (第{retry_count}次): {e}")
                    self._failed_saves.append(metrics)
            
            if self._failed_saves:
                import time
                time.sleep(0.1)  # 短暂等待后重试
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练总结"""
        if not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        total_time = time.time() - self.start_time if self.start_time else 0
        
        summary = {
            # 基础信息
            'total_generations': len(self.metrics_history),
            'total_time_seconds': total_time,
            'avg_generation_time': total_time / len(self.metrics_history) if self.metrics_history else 0,
            
            # 最佳性能
            'best_fitness_ever': self.best_fitness_ever,
            'best_generation': self.best_generation,
            'final_fitness': latest_metrics.best_fitness,
            
            # 收敛性分析
            'fitness_improvement_total': latest_metrics.best_fitness - self.metrics_history[0].best_fitness if len(self.metrics_history) > 1 else 0,
            'convergence_achieved': self._check_convergence(),
            'stagnation_periods': self._count_stagnation_periods(),
            
            # 多目标性能
            'final_pareto_front_size': latest_metrics.pareto_front_size,
            'final_hypervolume': latest_metrics.hypervolume,
            'avg_pareto_ratio': np.mean([m.pareto_ratio for m in self.metrics_history[-10:]]) if len(self.metrics_history) >= 10 else latest_metrics.pareto_ratio,
            
            # 交易性能
            'final_sharpe_ratio': latest_metrics.avg_sharpe_ratio,
            'final_max_drawdown': latest_metrics.avg_max_drawdown,
            'final_total_return': latest_metrics.avg_total_return,
            'final_win_rate': latest_metrics.avg_win_rate,
            
            # 系统性能
            'peak_gpu_memory': max([m.gpu_memory_allocated for m in self.metrics_history]),
            'avg_cpu_usage': np.mean([m.cpu_usage_percent for m in self.metrics_history]),
            'peak_system_memory': max([m.system_memory_gb for m in self.metrics_history]),
            
            # 数据退火效果
            'annealing_strategies_used': list(set([m.annealing_strategy for m in self.metrics_history])),
            'final_data_complexity': latest_metrics.complexity_score,
        }
        
        return summary
    
    def _check_convergence(self) -> bool:
        """检查是否收敛"""
        if len(self.fitness_history) < self.convergence_window:
            return False
        
        recent_fitness = list(self.fitness_history)[-self.convergence_window:]
        fitness_range = max(recent_fitness) - min(recent_fitness)
        
        return fitness_range < self.stagnation_threshold
    
    def _count_stagnation_periods(self) -> int:
        """统计停滞期数量"""
        if len(self.fitness_history) < self.convergence_window:
            return 0
        
        stagnation_count = 0
        recent_fitness = list(self.fitness_history)
        
        for i in range(self.convergence_window, len(recent_fitness)):
            window = recent_fitness[i-self.convergence_window:i]
            if max(window) - min(window) < self.stagnation_threshold:
                stagnation_count += 1
        
        return stagnation_count
    
    def plot_training_progress(self, save_path: Optional[Path] = None) -> Optional[str]:
        """绘制训练进度图"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime, timedelta
            
            if not self.metrics_history:
                return None
            
            # 创建子图
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('训练进度监控', fontsize=16)
            
            generations = [m.generation for m in self.metrics_history]
            
            # 1. 适应度进化
            axes[0, 0].plot(generations, [m.best_fitness for m in self.metrics_history], 'b-', label='最佳适应度')
            axes[0, 0].plot(generations, [m.avg_fitness for m in self.metrics_history], 'r--', label='平均适应度')
            axes[0, 0].set_title('适应度进化')
            axes[0, 0].set_xlabel('代数')
            axes[0, 0].set_ylabel('适应度')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 2. 多目标性能
            axes[0, 1].plot(generations, [m.pareto_front_size for m in self.metrics_history], 'g-', label='帕累托前沿大小')
            axes[0, 1].set_title('多目标优化性能')
            axes[0, 1].set_xlabel('代数')
            axes[0, 1].set_ylabel('帕累托前沿大小')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 3. 交易性能
            axes[0, 2].plot(generations, [m.avg_sharpe_ratio for m in self.metrics_history], 'purple', label='夏普比率')
            axes[0, 2].plot(generations, [m.avg_total_return for m in self.metrics_history], 'orange', label='总收益率')
            axes[0, 2].set_title('交易性能指标')
            axes[0, 2].set_xlabel('代数')
            axes[0, 2].set_ylabel('指标值')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
            
            # 4. 数据退火进度
            axes[1, 0].plot(generations, [m.data_ratio for m in self.metrics_history], 'brown', label='数据使用比例')
            axes[1, 0].plot(generations, [m.complexity_score for m in self.metrics_history], 'pink', label='复杂度得分')
            axes[1, 0].set_title('数据退火进度')
            axes[1, 0].set_xlabel('代数')
            axes[1, 0].set_ylabel('比例/得分')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 5. 系统性能
            axes[1, 1].plot(generations, [m.gpu_memory_allocated for m in self.metrics_history], 'red', label='GPU内存(GB)')
            axes[1, 1].plot(generations, [m.generation_time for m in self.metrics_history], 'blue', label='代数时间(秒)')
            axes[1, 1].set_title('系统性能')
            axes[1, 1].set_xlabel('代数')
            axes[1, 1].set_ylabel('资源使用')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            # 6. 收敛性分析
            if len(self.diversity_history) > 0:
                div_generations = generations[-len(self.diversity_history):]
                axes[1, 2].plot(div_generations, list(self.diversity_history), 'cyan', label='种群多样性')
            axes[1, 2].plot(generations, [m.fitness_improvement_rate for m in self.metrics_history], 'magenta', label='适应度改进率')
            axes[1, 2].set_title('收敛性分析')
            axes[1, 2].set_xlabel('代数')
            axes[1, 2].set_ylabel('指标值')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return str(save_path)
            else:
                plt.show()
                return "displayed"
        
        except ImportError:
            self.logger.warning("matplotlib不可用，无法绘制图表")
            return None
        except Exception as e:
            self.logger.error(f"绘制图表失败: {e}")
            return None
    
    def export_detailed_report(self, output_path: Path) -> bool:
        """导出详细报告"""
        try:
            report = {
                'training_summary': self.get_training_summary(),
                'metrics_history': [asdict(m) for m in self.metrics_history],
                'configuration': {
                    'monitoring_config': self._serialize_config(),
                    'convergence_window': self.convergence_window,
                    'stagnation_threshold': self.stagnation_threshold,
                },
                'analysis': {
                    'convergence_achieved': self._check_convergence(),
                    'stagnation_periods': self._count_stagnation_periods(),
                    'training_efficiency': self._calculate_training_efficiency(),
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"详细报告已导出: {output_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"导出报告失败: {e}")
            return False
    
    def _calculate_training_efficiency(self) -> Dict[str, float]:
        """计算训练效率"""
        if not self.metrics_history:
            return {}
        
        total_time = sum(m.generation_time for m in self.metrics_history)
        fitness_improvement = self.best_fitness_ever - self.metrics_history[0].best_fitness if len(self.metrics_history) > 1 else 0
        
        return {
            'fitness_per_second': fitness_improvement / total_time if total_time > 0 else 0,
            'generations_per_minute': len(self.metrics_history) / (total_time / 60) if total_time > 0 else 0,
            'improvement_rate': fitness_improvement / len(self.metrics_history) if self.metrics_history else 0,
        }


if __name__ == "__main__":
    # 测试增强监控系统
    print("=== 增强监控系统测试 ===")
    
    config = MonitoringConfig(
        log_file=Path("test_monitoring.log"),
        save_interval=5,
        detailed_logging=True,
        export_format="both"
    )
    
    monitor = EnhancedMonitor(config)
    monitor.start_monitoring(total_generations=50)
    
    # 模拟训练过程
    for generation in range(50):
        # 模拟基础统计
        basic_stats = {
            'best_fitness': 0.5 + generation * 0.01 + np.random.normal(0, 0.005),
            'avg_fitness': 0.3 + generation * 0.008 + np.random.normal(0, 0.003),
            'std_fitness': 0.1 + np.random.normal(0, 0.01),
            'generation_time': 2.0 + np.random.normal(0, 0.5),
            'no_improvement_count': max(0, 10 - generation // 5),
        }
        
        # 模拟多目标统计
        multi_objective_stats = {
            'pareto_front_size': 20 + generation // 2,
            'hypervolume': 0.1 + generation * 0.002,
            'pareto_ratio': 0.1 + generation * 0.001,
            'objective_stats': {
                'sharpe_ratio': {'mean': 1.0 + generation * 0.01},
                'max_drawdown': {'mean': 0.1 - generation * 0.001},
                'total_return': {'mean': 0.05 + generation * 0.002},
                'win_rate': {'mean': 0.6 + generation * 0.001},
            }
        }
        
        # 模拟数据退火统计
        annealing_stats = {
            'data_ratio': min(1.0, 0.3 + generation * 0.014),
            'complexity_score': min(1.0, generation / 50),
            'strategy': 'progressive',
            'progress': generation / 50,
        }
        
        # 模拟种群
        population = torch.randn(100, 1407)  # 100个个体，每个1407维
        
        # 更新监控
        metrics = monitor.update_metrics(
            generation, basic_stats, multi_objective_stats, 
            annealing_stats, population
        )
        
        if generation % 10 == 0:
            print(f"代数 {generation}: 最佳适应度={metrics.best_fitness:.4f}, "
                  f"帕累托前沿={metrics.pareto_front_size}, "
                  f"数据比例={metrics.data_ratio:.3f}")
    
    # 获取训练总结
    summary = monitor.get_training_summary()
    print(f"\n训练总结:")
    print(f"  总代数: {summary['total_generations']}")
    print(f"  最佳适应度: {summary['best_fitness_ever']:.4f}")
    print(f"  收敛状态: {summary['convergence_achieved']}")
    print(f"  平均代数时间: {summary['avg_generation_time']:.2f}秒")
    
    # 导出报告
    monitor.export_detailed_report(Path("test_detailed_report.json"))
    
    print("\n=== 测试完成 ===")