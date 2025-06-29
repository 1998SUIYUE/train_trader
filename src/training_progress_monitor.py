"""
训练进度监控器
提供实时的训练进度显示和统计信息
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading
import queue

class TrainingProgressMonitor:
    """训练进度监控器"""
    
    def __init__(self, log_file: Optional[Path] = None, update_interval: float = 1.0):
        """
        初始化进度监控器
        
        Args:
            log_file: 日志文件路径
            update_interval: 更新间隔（秒）
        """
        self.log_file = log_file
        self.update_interval = update_interval
        
        # 训练统计
        self.start_time = None
        self.generation_times = []
        self.fitness_history = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.improvement_history = []
        
        # 当前状态
        self.current_generation = 0
        self.total_generations = 0
        self.best_fitness = -float('inf')
        self.no_improvement_count = 0
        self.early_stop_patience = 50
        
        # 实时显示控制
        self.display_enabled = True
        self.detailed_stats = True
        self.show_progress_bar = True
        
        # 线程安全的消息队列
        self.message_queue = queue.Queue()
        self.display_thread = None
        self.stop_display = False
        
    def start_training(self, total_generations: int, early_stop_patience: int = 50):
        """开始训练监控"""
        self.start_time = time.time()
        self.total_generations = total_generations
        self.early_stop_patience = early_stop_patience
        self.current_generation = 0
        
        # 清空历史记录
        self.generation_times.clear()
        self.fitness_history.clear()
        self.best_fitness_history.clear()
        self.avg_fitness_history.clear()
        self.improvement_history.clear()
        
        print("=" * 80)
        print("🚀 CUDA遗传算法训练开始")
        print("=" * 80)
        print(f"目标代数: {total_generations if total_generations > 0 else '无限'}")
        print(f"早停耐心: {early_stop_patience}")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # 启动实时显示线程
        if self.display_enabled:
            self.start_display_thread()
    
    def update_generation(self, generation: int, stats: Dict[str, Any]):
        """更新代数信息"""
        self.current_generation = generation
        
        # 记录统计信息
        generation_time = stats.get('generation_time', 0)
        best_fitness = stats.get('best_fitness', 0)
        avg_fitness = stats.get('avg_fitness', 0)
        std_fitness = stats.get('std_fitness', 0)
        no_improvement = stats.get('no_improvement_count', 0)
        
        self.generation_times.append(generation_time)
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.no_improvement_count = no_improvement
        
        # 检查是否有改进
        improved = best_fitness > self.best_fitness
        if improved:
            self.best_fitness = best_fitness
            self.improvement_history.append(generation)
        
        # 准备显示信息
        display_info = {
            'generation': generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'std_fitness': std_fitness,
            'generation_time': generation_time,
            'no_improvement': no_improvement,
            'improved': improved,
            'total_time': time.time() - self.start_time if self.start_time else 0
        }
        
        # 发送到显示线程
        if self.display_enabled:
            self.message_queue.put(display_info)
        
        # 写入日志文件
        if self.log_file:
            self.write_log(display_info)
    
    def start_display_thread(self):
        """启动实时显示线程"""
        self.stop_display = False
        self.display_thread = threading.Thread(target=self._display_worker, daemon=True)
        self.display_thread.start()
    
    def _display_worker(self):
        """显示工作线程"""
        while not self.stop_display:
            try:
                # 等待新的显示信息
                display_info = self.message_queue.get(timeout=self.update_interval)
                self._display_generation_info(display_info)
                self.message_queue.task_done()
            except queue.Empty:
                # 超时，显示当前状态
                if self.current_generation > 0:
                    self._display_status_update()
            except Exception as e:
                print(f"显示线程错误: {e}")
    
    def _display_generation_info(self, info: Dict[str, Any]):
        """显示代数信息"""
        generation = info['generation']
        best_fitness = info['best_fitness']
        avg_fitness = info['avg_fitness']
        std_fitness = info['std_fitness']
        generation_time = info['generation_time']
        no_improvement = info['no_improvement']
        improved = info['improved']
        total_time = info['total_time']
        
        # 基本信息行
        improvement_indicator = "🔥" if improved else "  "
        progress_info = ""
        
        if self.total_generations > 0:
            progress_pct = (generation / self.total_generations) * 100
            progress_info = f"({progress_pct:5.1f}%)"
        
        print(f"\r{improvement_indicator} 代数 {generation:4d} {progress_info} | "
              f"最佳: {best_fitness:8.6f} | "
              f"平均: {avg_fitness:8.6f} | "
              f"标准差: {std_fitness:7.6f} | "
              f"无改进: {no_improvement:3d} | "
              f"时间: {generation_time:6.3f}s", end="")
        
        # 详细统计信息（每10代显示一次）
        if self.detailed_stats and generation % 10 == 0:
            self._display_detailed_stats(total_time)
        
        # 进度条（每20代显示一次）
        if self.show_progress_bar and generation % 20 == 0 and self.total_generations > 0:
            self._display_progress_bar(generation)
    
    def _display_detailed_stats(self, total_time: float):
        """显示详细统计信息"""
        print()  # 换行
        
        # 时间统计
        avg_time = np.mean(self.generation_times[-10:]) if self.generation_times else 0
        total_hours = total_time / 3600
        
        # 适应度统计
        recent_best = self.best_fitness_history[-10:] if self.best_fitness_history else [0]
        recent_avg = self.avg_fitness_history[-10:] if self.avg_fitness_history else [0]
        
        fitness_trend = "📈" if len(recent_best) > 1 and recent_best[-1] > recent_best[0] else "📉"
        
        # ETA估算
        eta_info = ""
        if self.total_generations > 0 and avg_time > 0:
            remaining_gens = self.total_generations - self.current_generation
            eta_seconds = remaining_gens * avg_time
            eta_hours = eta_seconds / 3600
            eta_info = f"ETA: {eta_hours:.1f}h"
        
        print(f"    📊 统计 | 总时间: {total_hours:.2f}h | 平均每代: {avg_time:.3f}s | {eta_info}")
        print(f"    {fitness_trend} 趋势 | 最近最佳: {np.max(recent_best):.6f} | 最近平均: {np.mean(recent_avg):.6f}")
        
        # GPU内存信息（如果可用）
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                cached = torch.cuda.memory_reserved() / 1e9
                print(f"    🖥️  GPU内存 | 已分配: {allocated:.2f}GB | 已缓存: {cached:.2f}GB")
        except:
            pass
    
    def _display_progress_bar(self, generation: int):
        """显示进度条"""
        if self.total_generations <= 0:
            return
        
        progress = generation / self.total_generations
        bar_length = 50
        filled_length = int(bar_length * progress)
        
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        print(f"\n    📈 进度 [{bar}] {progress*100:.1f}%")
    
    def _display_status_update(self):
        """显示状态更新"""
        if self.current_generation > 0:
            total_time = time.time() - self.start_time if self.start_time else 0
            print(f"\r    ⏱️  运行中... 代数: {self.current_generation}, 总时间: {total_time/3600:.2f}h", end="")
    
    def write_log(self, info: Dict[str, Any]):
        """写入日志文件"""
        if not self.log_file:
            return
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"写入日志失败: {e}")
    
    def display_final_summary(self, results: Dict[str, Any]):
        """显示最终总结"""
        self.stop_display = True
        if self.display_thread:
            self.display_thread.join(timeout=1.0)
        
        print("\n\n" + "=" * 80)
        print("🎉 训练完成！")
        print("=" * 80)
        
        total_time = results.get('total_time', 0)
        final_generation = results.get('final_generation', 0)
        best_fitness = results.get('best_fitness', 0)
        
        # 基本信息
        print(f"📊 训练统计:")
        print(f"   总代数: {final_generation}")
        print(f"   最佳适应度: {best_fitness:.8f}")
        print(f"   总训练时间: {total_time/3600:.2f} 小时")
        print(f"   平均每代时间: {total_time/final_generation:.3f} 秒")
        
        # 改进历史
        if self.improvement_history:
            print(f"   总改进次数: {len(self.improvement_history)}")
            print(f"   最后改进代数: {self.improvement_history[-1]}")
            improvement_rate = len(self.improvement_history) / final_generation * 100
            print(f"   改进率: {improvement_rate:.1f}%")
        
        # 性能统计
        if self.generation_times:
            avg_time = np.mean(self.generation_times)
            min_time = np.min(self.generation_times)
            max_time = np.max(self.generation_times)
            print(f"\n⏱️  性能统计:")
            print(f"   平均每代: {avg_time:.3f}s")
            print(f"   最快一代: {min_time:.3f}s")
            print(f"   最慢一代: {max_time:.3f}s")
        
        # 适应度统计
        if self.best_fitness_history:
            initial_fitness = self.best_fitness_history[0]
            final_fitness = self.best_fitness_history[-1]
            improvement = final_fitness - initial_fitness
            print(f"\n📈 适应度统计:")
            print(f"   初始适应度: {initial_fitness:.8f}")
            print(f"   最终适应度: {final_fitness:.8f}")
            print(f"   总体改进: {improvement:.8f} ({improvement/abs(initial_fitness)*100:.1f}%)")
        
        print("=" * 80)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'current_generation': self.current_generation,
            'total_time': total_time,
            'avg_generation_time': np.mean(self.generation_times) if self.generation_times else 0,
            'best_fitness': self.best_fitness,
            'improvement_count': len(self.improvement_history),
            'no_improvement_count': self.no_improvement_count,
            'fitness_history': self.best_fitness_history.copy(),
            'generation_times': self.generation_times.copy()
        }


class SimpleProgressDisplay:
    """简化的进度显示器（不使用线程）"""
    
    def __init__(self):
        self.start_time = None
        self.best_fitness = -float('inf')
        self.generation_count = 0
        
    def start_training(self, total_generations: int):
        """开始训练"""
        self.start_time = time.time()
        print("🚀 开始CUDA训练...")
        print(f"目标代数: {total_generations if total_generations > 0 else '无限'}")
        print("-" * 60)
    
    def update_generation(self, generation: int, stats: Dict[str, Any]):
        """更新代数信息"""
        self.generation_count = generation
        
        best_fitness = stats.get('best_fitness', 0)
        avg_fitness = stats.get('avg_fitness', 0)
        generation_time = stats.get('generation_time', 0)
        no_improvement = stats.get('no_improvement_count', 0)
        
        # 检查改进
        improved = best_fitness > self.best_fitness
        if improved:
            self.best_fitness = best_fitness
        
        # 显示信息
        indicator = "🔥" if improved else "  "
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print(f"{indicator} Gen {generation:4d} | "
              f"Best: {best_fitness:8.6f} | "
              f"Avg: {avg_fitness:8.6f} | "
              f"NoImpr: {no_improvement:3d} | "
              f"Time: {generation_time:6.3f}s | "
              f"Total: {total_time/60:.1f}m")
        
        # 每50代显示一次详细信息
        if generation % 50 == 0:
            self.display_milestone(generation, total_time)
    
    def display_milestone(self, generation: int, total_time: float):
        """显示里程碑信息"""
        print(f"    📊 里程碑 | 代数: {generation} | 总时间: {total_time/3600:.2f}h")
        
        # GPU内存信息
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"    🖥️  GPU内存: {allocated:.2f}GB")
        except:
            pass
    
    def display_final_summary(self, results: Dict[str, Any]):
        """显示最终总结"""
        print("\n" + "=" * 60)
        print("🎉 训练完成！")
        print("=" * 60)
        print(f"最佳适应度: {results.get('best_fitness', 0):.8f}")
        print(f"总代数: {results.get('final_generation', 0)}")
        print(f"总时间: {results.get('total_time', 0)/3600:.2f} 小时")
        print("=" * 60)


if __name__ == "__main__":
    # 测试进度监控器
    import random
    
    print("测试训练进度监控器...")
    
    # 创建监控器
    monitor = TrainingProgressMonitor()
    monitor.start_training(total_generations=100)
    
    # 模拟训练过程
    for gen in range(100):
        # 模拟训练统计
        stats = {
            'generation': gen,
            'best_fitness': random.uniform(0, 1) + gen * 0.01,
            'avg_fitness': random.uniform(0, 0.8) + gen * 0.005,
            'std_fitness': random.uniform(0.1, 0.3),
            'generation_time': random.uniform(0.5, 2.0),
            'no_improvement_count': random.randint(0, 10)
        }
        
        monitor.update_generation(gen, stats)
        time.sleep(0.1)  # 模拟训练时间
    
    # 显示最终总结
    final_results = {
        'best_fitness': 1.5,
        'final_generation': 100,
        'total_time': 150.0
    }
    
    monitor.display_final_summary(final_results)