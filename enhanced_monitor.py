#!/usr/bin/env python3
"""
增强版实时训练监控器
Enhanced Training Monitor for Enhanced CUDA GA
监控 enhanced_training_history.jsonl 文件并实时显示训练进度
"""

import json
import time
import os
import sys
from pathlib import Path
import threading
import queue
from collections import deque

# 检查matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    
    # 设置matplotlib参数避免警告
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    PLOTTING_AVAILABLE = True
    print("📊 Graphics available - showing charts")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("📝 matplotlib not available - text mode")

class EnhancedTrainingMonitor:
    def __init__(self):
        self.log_file = None
        self.data_queue = queue.Queue()
        self.max_points = 200  # 增强版显示更多数据点
        
        # 基础训练数据
        self.generations = deque(maxlen=self.max_points)
        self.best_fitness = deque(maxlen=self.max_points)
        self.avg_fitness = deque(maxlen=self.max_points)
        self.generation_times = deque(maxlen=self.max_points)
        
        # 增强版特有数据
        self.pareto_front_sizes = deque(maxlen=self.max_points)
        self.data_ratios = deque(maxlen=self.max_points)
        self.complexity_scores = deque(maxlen=self.max_points)
        self.population_diversity = deque(maxlen=self.max_points)
        
        # 交易性能数据
        self.sharpe_ratios = deque(maxlen=self.max_points)
        self.max_drawdowns = deque(maxlen=self.max_points)
        self.total_returns = deque(maxlen=self.max_points)
        self.win_rates = deque(maxlen=self.max_points)
        
        # 系统性能数据
        self.gpu_memory = deque(maxlen=self.max_points)
        self.system_memory = deque(maxlen=self.max_points)
        
        self.last_file_size = 0
        self.monitoring = True
        
        if not self.find_log_file():
            sys.exit(1)
    
    def find_log_file(self):
        """查找增强版训练日志文件"""
        paths = [
            # 主要路径
            Path("results/enhanced_training_history.jsonl"),
            Path("../results/enhanced_training_history.jsonl"),
            Path("enhanced_training_history.jsonl"),
            
            # 备份文件
            Path("results/enhanced_training_history.jsonl.backup"),
            Path("../results/enhanced_training_history.jsonl.backup"),
            
            # 兼容旧版本的路径
            Path("results/training_history.jsonl"),
            Path("../results/training_history.jsonl"),
            Path("training_history.jsonl"),
        ]
        
        for path in paths:
            if path.exists():
                self.log_file = path
                print(f"🎯 Found log file: {path}")
                return True
        
        print("❌ Enhanced training log file not found")
        print("Please start enhanced training first: python core/main_enhanced_cuda.py")
        print("Or check if log files exist at these paths:")
        for path in paths:
            print(f"  - {path}")
        return False
    
    def load_data(self):
        """Load historical data"""
        data = []
        encodings = ['utf-8', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(self.log_file, 'r', encoding=encoding, errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"⚠️  JSON parsing error (line {line_num}): {e}")
                                continue
                            except Exception as e:
                                print(f"⚠️  Parsing error (line {line_num}): {e}")
                                continue
                return data
            except Exception as e:
                print(f"⚠️  File reading error (encoding {encoding}): {e}")
                continue
        return []
    
    def extract_data_from_record(self, record):
        """Extract data from record"""
        # 基础数据
        self.generations.append(record.get('generation', 0))
        self.best_fitness.append(record.get('best_fitness', 0))
        self.avg_fitness.append(record.get('avg_fitness', 0))
        self.generation_times.append(record.get('generation_time', 0))
        
        # 增强版特有数据
        self.pareto_front_sizes.append(record.get('pareto_front_size', 0))
        self.data_ratios.append(record.get('data_ratio', 1.0))
        self.complexity_scores.append(record.get('complexity_score', 1.0))
        self.population_diversity.append(record.get('population_diversity', 0.0))
        
        # 交易性能数据
        self.sharpe_ratios.append(record.get('avg_sharpe_ratio', 0.0))
        self.max_drawdowns.append(record.get('avg_max_drawdown', 0.0))
        self.total_returns.append(record.get('avg_total_return', 0.0))
        self.win_rates.append(record.get('avg_win_rate', 0.0))
        
        # 系统性能数据
        self.gpu_memory.append(record.get('gpu_memory_allocated', 0.0))
        self.system_memory.append(record.get('system_memory_gb', 0.0))
    
    def start_monitoring(self):
        """Start monitoring"""
        # Load initial data
        print("📚 Loading historical data...")
        initial_data = self.load_data()
        for record in initial_data:
            self.extract_data_from_record(record)
        
        print(f"✅ Loaded {len(initial_data)} historical records")
        
        if self.log_file.exists():
            self.last_file_size = self.log_file.stat().st_size

        if PLOTTING_AVAILABLE:
            self.setup_plots()
            self.monitor_thread = threading.Thread(target=self.monitor_file, daemon=True)
            self.monitor_thread.start()
            self.show_plots()
        else:
            self.text_mode()
    
    def monitor_file(self):
        """Monitor file changes"""
        while self.monitoring:
            try:
                if self.log_file.exists():
                    current_size = self.log_file.stat().st_size
                    if current_size != self.last_file_size:
                        # 只读取新行
                        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            f.seek(self.last_file_size)
                            new_lines = f.readlines()
                            for line in new_lines:
                                line = line.strip()
                                if line:
                                    try:
                                        self.data_queue.put(json.loads(line))
                                    except json.JSONDecodeError as e:
                                        print(f"⚠️  JSON parsing error: {e}")
                                        continue
                                    except Exception as e:
                                        print(f"⚠️  Data processing error: {e}")
                                        continue
                        self.last_file_size = current_size
                time.sleep(1)
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(5)
    
    def setup_plots(self):
        """Setup plots"""
        self.fig, self.axes = plt.subplots(3, 3, figsize=(18, 12))
        self.fig.suptitle('🚀 Enhanced CUDA Genetic Algorithm Training Monitor', fontsize=16)
        
        # 1. Fitness Evolution (0,0)
        self.ax_fitness = self.axes[0, 0]
        self.line_best, = self.ax_fitness.plot([], [], 'b-', linewidth=2, label='Best Fitness')
        self.line_avg, = self.ax_fitness.plot([], [], 'r--', linewidth=1, label='Average Fitness')
        self.ax_fitness.set_title('🎯 Fitness Evolution')
        self.ax_fitness.set_xlabel('Generation')
        self.ax_fitness.set_ylabel('Fitness')
        self.ax_fitness.legend()
        self.ax_fitness.grid(True, alpha=0.3)
        
        # 2. Multi-Objective Optimization (0,1)
        self.ax_pareto = self.axes[0, 1]
        self.line_pareto, = self.ax_pareto.plot([], [], 'g-', linewidth=2, label='Pareto Front Size')
        self.ax_pareto.set_title('🎯 Multi-Objective Optimization')
        self.ax_pareto.set_xlabel('Generation')
        self.ax_pareto.set_ylabel('Pareto Front Size')
        self.ax_pareto.legend()
        self.ax_pareto.grid(True, alpha=0.3)
        
        # 3. Data Annealing (0,2)
        self.ax_annealing = self.axes[0, 2]
        self.line_data_ratio, = self.ax_annealing.plot([], [], 'orange', linewidth=2, label='Data Usage Ratio')
        self.line_complexity, = self.ax_annealing.plot([], [], 'purple', linewidth=2, label='Complexity Score')
        self.ax_annealing.set_title('🔥 Data Annealing Progress')
        self.ax_annealing.set_xlabel('Generation')
        self.ax_annealing.set_ylabel('Ratio/Score')
        self.ax_annealing.legend()
        self.ax_annealing.grid(True, alpha=0.3)
        
        # 4. Trading Performance (1,0)
        self.ax_trading = self.axes[1, 0]
        self.line_sharpe, = self.ax_trading.plot([], [], 'blue', linewidth=2, label='Sharpe Ratio')
        self.line_return, = self.ax_trading.plot([], [], 'green', linewidth=2, label='Total Return')
        self.ax_trading.set_title('💰 Trading Performance')
        self.ax_trading.set_xlabel('Generation')
        self.ax_trading.set_ylabel('Metric Value')
        self.ax_trading.legend()
        self.ax_trading.grid(True, alpha=0.3)
        
        # 5. Risk Metrics (1,1)
        self.ax_risk = self.axes[1, 1]
        self.line_drawdown, = self.ax_risk.plot([], [], 'red', linewidth=2, label='Max Drawdown')
        self.line_winrate, = self.ax_risk.plot([], [], 'cyan', linewidth=2, label='Win Rate')
        self.ax_risk.set_title('⚠️ Risk Metrics')
        self.ax_risk.set_xlabel('Generation')
        self.ax_risk.set_ylabel('Metric Value')
        self.ax_risk.legend()
        self.ax_risk.grid(True, alpha=0.3)
        
        # 6. System Performance (1,2)
        self.ax_system = self.axes[1, 2]
        self.line_gpu, = self.ax_system.plot([], [], 'red', linewidth=2, label='GPU Memory(GB)')
        self.line_time, = self.ax_system.plot([], [], 'blue', linewidth=2, label='Generation Time(s)')
        self.ax_system.set_title('💻 System Performance')
        self.ax_system.set_xlabel('Generation')
        self.ax_system.set_ylabel('Resource Usage')
        self.ax_system.legend()
        self.ax_system.grid(True, alpha=0.3)
        
        # 7. Population Diversity (2,0)
        self.ax_diversity = self.axes[2, 0]
        self.line_diversity, = self.ax_diversity.plot([], [], 'magenta', linewidth=2, label='Population Diversity')
        self.ax_diversity.set_title('🌈 Population Diversity')
        self.ax_diversity.set_xlabel('Generation')
        self.ax_diversity.set_ylabel('Diversity Metric')
        self.ax_diversity.legend()
        self.ax_diversity.grid(True, alpha=0.3)
        
        # 8. Fitness Distribution (2,1)
        self.ax_dist = self.axes[2, 1]
        self.ax_dist.set_title('📊 Recent Fitness Distribution')
        
        # 9. Statistics (2,2)
        self.ax_stats = self.axes[2, 2]
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.05, 0.95, '', transform=self.ax_stats.transAxes,
                                           fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        self.ani = animation.FuncAnimation(self.fig, self.update_plots,
                                         interval=2000, blit=False, 
                                         cache_frame_data=False)
    
    def update_plots(self, frame):
        """Update plots"""
        # 处理新数据
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                self.extract_data_from_record(data)
            except:
                break
        
        if len(self.generations) == 0:
            return
        
        gens = list(self.generations)
        
        # 更新适应度图
        self.line_best.set_data(gens, list(self.best_fitness))
        self.line_avg.set_data(gens, list(self.avg_fitness))
        self.ax_fitness.relim()
        self.ax_fitness.autoscale_view()
        
        # 更新多目标优化图
        self.line_pareto.set_data(gens, list(self.pareto_front_sizes))
        self.ax_pareto.relim()
        self.ax_pareto.autoscale_view()
        
        # 更新数据退火图
        self.line_data_ratio.set_data(gens, list(self.data_ratios))
        self.line_complexity.set_data(gens, list(self.complexity_scores))
        self.ax_annealing.relim()
        self.ax_annealing.autoscale_view()
        
        # 更新交易性能图
        self.line_sharpe.set_data(gens, list(self.sharpe_ratios))
        self.line_return.set_data(gens, list(self.total_returns))
        self.ax_trading.relim()
        self.ax_trading.autoscale_view()
        
        # 更新风险指标图
        self.line_drawdown.set_data(gens, list(self.max_drawdowns))
        self.line_winrate.set_data(gens, list(self.win_rates))
        self.ax_risk.relim()
        self.ax_risk.autoscale_view()
        
        # 更新系统性能图
        self.line_gpu.set_data(gens, list(self.gpu_memory))
        self.line_time.set_data(gens, list(self.generation_times))
        self.ax_system.relim()
        self.ax_system.autoscale_view()
        
        # 更新种群多样性图
        if len(self.population_diversity) > 0:
            self.line_diversity.set_data(gens, list(self.population_diversity))
            self.ax_diversity.relim()
            self.ax_diversity.autoscale_view()
        
        # Update fitness distribution
        if len(self.best_fitness) >= 20:
            self.ax_dist.clear()
            recent_fitness = list(self.best_fitness)[-30:]
            self.ax_dist.hist(recent_fitness, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
            self.ax_dist.set_title('📊 Recent Fitness Distribution')
            self.ax_dist.grid(True, alpha=0.3)
        
        # Update statistics
        if len(self.generations) > 0:
            self.update_stats_text()
    
    def update_stats_text(self):
        """Update statistics text"""
        current_gen = self.generations[-1]
        current_best = self.best_fitness[-1]
        current_avg = self.avg_fitness[-1]
        current_time = self.generation_times[-1] if self.generation_times else 0
        
        # 计算统计信息
        total_time = sum(self.generation_times)
        avg_time = total_time / len(self.generation_times) if self.generation_times else 0
        best_ever = max(self.best_fitness) if self.best_fitness else 0
        
        # 增强版特有信息
        current_pareto = self.pareto_front_sizes[-1] if self.pareto_front_sizes else 0
        current_data_ratio = self.data_ratios[-1] if self.data_ratios else 1.0
        current_complexity = self.complexity_scores[-1] if self.complexity_scores else 1.0
        current_diversity = self.population_diversity[-1] if self.population_diversity else 0.0
        
        # 交易性能
        current_sharpe = self.sharpe_ratios[-1] if self.sharpe_ratios else 0.0
        current_drawdown = self.max_drawdowns[-1] if self.max_drawdowns else 0.0
        current_return = self.total_returns[-1] if self.total_returns else 0.0
        current_winrate = self.win_rates[-1] if self.win_rates else 0.0
        
        # 系统性能
        current_gpu = self.gpu_memory[-1] if self.gpu_memory else 0.0
        current_sys_mem = self.system_memory[-1] if self.system_memory else 0.0
        
        stats_info = f"""🚀 Enhanced Training Statistics

📈 Basic Metrics:
  Generation: {current_gen}
  Best Fitness: {current_best:.6f}
  Avg Fitness: {current_avg:.6f}
  Best Ever: {best_ever:.6f}

⏱️ Time Statistics:
  Current Gen Time: {current_time:.2f}s
  Avg Gen Time: {avg_time:.2f}s
  Total Time: {total_time/3600:.2f}h

🎯 Multi-Objective:
  Pareto Front: {current_pareto}
  
🔥 Data Annealing:
  Data Usage Ratio: {current_data_ratio:.3f}
  Complexity Score: {current_complexity:.3f}

💰 Trading Performance:
  Sharpe Ratio: {current_sharpe:.3f}
  Max Drawdown: {current_drawdown:.3f}
  Total Return: {current_return:.3f}
  Win Rate: {current_winrate:.3f}

🌈 Algorithm Status:
  Population Diversity: {current_diversity:.3f}

💻 System Resources:
  GPU Memory: {current_gpu:.2f}GB
  System Memory: {current_sys_mem:.2f}GB

📊 Data Points: {len(self.generations)}
🕒 Update Time: {time.strftime('%H:%M:%S')}
"""
        self.stats_text.set_text(stats_info)
    
    def show_plots(self):
        """Show plots"""
        print("🚀 Starting enhanced chart monitoring")
        print("Close window to stop monitoring")
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped")
        finally:
            self.monitoring = False
    
    def text_mode(self):
        """Text mode monitoring"""
        print("📝 Text mode monitoring")
        print("Press Ctrl+C to stop\n")
        
        last_size = 0
        
        try:
            while True:
                if self.log_file.exists():
                    current_size = self.log_file.stat().st_size
                    if current_size != last_size:
                        data = self.load_data()
                        if data:
                            self.display_text(data[-1], len(data))
                        last_size = current_size
                    else:
                        print(f"\r⏳ Waiting for updates... {time.strftime('%H:%M:%S')}", end="", flush=True)
                else:
                    print(f"\r⏳ Waiting for log file... {time.strftime('%H:%M:%S')}", end="", flush=True)
                
                time.sleep(3)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped")
    
    def display_text(self, data, total_count):
        """Display text information"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 80)
        print("           🚀 Enhanced CUDA Genetic Algorithm Training Monitor")
        print("=" * 80)
        
        # Basic information
        print(f"📈 Generation: {data.get('generation', 0)}")
        print(f"🎯 Best Fitness: {data.get('best_fitness', 0):.6f}")
        print(f"📊 Average Fitness: {data.get('avg_fitness', 0):.6f}")
        print(f"⏱️  Generation Time: {data.get('generation_time', 0):.2f}s")
        
        # Enhanced version specific information
        print(f"\n🔥 Data Annealing:")
        print(f"   Data Usage Ratio: {data.get('data_ratio', 1.0):.3f}")
        print(f"   Complexity Score: {data.get('complexity_score', 1.0):.3f}")
        
        print(f"\n🎯 Multi-Objective Optimization:")
        print(f"   Pareto Front Size: {data.get('pareto_front_size', 0)}")
        
        print(f"\n💰 Trading Performance:")
        print(f"   Sharpe Ratio: {data.get('avg_sharpe_ratio', 0.0):.3f}")
        print(f"   Max Drawdown: {data.get('avg_max_drawdown', 0.0):.3f}")
        print(f"   Total Return: {data.get('avg_total_return', 0.0):.3f}")
        print(f"   Win Rate: {data.get('avg_win_rate', 0.0):.3f}")
        
        print(f"\n🌈 Algorithm Status:")
        print(f"   Population Diversity: {data.get('population_diversity', 0.0):.3f}")
        
        # System performance
        print(f"\n💻 System Performance:")
        if 'gpu_memory_allocated' in data:
            print(f"   GPU Memory: {data['gpu_memory_allocated']:.2f}GB")
        if 'system_memory_gb' in data:
            print(f"   System Memory: {data['system_memory_gb']:.2f}GB")
        
        print(f"\n📊 Total Records: {total_count}")
        print("=" * 80)
        print(f"🕒 Update Time: {time.strftime('%H:%M:%S')}")
        print("Press Ctrl+C to stop monitoring")
        print("=" * 80)

def main():
    print("🚀 Enhanced CUDA Genetic Algorithm Training Monitor")
    print("=" * 50)
    
    try:
        monitor = EnhancedTrainingMonitor()
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()