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
    print("📊 图形界面可用 - 显示图表")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("📝 matplotlib不可用 - 文本模式")

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
                print(f"🎯 找到日志文件: {path}")
                return True
        
        print("❌ 未找到增强版训练日志文件")
        print("请先启动增强版训练: python core/main_enhanced_cuda.py")
        print("或检查以下路径是否存在日志文件:")
        for path in paths:
            print(f"  - {path}")
        return False
    
    def load_data(self):
        """加载历史数据"""
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
                                print(f"⚠️  JSON解析错误 (行{line_num}): {e}")
                                continue
                            except Exception as e:
                                print(f"⚠️  解析错误 (行{line_num}): {e}")
                                continue
                return data
            except Exception as e:
                print(f"⚠️  文件读取错误 (编码{encoding}): {e}")
                continue
        return []
    
    def extract_data_from_record(self, record):
        """从记录中提取数据"""
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
        """开始监控"""
        # 加载初始数据
        print("📚 加载历史数据...")
        initial_data = self.load_data()
        for record in initial_data:
            self.extract_data_from_record(record)
        
        print(f"✅ 已加载 {len(initial_data)} 条历史记录")
        
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
        """监控文件变化"""
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
                                        print(f"⚠️  JSON解析错误: {e}")
                                        continue
                                    except Exception as e:
                                        print(f"⚠️  数据处理错误: {e}")
                                        continue
                        self.last_file_size = current_size
                time.sleep(1)
            except Exception as e:
                print(f"❌ 监控错误: {e}")
                time.sleep(5)
    
    def setup_plots(self):
        """设置图表"""
        self.fig, self.axes = plt.subplots(3, 3, figsize=(18, 12))
        self.fig.suptitle('🚀 增强版CUDA遗传算法训练监控', fontsize=16)
        
        # 1. 适应度进化 (0,0)
        self.ax_fitness = self.axes[0, 0]
        self.line_best, = self.ax_fitness.plot([], [], 'b-', linewidth=2, label='最佳适应度')
        self.line_avg, = self.ax_fitness.plot([], [], 'r--', linewidth=1, label='平均适应度')
        self.ax_fitness.set_title('🎯 适应度进化')
        self.ax_fitness.set_xlabel('代数')
        self.ax_fitness.set_ylabel('适应度')
        self.ax_fitness.legend()
        self.ax_fitness.grid(True, alpha=0.3)
        
        # 2. 多目标优化 (0,1)
        self.ax_pareto = self.axes[0, 1]
        self.line_pareto, = self.ax_pareto.plot([], [], 'g-', linewidth=2, label='帕累托前沿大小')
        self.ax_pareto.set_title('🎯 多目标优化')
        self.ax_pareto.set_xlabel('代数')
        self.ax_pareto.set_ylabel('帕累托前沿大小')
        self.ax_pareto.legend()
        self.ax_pareto.grid(True, alpha=0.3)
        
        # 3. 数据退火 (0,2)
        self.ax_annealing = self.axes[0, 2]
        self.line_data_ratio, = self.ax_annealing.plot([], [], 'orange', linewidth=2, label='数据使用比例')
        self.line_complexity, = self.ax_annealing.plot([], [], 'purple', linewidth=2, label='复杂度得分')
        self.ax_annealing.set_title('🔥 数据退火进度')
        self.ax_annealing.set_xlabel('代数')
        self.ax_annealing.set_ylabel('比例/得分')
        self.ax_annealing.legend()
        self.ax_annealing.grid(True, alpha=0.3)
        
        # 4. 交易性能 (1,0)
        self.ax_trading = self.axes[1, 0]
        self.line_sharpe, = self.ax_trading.plot([], [], 'blue', linewidth=2, label='夏普比率')
        self.line_return, = self.ax_trading.plot([], [], 'green', linewidth=2, label='总收益率')
        self.ax_trading.set_title('💰 交易性能')
        self.ax_trading.set_xlabel('代数')
        self.ax_trading.set_ylabel('指标值')
        self.ax_trading.legend()
        self.ax_trading.grid(True, alpha=0.3)
        
        # 5. 风险指标 (1,1)
        self.ax_risk = self.axes[1, 1]
        self.line_drawdown, = self.ax_risk.plot([], [], 'red', linewidth=2, label='最大回撤')
        self.line_winrate, = self.ax_risk.plot([], [], 'cyan', linewidth=2, label='胜率')
        self.ax_risk.set_title('⚠️ 风险指标')
        self.ax_risk.set_xlabel('代数')
        self.ax_risk.set_ylabel('指标值')
        self.ax_risk.legend()
        self.ax_risk.grid(True, alpha=0.3)
        
        # 6. 系统性能 (1,2)
        self.ax_system = self.axes[1, 2]
        self.line_gpu, = self.ax_system.plot([], [], 'red', linewidth=2, label='GPU内存(GB)')
        self.line_time, = self.ax_system.plot([], [], 'blue', linewidth=2, label='代数时间(s)')
        self.ax_system.set_title('💻 系统性能')
        self.ax_system.set_xlabel('代数')
        self.ax_system.set_ylabel('资源使用')
        self.ax_system.legend()
        self.ax_system.grid(True, alpha=0.3)
        
        # 7. 种群多样性 (2,0)
        self.ax_diversity = self.axes[2, 0]
        self.line_diversity, = self.ax_diversity.plot([], [], 'magenta', linewidth=2, label='种群多样性')
        self.ax_diversity.set_title('🌈 种群多样性')
        self.ax_diversity.set_xlabel('代数')
        self.ax_diversity.set_ylabel('多样性指标')
        self.ax_diversity.legend()
        self.ax_diversity.grid(True, alpha=0.3)
        
        # 8. 适应度分布 (2,1)
        self.ax_dist = self.axes[2, 1]
        self.ax_dist.set_title('📊 最近适应度分布')
        
        # 9. 统计信息 (2,2)
        self.ax_stats = self.axes[2, 2]
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.05, 0.95, '', transform=self.ax_stats.transAxes,
                                           fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        self.ani = animation.FuncAnimation(self.fig, self.update_plots,
                                         interval=2000, blit=False, 
                                         cache_frame_data=False)
    
    def update_plots(self, frame):
        """更新图表"""
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
        
        # 更新适应度分布
        if len(self.best_fitness) >= 20:
            self.ax_dist.clear()
            recent_fitness = list(self.best_fitness)[-30:]
            self.ax_dist.hist(recent_fitness, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
            self.ax_dist.set_title('📊 最近适应度分布')
            self.ax_dist.grid(True, alpha=0.3)
        
        # 更新统计信息
        if len(self.generations) > 0:
            self.update_stats_text()
    
    def update_stats_text(self):
        """更新统计文本"""
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
        
        stats_info = f"""🚀 增强版训练统计

📈 基础指标:
  代数: {current_gen}
  最佳适应度: {current_best:.6f}
  平均适应度: {current_avg:.6f}
  历史最佳: {best_ever:.6f}

⏱️ 时间统计:
  当前代时间: {current_time:.2f}s
  平均代时间: {avg_time:.2f}s
  总训练时间: {total_time/3600:.2f}h

🎯 多目标优化:
  帕累托前沿: {current_pareto}
  
🔥 数据退火:
  数据使用比例: {current_data_ratio:.3f}
  复杂度得分: {current_complexity:.3f}

💰 交易性能:
  夏普比率: {current_sharpe:.3f}
  最大回撤: {current_drawdown:.3f}
  总收益率: {current_return:.3f}
  胜率: {current_winrate:.3f}

🌈 算法状态:
  种群多样性: {current_diversity:.3f}

💻 系统资源:
  GPU内存: {current_gpu:.2f}GB
  系统内存: {current_sys_mem:.2f}GB

📊 数据点数: {len(self.generations)}
🕒 更新时间: {time.strftime('%H:%M:%S')}
"""
        self.stats_text.set_text(stats_info)
    
    def show_plots(self):
        """显示图表"""
        print("🚀 启动增强版图表监控")
        print("关闭窗口停止监控")
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n⏹️  监控已停止")
        finally:
            self.monitoring = False
    
    def text_mode(self):
        """文本模式监控"""
        print("📝 文本模式监控")
        print("按 Ctrl+C 停止\n")
        
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
                        print(f"\r⏳ 等待更新... {time.strftime('%H:%M:%S')}", end="", flush=True)
                else:
                    print(f"\r⏳ 等待日志文件... {time.strftime('%H:%M:%S')}", end="", flush=True)
                
                time.sleep(3)
                
        except KeyboardInterrupt:
            print("\n⏹️  监控已停止")
    
    def display_text(self, data, total_count):
        """显示文本信息"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 80)
        print("           🚀 增强版CUDA遗传算法训练监控")
        print("=" * 80)
        
        # 基础信息
        print(f"📈 代数: {data.get('generation', 0)}")
        print(f"🎯 最佳适应度: {data.get('best_fitness', 0):.6f}")
        print(f"📊 平均适应度: {data.get('avg_fitness', 0):.6f}")
        print(f"⏱️  代数时间: {data.get('generation_time', 0):.2f}s")
        
        # 增强版特有信息
        print(f"\n🔥 数据退火:")
        print(f"   数据使用比例: {data.get('data_ratio', 1.0):.3f}")
        print(f"   复杂度得分: {data.get('complexity_score', 1.0):.3f}")
        
        print(f"\n🎯 多目标优化:")
        print(f"   帕累托前沿大小: {data.get('pareto_front_size', 0)}")
        
        print(f"\n💰 交易性能:")
        print(f"   夏普比率: {data.get('avg_sharpe_ratio', 0.0):.3f}")
        print(f"   最大回撤: {data.get('avg_max_drawdown', 0.0):.3f}")
        print(f"   总收益率: {data.get('avg_total_return', 0.0):.3f}")
        print(f"   胜率: {data.get('avg_win_rate', 0.0):.3f}")
        
        print(f"\n🌈 算法状态:")
        print(f"   种群多样性: {data.get('population_diversity', 0.0):.3f}")
        
        # 系统性能
        print(f"\n💻 系统性能:")
        if 'gpu_memory_allocated' in data:
            print(f"   GPU内存: {data['gpu_memory_allocated']:.2f}GB")
        if 'system_memory_gb' in data:
            print(f"   系统内存: {data['system_memory_gb']:.2f}GB")
        
        print(f"\n📊 总记录数: {total_count}")
        print("=" * 80)
        print(f"🕒 更新时间: {time.strftime('%H:%M:%S')}")
        print("按 Ctrl+C 停止监控")
        print("=" * 80)

def main():
    print("🚀 增强版CUDA遗传算法训练监控器")
    print("=" * 50)
    
    try:
        monitor = EnhancedTrainingMonitor()
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n⏹️  监控被中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()