#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无字体问题的实时训练监控器
使用纯英文界面，避免所有字体相关警告
"""

import json
import time
import os
import sys
from pathlib import Path
import threading
import queue
from collections import deque

# 检查matplotlib是否可用
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    PLOTTING_AVAILABLE = True
    print("Graphics library available - will show dynamic charts")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("matplotlib not available - will use text mode")

class CleanTrainingMonitor:
    """清洁版训练监控器 - 无字体警告"""
    
    def __init__(self):
        self.log_file = None
        self.data_queue = queue.Queue()
        self.max_points = 100
        
        # 数据存储
        self.generations = deque(maxlen=self.max_points)
        self.best_fitness = deque(maxlen=self.max_points)
        self.mean_fitness = deque(maxlen=self.max_points)
        self.generation_times = deque(maxlen=self.max_points)
        
        # 文件监控
        self.last_file_size = 0
        self.monitoring = True
        
        # 查找日志文件
        if not self.find_log_file():
            sys.exit(1)
    
    def find_log_file(self):
        """查找训练日志文件"""
        possible_paths = [
            Path("results/training_history_cuda.jsonl"),
            Path("results/training_history.jsonl"),
            Path("training_history_cuda.jsonl"), 
            Path("training_history.jsonl"),
            Path("../results/training_history_cuda.jsonl"),
            Path("../results/training_history.jsonl")
        ]
        
        for path in possible_paths:
            if path.exists():
                self.log_file = path
                print(f"Found log file: {path}")
                return True
        
        print("No training log file found")
        print("Please ensure training has started and generated log files")
        print("\nPossible log file locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nTip: Run 'python core/main_cuda.py' to start training first")
        return False
    
    def load_data_safe(self):
        """安全地加载数据"""
        data = []
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(self.log_file, 'r', encoding=encoding, errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                return data
            except Exception:
                continue
        
        print("Warning: Unable to read log file")
        return []
    
    def start_monitoring(self):
        """启动文件监控"""
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
                        data = self.load_data_safe()
                        if data:
                            # 只添加新数据
                            for item in data[len(self.generations):]:
                                self.data_queue.put(item)
                        self.last_file_size = current_size
                
                time.sleep(1)
            except Exception as e:
                print(f"Error monitoring file: {e}")
                time.sleep(5)
    
    def setup_plots(self):
        """设置图表 - 使用默认字体避免警告"""
        # 使用默认字体设置
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
        
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('CUDA Training Real-time Monitor', fontsize=16)
        
        # 适应度曲线
        self.ax1 = self.axes[0, 0]
        self.line_best, = self.ax1.plot([], [], 'b-', linewidth=2, label='Best Fitness')
        self.line_mean, = self.ax1.plot([], [], 'r--', linewidth=1, label='Mean Fitness')
        self.ax1.set_title('Fitness Evolution')
        self.ax1.set_xlabel('Generation')
        self.ax1.set_ylabel('Fitness')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # 训练时间
        self.ax2 = self.axes[0, 1]
        self.line_time, = self.ax2.plot([], [], 'g-', linewidth=2)
        self.ax2.set_title('Training Time per Generation')
        self.ax2.set_xlabel('Generation')
        self.ax2.set_ylabel('Time (seconds)')
        self.ax2.grid(True, alpha=0.3)
        
        # 适应度分布
        self.ax3 = self.axes[1, 0]
        self.ax3.set_title('Recent Fitness Distribution')
        
        # 统计信息
        self.ax4 = self.axes[1, 1]
        self.ax4.axis('off')
        self.stats_text = self.ax4.text(0.05, 0.95, '', transform=self.ax4.transAxes,
                                       fontsize=9, verticalalignment='top',
                                       fontfamily='monospace')
        
        plt.tight_layout()
        
        # 创建动画，避免警告
        self.ani = animation.FuncAnimation(self.fig, self.update_plots,
                                         interval=2000, blit=False, 
                                         cache_frame_data=False)
    
    def update_plots(self, frame):
        """更新图表"""
        # 处理新数据
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                self.generations.append(data.get('generation', 0))
                self.best_fitness.append(data.get('best_fitness', 0))
                self.mean_fitness.append(data.get('avg_fitness', data.get('mean_fitness', 0)))
                self.generation_times.append(data.get('generation_time', 0))
            except queue.Empty:
                break
        
        if len(self.generations) == 0:
            return
        
        # 更新数据
        gens = list(self.generations)
        best_fit = list(self.best_fitness)
        mean_fit = list(self.mean_fitness)
        times = list(self.generation_times)
        
        # 更新适应度曲线
        self.line_best.set_data(gens, best_fit)
        self.line_mean.set_data(gens, mean_fit)
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # 更新训练时间
        self.line_time.set_data(gens, times)
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # 更新适应度分布
        if len(best_fit) >= 10:
            self.ax3.clear()
            self.ax3.hist(best_fit[-20:], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            self.ax3.set_title('Recent 20 Generations Fitness')
            self.ax3.set_xlabel('Fitness')
            self.ax3.set_ylabel('Count')
            self.ax3.grid(True, alpha=0.3)
        
        # 更新统计信息
        if len(self.generations) > 0:
            current_gen = self.generations[-1]
            current_best = self.best_fitness[-1]
            current_mean = self.mean_fitness[-1]
            current_time = self.generation_times[-1] if self.generation_times else 0
            
            total_time = sum(self.generation_times)
            avg_time = total_time / len(self.generation_times) if self.generation_times else 0
            best_ever = max(self.best_fitness)
            
            # 计算改进趋势
            if len(self.best_fitness) >= 5:
                recent_trend = np.polyfit(range(5), list(self.best_fitness)[-5:], 1)[0]
                trend_str = "Improving" if recent_trend > 0 else "Declining" if recent_trend < 0 else "Stable"
            else:
                trend_str = "Collecting..."
            
            stats_info = f"""Training Statistics

Current Generation: {current_gen}
Current Best: {current_best:.6f}
Current Mean: {current_mean:.6f}
Generation Time: {current_time:.2f}s

Best Ever: {best_ever:.6f}
Total Time: {total_time/3600:.2f}h
Average Time: {avg_time:.2f}s
Data Points: {len(self.generations)}

Recent Trend: {trend_str}
Last Update: {time.strftime('%H:%M:%S')}
"""
            self.stats_text.set_text(stats_info)
    
    def show_plots(self):
        """显示图表"""
        print("Starting real-time chart monitoring")
        print("Close window to stop monitoring")
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
        finally:
            self.monitoring = False
    
    def text_mode(self):
        """文本模式监控"""
        print("Starting text mode monitoring")
        print("Press Ctrl+C to stop monitoring\n")
        
        last_size = 0
        
        try:
            while True:
                if self.log_file.exists():
                    current_size = self.log_file.stat().st_size
                    if current_size != last_size:
                        data = self.load_data_safe()
                        if data:
                            self.display_text_info(data[-1], len(data))
                        last_size = current_size
                    else:
                        print(f"\rWaiting for new data... {time.strftime('%H:%M:%S')}", end="", flush=True)
                else:
                    print(f"\rWaiting for log file... {time.strftime('%H:%M:%S')}", end="", flush=True)
                
                time.sleep(3)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
    
    def display_text_info(self, data, total_count):
        """显示文本信息"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 70)
        print("                CUDA Training Real-time Monitor")
        print("=" * 70)
        
        print(f"Current Generation: {data.get('generation', 0)}")
        print(f"Best Fitness: {data.get('best_fitness', 0):.6f}")
        print(f"Mean Fitness: {data.get('avg_fitness', data.get('mean_fitness', 0)):.6f}")
        print(f"Std Fitness: {data.get('std_fitness', 0):.6f}")
        print(f"Generation Time: {data.get('generation_time', 0):.2f}s")
        
        if 'system_memory_gb' in data:
            print(f"System Memory: {data['system_memory_gb']:.2f}GB")
        
        print(f"\nTotal Records: {total_count}")
        print("=" * 70)
        print(f"Last Update: {time.strftime('%H:%M:%S')}")
        print("Press Ctrl+C to stop monitoring")
        print("=" * 70)

def main():
    """主函数"""
    print("Clean Real-time Training Monitor")
    print("=" * 50)
    
    try:
        monitor = CleanTrainingMonitor()
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nUser interrupted")
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Please ensure training has started and generated log files")

if __name__ == "__main__":
    main()