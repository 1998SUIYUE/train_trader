#!/usr/bin/env python3
"""
最终版实时训练监控器
解决所有字体和编码问题
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
    print("Graphics available - showing charts")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("matplotlib not available - text mode")

class FinalMonitor:
    def __init__(self):
        self.log_file = None
        self.data_queue = queue.Queue()
        self.max_points = 100
        
        self.generations = deque(maxlen=self.max_points)
        self.best_fitness = deque(maxlen=self.max_points)
        self.mean_fitness = deque(maxlen=self.max_points)
        self.generation_times = deque(maxlen=self.max_points)
        
        self.last_file_size = 0
        self.monitoring = True
        
        if not self.find_log_file():
            sys.exit(1)
    
    def find_log_file(self):
        paths = [
            Path("results/training_history_cuda.jsonl"),
            Path("results/training_history.jsonl"),
            Path("training_history_cuda.jsonl"), 
            Path("training_history.jsonl")
        ]
        
        for path in paths:
            if path.exists():
                self.log_file = path
                print(f"Found: {path}")
                return True
        
        print("No log file found")
        print("Start training first: python core/main_cuda.py")
        return False
    
    def load_data(self):
        data = []
        encodings = ['utf-8', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(self.log_file, 'r', encoding=encoding, errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"JSON parsing error in log file: {e} on line: {line.strip()}")
                                continue
                            except Exception as e:
                                print(f"Unexpected error parsing log file: {e} on line: {line.strip()}")
                                continue
                return data
            except:
                continue
        return []
    
    def start_monitoring(self):
        # Load initial data and populate deques
        initial_data = self.load_data()
        for item in initial_data:
            self.generations.append(item.get('generation', 0))
            self.best_fitness.append(item.get('best_fitness', 0))
            self.mean_fitness.append(item.get('avg_fitness', item.get('mean_fitness', 0)))
            self.generation_times.append(item.get('generation_time', 0))
        
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
        while self.monitoring:
            try:
                if self.log_file.exists():
                    current_size = self.log_file.stat().st_size
                    if current_size != self.last_file_size:
                        # Read only new lines
                        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            f.seek(self.last_file_size) # Seek to the last read position
                            new_lines = f.readlines()
                            for line in new_lines:
                                line = line.strip()
                                if line:
                                    try:
                                        self.data_queue.put(json.loads(line))
                                    except json.JSONDecodeError as e:
                                        print(f"JSON parsing error in log file: {e} on line: {line.strip()}")
                                        continue
                                    except Exception as e:
                                        print(f"Unexpected error parsing log file: {e} on line: {line.strip()}")
                                        continue
                        self.last_file_size = current_size
                time.sleep(1)
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(5)
    
    def setup_plots(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Training Monitor', fontsize=14)
        
        # Fitness plot
        self.ax1 = self.axes[0, 0]
        self.line_best, = self.ax1.plot([], [], 'b-', linewidth=2, label='Best')
        self.line_mean, = self.ax1.plot([], [], 'r--', linewidth=1, label='Mean')
        self.ax1.set_title('Fitness Evolution')
        self.ax1.set_xlabel('Generation')
        self.ax1.set_ylabel('Fitness')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Time plot
        self.ax2 = self.axes[0, 1]
        self.line_time, = self.ax2.plot([], [], 'g-', linewidth=2)
        self.ax2.set_title('Training Time')
        self.ax2.set_xlabel('Generation')
        self.ax2.set_ylabel('Seconds')
        self.ax2.grid(True, alpha=0.3)
        
        # Distribution
        self.ax3 = self.axes[1, 0]
        self.ax3.set_title('Fitness Distribution')
        
        # Stats
        self.ax4 = self.axes[1, 1]
        self.ax4.axis('off')
        self.stats_text = self.ax4.text(0.05, 0.95, '', transform=self.ax4.transAxes,
                                       fontsize=9, verticalalignment='top')
        
        plt.tight_layout()
        
        self.ani = animation.FuncAnimation(self.fig, self.update_plots,
                                         interval=1000, blit=False, 
                                         cache_frame_data=False)
    
    def update_plots(self, frame):
        # Process new data
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                self.generations.append(data.get('generation', 0))
                self.best_fitness.append(data.get('best_fitness', 0))
                self.mean_fitness.append(data.get('avg_fitness', data.get('mean_fitness', 0)))
                self.generation_times.append(data.get('generation_time', 0))
            except:
                break
        
        if len(self.generations) == 0:
            return
        
        gens = list(self.generations)
        best_fit = list(self.best_fitness)
        mean_fit = list(self.mean_fitness)
        times = list(self.generation_times)
        
        # Update fitness plot
        self.line_best.set_data(gens, best_fit)
        self.line_mean.set_data(gens, mean_fit)
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # Update time plot
        self.line_time.set_data(gens, times)
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Update distribution
        if len(best_fit) >= 10:
            self.ax3.clear()
            self.ax3.hist(best_fit[-20:], bins=10, alpha=0.7, color='lightblue')
            self.ax3.set_title('Recent Fitness')
            self.ax3.grid(True, alpha=0.3)
        
        # Update stats
        if len(self.generations) > 0:
            current_gen = self.generations[-1]
            current_best = self.best_fitness[-1]
            current_mean = self.mean_fitness[-1]
            current_time = self.generation_times[-1] if self.generation_times else 0
            
            total_time = sum(self.generation_times)
            avg_time = total_time / len(self.generation_times) if self.generation_times else 0
            best_ever = max(self.best_fitness)
            
            stats_info = f"""Training Stats

Generation: {current_gen}
Best: {current_best:.6f}
Mean: {current_mean:.6f}
Time: {current_time:.2f}s

Best Ever: {best_ever:.6f}
Total Time: {total_time/3600:.2f}h
Avg Time: {avg_time:.2f}s
Points: {len(self.generations)}

Update: {time.strftime('%H:%M:%S')}
"""
            self.stats_text.set_text(stats_info)
    
    def show_plots(self):
        print("Starting chart monitor")
        print("Close window to stop")
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nStopped")
        finally:
            self.monitoring = False
    
    def text_mode(self):
        print("Text mode monitor")
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
                        print(f"\rWaiting... {time.strftime('%H:%M:%S')}", end="", flush=True)
                else:
                    print(f"\rWaiting for file... {time.strftime('%H:%M:%S')}", end="", flush=True)
                
                time.sleep(3)
                
        except KeyboardInterrupt:
            print("\nStopped")
    
    def display_text(self, data, total_count):
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 60)
        print("           Training Monitor")
        print("=" * 60)
        
        print(f"Generation: {data.get('generation', 0)}")
        print(f"Best: {data.get('best_fitness', 0):.6f}")
        print(f"Mean: {data.get('avg_fitness', data.get('mean_fitness', 0)):.6f}")
        print(f"Time: {data.get('generation_time', 0):.2f}s")
        
        if 'system_memory_gb' in data:
            print(f"Memory: {data['system_memory_gb']:.2f}GB")
        
        print(f"\nRecords: {total_count}")
        print("=" * 60)
        print(f"Update: {time.strftime('%H:%M:%S')}")
        print("Ctrl+C to stop")
        print("=" * 60)

def main():
    print("Final Training Monitor")
    print("=" * 40)
    
    try:
        monitor = FinalMonitor()
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()