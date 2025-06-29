#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时训练进度动态图表
直接运行即可显示实时更新的训练进度图表
"""

import json
import time
import os
import sys
from pathlib import Path
import threading
import queue
from collections import deque
from datetime import datetime

# 尝试导入绘图库
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    from matplotlib.dates import DateFormatter
    import matplotlib.dates as mdates
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

class LiveTrainingChart:
    """实时训练图表显示器"""
    
    def __init__(self):
        self.log_file = None
        self.data_queue = queue.Queue()
        self.max_points = 100
        
        # 数据存储
        self.generations = deque(maxlen=self.max_points)
        self.best_fitness = deque(maxlen=self.max_points)
        self.mean_fitness = deque(maxlen=self.max_points)
        self.generation_times = deque(maxlen=self.max_points)
        self.sharpe_ratios = deque(maxlen=self.max_points)
        self.memory_usage = deque(maxlen=self.max_points)
        self.timestamps = deque(maxlen=self.max_points)
        
        # 文件监控
        self.last_file_size = 0
        self.last_data_count = 0
        self.monitoring = True
        
        # 查找日志文件
        self.find_log_file()
        
        if PLOTTING_AVAILABLE:
            self.setup_plots()
            self.start_monitoring()
        else:
            self.text_mode()
    
    def find_log_file(self):
        """自动查找训练日志文件"""
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
                print(f"🔍 找到日志文件: {path}")
                return
        
        print("❌ 未找到训练日志文件")
        print("请确保训练已经开始并生成了日志文件")
        print("\n可能的日志文件位置:")
        for path in possible_paths:
            print(f"  - {path}")
        sys.exit(1)
    
    def setup_plots(self):
        """设置图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表窗口
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('🚀 CUDA训练实时监控', fontsize=16, fontweight='bold')
        
        # 适应度曲线 (左上)
        self.ax1 = self.axes[0, 0]
        self.line_best, = self.ax1.plot([], [], 'b-', linewidth=2, label='最佳适应度')
        self.line_mean, = self.ax1.plot([], [], 'r--', linewidth=1.5, label='平均适应度')
        self.ax1.set_title('📈 适应度进化')
        self.ax1.set_xlabel('代数')
        self.ax1.set_ylabel('适应度')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # 训练时间 (右上)
        self.ax2 = self.axes[0, 1]
        self.line_time, = self.ax2.plot([], [], 'g-', linewidth=2)
        self.ax2.set_title('⏱️ 训练时间')
        self.ax2.set_xlabel('代数')
        self.ax2.set_ylabel('时间(秒)')
        self.ax2.grid(True, alpha=0.3)
        
        # 夏普比率 (左下)
        self.ax3 = self.axes[1, 0]
        self.line_sharpe, = self.ax3.plot([], [], 'purple', linewidth=2)
        self.ax3.set_title('📊 夏普比率')
        self.ax3.set_xlabel('代数')
        self.ax3.set_ylabel('夏普比率')
        self.ax3.grid(True, alpha=0.3)
        
        # 统计信息 (右下)
        self.ax4 = self.axes[1, 1]
        self.ax4.axis('off')
        self.stats_text = self.ax4.text(0.1, 0.9, '', transform=self.ax4.transAxes, 
                                       fontsize=11, verticalalignment='top', 
                                       fontfamily='monospace')
        
        plt.tight_layout()
        
        # 创建动画
        self.ani = animation.FuncAnimation(self.fig, self.update_plots, 
                                         interval=2000, blit=False)
    
    def start_monitoring(self):
        """启动文件监控线程"""
        self.monitor_thread = threading.Thread(target=self.monitor_file, daemon=True)
        self.monitor_thread.start()
    
    def monitor_file(self):
        """监控日志文件变化"""
        while self.monitoring:
            try:
                if self.log_file.exists():
                    current_size = self.log_file.stat().st_size
                    if current_size != self.last_file_size:
                        new_data = self.load_new_data()
                        if new_data:
                            for data_point in new_data:
                                self.data_queue.put(data_point)
                        self.last_file_size = current_size
                
                time.sleep(1)
                
            except Exception as e:
                print(f"监控文件时出错: {e}")
                time.sleep(5)
    
    def load_new_data(self):
        """加载新的训练数据"""
        try:
            all_data = []
            # 尝试不同的编码方式
            encodings = ['utf-8', 'utf-8-sig', 'gbk', 'latin1']
            
            for encoding in encodings:
                try:
                    with open(self.log_file, 'r', encoding=encoding) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    all_data.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue
                    break  # 成功读取，跳出循环
                except UnicodeDecodeError:
                    continue  # 尝试下一个编码
            
            new_data = all_data[self.last_data_count:]
            self.last_data_count = len(all_data)
            return new_data
            
        except Exception as e:
            print(f"读取数据时出错: {e}")
            return []
    
    def update_data(self):
        """更新数据缓存"""
        while not self.data_queue.empty():
            try:
                data_point = self.data_queue.get_nowait()
                
                self.timestamps.append(datetime.now())
                self.generations.append(data_point.get('generation', 0))
                self.best_fitness.append(data_point.get('best_fitness', 0))
                self.mean_fitness.append(data_point.get('mean_fitness', 0))
                self.generation_times.append(data_point.get('generation_time', 0))
                self.sharpe_ratios.append(data_point.get('mean_sharpe_ratio', 0))
                self.memory_usage.append(data_point.get('system_memory_gb', 0))
                
            except queue.Empty:
                break
    
    def update_plots(self, frame):
        """更新图表"""
        self.update_data()
        
        if len(self.generations) == 0:
            return
        
        gens = np.array(self.generations)
        best_fit = np.array(self.best_fitness)
        mean_fit = np.array(self.mean_fitness)
        times = np.array(self.generation_times)
        sharpe = np.array(self.sharpe_ratios)
        
        # 更新适应度曲线
        self.line_best.set_data(gens, best_fit)
        self.line_mean.set_data(gens, mean_fit)
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # 更新训练时间
        self.line_time.set_data(gens, times)
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # 更新夏普比率
        if len(sharpe) > 0 and np.any(sharpe != 0):
            self.line_sharpe.set_data(gens, sharpe)
            self.ax3.relim()
            self.ax3.autoscale_view()
        
        # 更新统计信息
        if len(self.generations) > 0:
            current_gen = self.generations[-1]
            current_best = self.best_fitness[-1]
            current_mean = self.mean_fitness[-1]
            current_time = self.generation_times[-1] if self.generation_times else 0
            
            total_time = sum(self.generation_times) if self.generation_times else 0
            avg_time = total_time / len(self.generation_times) if self.generation_times else 0
            best_ever = max(self.best_fitness) if self.best_fitness else 0
            
            # 计算趋势
            if len(self.best_fitness) >= 10:
                recent_trend = np.polyfit(range(10), list(self.best_fitness)[-10:], 1)[0]
                trend_str = "📈 上升" if recent_trend > 0 else "📉 下降" if recent_trend < 0 else "➡️ 平稳"
            else:
                trend_str = "📊 收集中"
            
            stats_info = f"""
📊 实时训练统计

🔢 当前代数: {current_gen}
🏆 当前最佳: {current_best:.6f}
📈 当前平均: {current_mean:.6f}
⏱️  本代用时: {current_time:.2f}秒

🎯 历史最佳: {best_ever:.6f}
⏰ 总训练时间: {total_time/3600:.2f}小时
📊 平均每代: {avg_time:.2f}秒
📈 最近趋势: {trend_str}

📁 数据点数: {len(self.generations)}
🕐 最后更新: {datetime.now().strftime('%H:%M:%S')}
"""
            
            self.stats_text.set_text(stats_info)
        
        return (self.line_best, self.line_mean, self.line_time, 
                self.line_sharpe, self.stats_text)
    
    def text_mode(self):
        """文本模式监控（当matplotlib不可用时）"""
        print("⚠️  图形库不可用，使用文本模式")
        print("安装图形库: pip install matplotlib numpy")
        print()
        
        last_size = 0
        
        try:
            while True:
                if self.log_file.exists():
                    current_size = self.log_file.stat().st_size
                    if current_size != last_size:
                        data = self.load_latest_data()
                        if data:
                            self.display_text_info(data)
                        last_size = current_size
                    else:
                        print(f"\r⏳ 等待新数据... {time.strftime('%H:%M:%S')}", end="", flush=True)
                else:
                    print(f"\r📁 等待日志文件... {time.strftime('%H:%M:%S')}", end="", flush=True)
                
                time.sleep(3)
                
        except KeyboardInterrupt:
            print("\n👋 监控已停止")
    
    def load_latest_data(self):
        """加载最新数据（文本模式用）"""
        try:
            # 尝试不同的编码方式
            encodings = ['utf-8', 'utf-8-sig', 'gbk', 'latin1']
            
            for encoding in encodings:
                try:
                    with open(self.log_file, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    
                    if lines:
                        latest_line = lines[-1].strip()
                        if latest_line:
                            return json.loads(latest_line)
                    return None
                except UnicodeDecodeError:
                    continue
            
            return None
        except Exception as e:
            print(f"读取数据出错: {e}")
            return None
    
    def display_text_info(self, data):
        """显示文本信息"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🚀" + "=" * 70 + "🚀")
        print("                    CUDA训练实时监控")
        print("🚀" + "=" * 70 + "🚀")
        
        print(f"📊 当前代数: {data.get('generation', 0)}")
        print(f"🏆 最佳适应度: {data.get('best_fitness', 0):.6f}")
        print(f"📈 平均适应度: {data.get('mean_fitness', 0):.6f}")
        print(f"📉 标准差: {data.get('std_fitness', 0):.6f}")
        print(f"⏱️  本代用时: {data.get('generation_time', 0):.2f}秒")
        
        if 'mean_sharpe_ratio' in data:
            print(f"📈 夏普比率: {data['mean_sharpe_ratio']:.6f}")
        if 'mean_sortino_ratio' in data:
            print(f"📊 索提诺比率: {data['mean_sortino_ratio']:.6f}")
        if 'system_memory_gb' in data:
            print(f"💾 系统内存: {data['system_memory_gb']:.2f}GB")
        
        print("=" * 72)
        print(f"🕐 最后更新: {time.strftime('%H:%M:%S')}")
        print("💡 按 Ctrl+C 停止监控")
        print("=" * 72)
    
    def show(self):
        """显示图表"""
        if PLOTTING_AVAILABLE:
            print(f"🚀 启动实时训练监控图表")
            print(f"📁 监控文件: {self.log_file}")
            print("💡 关闭窗口或按 Ctrl+C 停止监控\n")
            
            try:
                plt.show()
            except KeyboardInterrupt:
                print("\n👋 监控已停止")
            finally:
                self.monitoring = False

def main():
    """主函数"""
    print("🚀 实时训练进度动态图表")
    print("=" * 50)
    
    try:
        chart = LiveTrainingChart()
        chart.show()
    except KeyboardInterrupt:
        print("\n👋 用户中断")
    except Exception as e:
        print(f"\n❌ 出现错误: {e}")
        print("请确保:")
        print("1. 训练已经开始并生成了日志文件")
        print("2. 已安装必要的依赖: pip install matplotlib numpy")

if __name__ == "__main__":
    main()