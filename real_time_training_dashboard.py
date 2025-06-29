#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时训练进度动态图表监控器
提供实时更新的动态图表来监控CUDA训练进度
"""

import json
import time
import os
import sys
from pathlib import Path
import argparse
from collections import deque
import threading
import queue

# 尝试导入绘图库
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.dates import DateFormatter
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("警告: 需要安装matplotlib和pandas来使用动态图表功能")
    print("请运行: pip install matplotlib pandas")

class RealTimeTrainingDashboard:
    """实时训练进度动态图表监控器"""
    
    def __init__(self, log_file, max_points=100, update_interval=2000):
        """
        初始化实时监控器
        
        Args:
            log_file: 训练日志文件路径
            max_points: 图表中显示的最大数据点数
            update_interval: 更新间隔(毫秒)
        """
        self.log_file = Path(log_file)
        self.max_points = max_points
        self.update_interval = update_interval
        
        # 数据存储
        self.data_queue = queue.Queue()
        self.generations = deque(maxlen=max_points)
        self.best_fitness = deque(maxlen=max_points)
        self.mean_fitness = deque(maxlen=max_points)
        self.generation_times = deque(maxlen=max_points)
        self.sharpe_ratios = deque(maxlen=max_points)
        self.memory_usage = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)
        
        # 文件监控
        self.last_file_size = 0
        self.last_data_count = 0
        
        # 创建图表
        self.setup_plots()
        
        # 启动数据读取线程
        self.data_thread = threading.Thread(target=self.monitor_log_file, daemon=True)
        self.data_thread.start()
    
    def setup_plots(self):
        """设置图表布局"""
        if not PLOTTING_AVAILABLE:
            return
            
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建子图
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('🚀 CUDA训练实时监控面板', fontsize=16, fontweight='bold')
        
        # 适应度曲线 (左上)
        self.ax_fitness = self.axes[0, 0]
        self.line_best, = self.ax_fitness.plot([], [], 'b-', linewidth=2, label='最佳适应度')
        self.line_mean, = self.ax_fitness.plot([], [], 'r--', linewidth=1.5, label='平均适应度')
        self.ax_fitness.set_title('📈 适应度进化曲线')
        self.ax_fitness.set_xlabel('代数')
        self.ax_fitness.set_ylabel('适应度')
        self.ax_fitness.legend()
        self.ax_fitness.grid(True, alpha=0.3)
        
        # 训练时间 (中上)
        self.ax_time = self.axes[0, 1]
        self.line_time, = self.ax_time.plot([], [], 'g-', linewidth=2)
        self.ax_time.set_title('⏱️ 每代训练时间')
        self.ax_time.set_xlabel('代数')
        self.ax_time.set_ylabel('时间(秒)')
        self.ax_time.grid(True, alpha=0.3)
        
        # 夏普比率 (右上)
        self.ax_sharpe = self.axes[0, 2]
        self.line_sharpe, = self.ax_sharpe.plot([], [], 'purple', linewidth=2)
        self.ax_sharpe.set_title('📊 夏普比率趋势')
        self.ax_sharpe.set_xlabel('代数')
        self.ax_sharpe.set_ylabel('夏普比率')
        self.ax_sharpe.grid(True, alpha=0.3)
        
        # 内存使用 (左下)
        self.ax_memory = self.axes[1, 0]
        self.line_memory, = self.ax_memory.plot([], [], 'orange', linewidth=2)
        self.ax_memory.set_title('💾 系统内存使用')
        self.ax_memory.set_xlabel('代数')
        self.ax_memory.set_ylabel('内存(GB)')
        self.ax_memory.grid(True, alpha=0.3)
        
        # 适应度分布直方图 (中下)
        self.ax_hist = self.axes[1, 1]
        self.ax_hist.set_title('📊 最近适应度分布')
        self.ax_hist.set_xlabel('适应度')
        self.ax_hist.set_ylabel('频次')
        
        # 训练统计信息 (右下)
        self.ax_stats = self.axes[1, 2]
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.1, 0.9, '', transform=self.ax_stats.transAxes, 
                                           fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # 创建动画
        self.ani = animation.FuncAnimation(self.fig, self.update_plots, 
                                         interval=self.update_interval, blit=False)
    
    def monitor_log_file(self):
        """监控日志文件变化的线程函数"""
        while True:
            try:
                if self.log_file.exists():
                    current_size = self.log_file.stat().st_size
                    if current_size != self.last_file_size:
                        # 文件有更新，读取新数据
                        new_data = self.load_new_data()
                        if new_data:
                            for data_point in new_data:
                                self.data_queue.put(data_point)
                        self.last_file_size = current_size
                
                time.sleep(1)  # 每秒检查一次文件
                
            except Exception as e:
                print(f"监控文件时出错: {e}")
                time.sleep(5)
    
    def load_new_data(self):
        """加载新的训练数据"""
        try:
            all_data = []
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
            # 返回新增的数据
            new_data = all_data[self.last_data_count:]
            self.last_data_count = len(all_data)
            return new_data
            
        except Exception as e:
            print(f"读取数据时出错: {e}")
            return []
    
    def update_data(self):
        """更新数据缓存"""
        # 处理队列中的新数据
        while not self.data_queue.empty():
            try:
                data_point = self.data_queue.get_nowait()
                
                # 添加时间戳
                timestamp = datetime.now()
                self.timestamps.append(timestamp)
                
                # 提取数据
                self.generations.append(data_point.get('generation', 0))
                self.best_fitness.append(data_point.get('best_fitness', 0))
                self.mean_fitness.append(data_point.get('mean_fitness', 0))
                self.generation_times.append(data_point.get('generation_time', 0))
                self.sharpe_ratios.append(data_point.get('mean_sharpe_ratio', 0))
                self.memory_usage.append(data_point.get('system_memory_gb', 0))
                
            except queue.Empty:
                break
    
    def update_plots(self, frame):
        """更新图表的回调函数"""
        if not PLOTTING_AVAILABLE:
            return
            
        # 更新数据
        self.update_data()
        
        if len(self.generations) == 0:
            return
        
        # 转换为numpy数组以便绘图
        gens = np.array(self.generations)
        best_fit = np.array(self.best_fitness)
        mean_fit = np.array(self.mean_fitness)
        times = np.array(self.generation_times)
        sharpe = np.array(self.sharpe_ratios)
        memory = np.array(self.memory_usage)
        
        # 更新适应度曲线
        self.line_best.set_data(gens, best_fit)
        self.line_mean.set_data(gens, mean_fit)
        self.ax_fitness.relim()
        self.ax_fitness.autoscale_view()
        
        # 更新训练时间
        self.line_time.set_data(gens, times)
        self.ax_time.relim()
        self.ax_time.autoscale_view()
        
        # 更新夏普比率
        if len(sharpe) > 0 and np.any(sharpe != 0):
            self.line_sharpe.set_data(gens, sharpe)
            self.ax_sharpe.relim()
            self.ax_sharpe.autoscale_view()
        
        # 更新内存使用
        if len(memory) > 0 and np.any(memory != 0):
            self.line_memory.set_data(gens, memory)
            self.ax_memory.relim()
            self.ax_memory.autoscale_view()
        
        # 更新适应度分布直方图
        if len(best_fit) >= 10:
            self.ax_hist.clear()
            self.ax_hist.hist(best_fit[-20:], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            self.ax_hist.set_title('📊 最近20代适应度分布')
            self.ax_hist.set_xlabel('适应度')
            self.ax_hist.set_ylabel('频次')
            self.ax_hist.grid(True, alpha=0.3)
        
        # 更新统计信息
        if len(self.generations) > 0:
            current_gen = self.generations[-1]
            current_best = self.best_fitness[-1]
            current_mean = self.mean_fitness[-1]
            current_time = self.generation_times[-1] if self.generation_times else 0
            
            # 计算统计信息
            total_time = sum(self.generation_times) if self.generation_times else 0
            avg_time = total_time / len(self.generation_times) if self.generation_times else 0
            best_ever = max(self.best_fitness) if self.best_fitness else 0
            
            # 计算改进趋势
            if len(self.best_fitness) >= 10:
                recent_trend = np.polyfit(range(10), list(self.best_fitness)[-10:], 1)[0]
                trend_str = "📈 上升" if recent_trend > 0 else "📉 下降" if recent_trend < 0 else "➡️ 平稳"
            else:
                trend_str = "📊 收集中"
            
            stats_info = f"""
📊 训练统计信息

🔢 当前代数: {current_gen}
🏆 当前最佳: {current_best:.6f}
📈 当前平均: {current_mean:.6f}
⏱️  本代用时: {current_time:.2f}秒

🎯 历史最佳: {best_ever:.6f}
⏰ 总训练时间: {total_time/3600:.2f}小时
📊 平均每代: {avg_time:.2f}秒
📈 最近趋势: {trend_str}

🕐 最后更新: {datetime.now().strftime('%H:%M:%S')}
"""
            
            self.stats_text.set_text(stats_info)
        
        return (self.line_best, self.line_mean, self.line_time, 
                self.line_sharpe, self.line_memory, self.stats_text)
    
    def show(self):
        """显示动态图表"""
        if not PLOTTING_AVAILABLE:
            print("错误: 无法显示图表，请安装matplotlib和pandas")
            return
            
        print(f"🚀 启动实时训练监控面板")
        print(f"📁 监控文件: {self.log_file}")
        print(f"🔄 更新间隔: {self.update_interval/1000:.1f}秒")
        print(f"📊 最大数据点: {self.max_points}")
        print("按 Ctrl+C 或关闭窗口停止监控\n")
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n👋 监控已停止")

def create_simple_dashboard(log_file):
    """创建简单的文本监控面板（无图形界面）"""
    print("🚀 启动简单文本监控面板")
    print(f"📁 监控文件: {log_file}")
    print("按 Ctrl+C 停止监控\n")
    
    last_size = 0
    
    try:
        while True:
            if Path(log_file).exists():
                current_size = Path(log_file).stat().st_size
                if current_size != last_size:
                    # 读取最新数据
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        if lines:
                            latest_line = lines[-1].strip()
                            if latest_line:
                                data = json.loads(latest_line)
                                
                                # 清屏并显示信息
                                os.system('cls' if os.name == 'nt' else 'clear')
                                
                                print("=" * 80)
                                print("🚀 CUDA训练实时监控 (文本模式)")
                                print("=" * 80)
                                print(f"📈 当前代数: {data.get('generation', 0)}")
                                print(f"🏆 最佳适应度: {data.get('best_fitness', 0):.6f}")
                                print(f"📊 平均适应度: {data.get('mean_fitness', 0):.6f}")
                                print(f"📉 标准差: {data.get('std_fitness', 0):.6f}")
                                print(f"⏱️  本代用时: {data.get('generation_time', 0):.2f}秒")
                                
                                if 'mean_sharpe_ratio' in data:
                                    print(f"📈 夏普比率: {data['mean_sharpe_ratio']:.6f}")
                                if 'mean_sortino_ratio' in data:
                                    print(f"📊 索提诺比率: {data['mean_sortino_ratio']:.6f}")
                                if 'system_memory_gb' in data:
                                    print(f"💾 系统内存: {data['system_memory_gb']:.2f}GB")
                                
                                print("=" * 80)
                                print(f"🕐 最后更新: {time.strftime('%H:%M:%S')}")
                                print("按 Ctrl+C 停止监控")
                    
                    except Exception as e:
                        print(f"读取数据时出错: {e}")
                    
                    last_size = current_size
                else:
                    print(f"\r⏳ 等待新数据... {time.strftime('%H:%M:%S')}", end="", flush=True)
            else:
                print(f"\r📁 等待日志文件创建... {time.strftime('%H:%M:%S')}", end="", flush=True)
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")

def main():
    parser = argparse.ArgumentParser(description='实时训练进度动态图表监控器')
    parser.add_argument('log_file', nargs='?', 
                       default='results/training_history.jsonl',
                       help='训练日志文件路径 (默认: results/training_history.jsonl)')
    parser.add_argument('--max-points', type=int, default=100,
                       help='图表中显示的最大数据点数 (默认: 100)')
    parser.add_argument('--interval', type=int, default=2000,
                       help='更新间隔(毫秒) (默认: 2000)')
    parser.add_argument('--text-mode', action='store_true',
                       help='使用文本模式（无图形界面）')
    parser.add_argument('--auto', action='store_true',
                       help='自动查找训练日志文件')
    
    args = parser.parse_args()
    
    # 自动查找日志文件
    if args.auto:
        possible_paths = [
            Path("results/training_history.jsonl"),
            Path("results/training_history_cuda.jsonl"),
            Path("training_history.jsonl"),
            Path("../results/training_history.jsonl"),
            Path("../results/training_history_cuda.jsonl")
        ]
        
        log_file = None
        for path in possible_paths:
            if path.exists():
                log_file = path
                break
        
        if log_file:
            args.log_file = str(log_file)
            print(f"🔍 自动发现日志文件: {args.log_file}")
        else:
            print("❌ 未找到训练日志文件")
            print("请确保训练已经开始并生成了日志文件")
            return
    
    # 检查文件是否存在
    if not Path(args.log_file).exists():
        print(f"❌ 日志文件不存在: {args.log_file}")
        print("请确保训练已经开始并生成了日志文件")
        print("或者使用 --auto 参数自动查找日志文件")
        return
    
    # 选择监控模式
    if args.text_mode or not PLOTTING_AVAILABLE:
        if not PLOTTING_AVAILABLE:
            print("⚠️  图形库不可用，使用文本模式")
        create_simple_dashboard(args.log_file)
    else:
        # 创建动态图表监控器
        dashboard = RealTimeTrainingDashboard(
            log_file=args.log_file,
            max_points=args.max_points,
            update_interval=args.interval
        )
        dashboard.show()

if __name__ == "__main__":
    main()