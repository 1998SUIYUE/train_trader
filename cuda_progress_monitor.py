#!/usr/bin/env python3
"""
CUDA训练进度监控器
提供多种方式查看和监控CUDA遗传算法训练进度
"""

import json
import time
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

class CUDAProgressMonitor:
    def __init__(self, log_file="results/training_history.jsonl"):
        self.log_file = Path(log_file)
        
    def load_data(self):
        """加载训练数据"""
        data = []
        if not self.log_file.exists():
            return data
            
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        except Exception as e:
            print(f"加载数据失败: {e}")
        return data
    
    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_current_status(self, data):
        """显示当前训练状态"""
        if not data:
            print("📊 等待训练数据...")
            return
        
        latest = data[-1]
        
        print("=" * 80)
        print("🚀 CUDA遗传算法训练进度监控")
        print("=" * 80)
        
        # 基本训练信息
        print(f"📈 当前代数: {latest.get('generation', 0)}")
        print(f"🏆 最佳适应度: {latest.get('best_fitness', 0):.8f}")
        print(f"📊 平均适应度: {latest.get('avg_fitness', latest.get('mean_fitness', 0)):.8f}")
        print(f"📉 标准差: {latest.get('std_fitness', 0):.8f}")
        print(f"⏱️  本代用时: {latest.get('generation_time', 0):.2f}秒")
        
        # 改进信息
        if 'no_improvement_count' in latest:
            print(f"🔄 无改进代数: {latest['no_improvement_count']}")
        elif 'no_improvement' in latest:
            print(f"🔄 无改进代数: {latest['no_improvement']}")
        
        # 交易性能指标
        if 'mean_sharpe_ratio' in latest:
            print(f"📈 夏普比率: {latest['mean_sharpe_ratio']:.6f}")
        if 'mean_sortino_ratio' in latest:
            print(f"📊 索提诺比率: {latest['mean_sortino_ratio']:.6f}")
        if 'mean_max_drawdown' in latest:
            print(f"📉 最大回撤: {latest['mean_max_drawdown']:.6f}")
        if 'mean_overall_return' in latest:
            print(f"💰 总回报: {latest['mean_overall_return']:.6f}")
        
        # 系统资源信息
        if 'gpu_memory_allocated' in latest:
            print(f"🖥️  GPU内存(已分配): {latest['gpu_memory_allocated']:.3f}GB")
        if 'gpu_memory_reserved' in latest:
            print(f"🖥️  GPU内存(已保留): {latest['gpu_memory_reserved']:.3f}GB")
        if 'system_memory_gb' in latest:
            print(f"💾 系统内存: {latest['system_memory_gb']:.2f}GB")
        
        # 训练统计
        if len(data) > 1:
            total_time = latest.get('total_time', 0)
            if total_time == 0:
                total_time = sum(d.get('generation_time', 0) for d in data)
            
            avg_time = total_time / len(data) if len(data) > 0 else 0
            best_ever = max(d.get('best_fitness', 0) for d in data)
            
            print(f"\n📊 训练统计:")
            print(f"   总代数: {len(data)}")
            print(f"   历史最佳: {best_ever:.8f}")
            print(f"   总训练时间: {total_time/3600:.2f}小时")
            print(f"   平均每代: {avg_time:.2f}秒")
            
            # 改进趋势分析
            recent_10 = data[-10:] if len(data) >= 10 else data
            recent_best = [d.get('best_fitness', 0) for d in recent_10]
            if len(recent_best) > 1:
                trend = "📈 上升" if recent_best[-1] > recent_best[0] else "📉 下降"
                improvement = recent_best[-1] - recent_best[0]
                print(f"   最近趋势: {trend} (变化: {improvement:.8f})")
        
        # 时间戳
        if 'timestamp' in latest:
            print(f"\n⏰ 最后更新: {latest['timestamp']}")
        
        print("=" * 80)
    
    def display_fitness_chart(self, data, max_points=60):
        """显示适应度趋势ASCII图表"""
        if len(data) < 2:
            return
        
        print(f"\n📈 适应度趋势图 (最近{min(max_points, len(data))}代):")
        
        # 获取最近的数据点
        recent_data = data[-max_points:] if len(data) > max_points else data
        fitness_values = [d.get('best_fitness', 0) for d in recent_data]
        
        if not fitness_values:
            return
        
        # 归一化到0-20的范围用于ASCII图表
        min_val = min(fitness_values)
        max_val = max(fitness_values)
        
        if max_val == min_val:
            normalized = [10] * len(fitness_values)
        else:
            normalized = [int((val - min_val) / (max_val - min_val) * 20) for val in fitness_values]
        
        # 绘制ASCII图表
        for row in range(20, -1, -1):
            if max_val == min_val:
                line = f"{max_val:12.6f} |"
            else:
                val = max_val - (max_val - min_val) * (20 - row) / 20
                line = f"{val:12.6f} |"
            
            for norm_val in normalized:
                if norm_val >= row:
                    line += "█"
                else:
                    line += " "
            print(line)
        
        # 底部标尺
        print(" " * 14 + "+" + "-" * len(normalized))
        print(f"             最小值: {min_val:.6f}, 最大值: {max_val:.6f}")
    
    def display_recent_history(self, data, count=10):
        """显示最近的训练历史"""
        if not data:
            return
        
        recent = data[-count:] if len(data) > count else data
        
        print(f"\n📋 最近{len(recent)}代训练历史:")
        print("-" * 80)
        print("代数    最佳适应度        平均适应度        用时(秒)   改进")
        print("-" * 80)
        
        for d in recent:
            gen = d.get('generation', 0)
            best = d.get('best_fitness', 0)
            avg = d.get('avg_fitness', d.get('mean_fitness', 0))
            time_taken = d.get('generation_time', 0)
            improved = d.get('improved', False)
            
            indicator = "🔥" if improved else "  "
            print(f"{gen:4d}   {best:12.8f}   {avg:12.8f}   {time_taken:7.2f}   {indicator}")
    
    def show_summary(self):
        """显示训练摘要"""
        data = self.load_data()
        self.display_current_status(data)
        self.display_fitness_chart(data)
        self.display_recent_history(data)
    
    def watch_training(self, refresh_interval=3.0, show_chart=True):
        """实时监控训练进度"""
        print(f"🔍 开始监控CUDA训练: {self.log_file}")
        print(f"🔄 刷新间隔: {refresh_interval}秒")
        print("按 Ctrl+C 停止监控\n")
        
        last_size = 0
        
        try:
            while True:
                if self.log_file.exists():
                    current_size = self.log_file.stat().st_size
                    if current_size != last_size:
                        # 文件有更新
                        data = self.load_data()
                        self.clear_screen()
                        self.display_current_status(data)
                        
                        if show_chart:
                            self.display_fitness_chart(data)
                        
                        last_size = current_size
                        print(f"\n⏰ 监控时间: {datetime.now().strftime('%H:%M:%S')}")
                        print("按 Ctrl+C 停止监控")
                    else:
                        # 文件没有更新
                        print(f"\r⏳ 等待新数据... {datetime.now().strftime('%H:%M:%S')}", 
                              end="", flush=True)
                else:
                    print(f"\r📁 等待日志文件创建... {datetime.now().strftime('%H:%M:%S')}", 
                          end="", flush=True)
                
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 监控已停止")

def main():
    parser = argparse.ArgumentParser(description='CUDA训练进度监控器')
    parser.add_argument('--log-file', default='results/training_history.jsonl',
                       help='训练日志文件路径')
    parser.add_argument('--watch', '-w', action='store_true',
                       help='实时监控模式')
    parser.add_argument('--interval', '-i', type=float, default=3.0,
                       help='刷新间隔(秒)')
    parser.add_argument('--no-chart', action='store_true',
                       help='不显示图表')
    parser.add_argument('--tail', type=int,
                       help='只显示最后N条记录')
    
    args = parser.parse_args()
    
    monitor = CUDAProgressMonitor(args.log_file)
    
    if args.watch:
        # 实时监控模式
        monitor.watch_training(args.interval, not args.no_chart)
    else:
        # 一次性显示模式
        data = monitor.load_data()
        
        if not data:
            print(f"❌ 未找到训练数据: {args.log_file}")
            print("请确保训练已经开始并生成了日志文件")
            return
        
        if args.tail:
            data = data[-args.tail:]
        
        monitor.display_current_status(data)
        
        if not args.no_chart:
            monitor.display_fitness_chart(data)
        
        monitor.display_recent_history(data)

if __name__ == "__main__":
    main()