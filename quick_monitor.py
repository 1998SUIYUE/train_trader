#!/usr/bin/env python3
"""
快速增强版训练监控器
Quick Enhanced Training Monitor - 简化版实时监控
"""

import json
import time
import os
import sys
from pathlib import Path
from collections import deque

class QuickEnhancedMonitor:
    def __init__(self):
        self.log_file = None
        self.last_file_size = 0
        
        if not self.find_log_file():
            sys.exit(1)
    
    def find_log_file(self):
        """查找日志文件"""
        paths = [
            # 增强版日志文件
            Path("results/enhanced_training_history.jsonl"),
            Path("../results/enhanced_training_history.jsonl"),
            Path("enhanced_training_history.jsonl"),
            
            # 备份文件
            Path("results/enhanced_training_history.jsonl.backup"),
            Path("../results/enhanced_training_history.jsonl.backup"),
            
            # 普通版本日志文件
            Path("results/training_history.jsonl"),
            Path("../results/training_history.jsonl"),
            Path("training_history.jsonl"),
        ]
        
        for path in paths:
            if path.exists():
                self.log_file = path
                print(f"🎯 Found log file: {path}")
                return True
        
        print("❌ Training log file not found")
        print("Please start training first:")
        print("  - Enhanced: python core/main_enhanced_cuda.py")
        print("  - Regular: python core/main_cuda.py")
        return False
    
    def load_latest_data(self):
        """加载最新数据"""
        data = []
        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"⚠️  File reading error: {e}")
        
        return data
    
    def display_status(self, data):
        """Display training status"""
        if not data:
            print("📝 No training data available")
            return
        
        latest = data[-1]
        total_records = len(data)
        
        # 清屏
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 80)
        print("           🚀 Enhanced CUDA Genetic Algorithm Training Monitor (Quick)")
        print("=" * 80)
        
        # Basic information
        print(f"📈 Current Generation: {latest.get('generation', 0)}")
        print(f"🎯 Best Fitness: {latest.get('best_fitness', 0):.6f}")
        print(f"📊 Average Fitness: {latest.get('avg_fitness', 0):.6f}")
        print(f"⏱️  Generation Time: {latest.get('generation_time', 0):.2f}s")
        print(f"🔄 No Improvement Count: {latest.get('no_improvement_count', 0)}")
        
        # Enhanced version specific information
        if 'data_ratio' in latest:
            print(f"\n🔥 Data Annealing:")
            print(f"   📊 Data Usage Ratio: {latest.get('data_ratio', 1.0):.3f}")
            print(f"   🎯 Complexity Score: {latest.get('complexity_score', 1.0):.3f}")
            print(f"   📈 Annealing Strategy: {latest.get('annealing_strategy', 'none')}")
        
        if 'pareto_front_size' in latest:
            print(f"\n🎯 Multi-Objective Optimization:")
            print(f"   📊 Pareto Front: {latest.get('pareto_front_size', 0)}")
            print(f"   📈 Hypervolume: {latest.get('hypervolume', 0.0):.4f}")
        
        if 'avg_sharpe_ratio' in latest:
            print(f"\n💰 Trading Performance:")
            print(f"   📈 Sharpe Ratio: {latest.get('avg_sharpe_ratio', 0.0):.3f}")
            print(f"   📉 Max Drawdown: {latest.get('avg_max_drawdown', 0.0):.3f}")
            print(f"   💵 Total Return: {latest.get('avg_total_return', 0.0):.3f}")
            print(f"   🎯 Win Rate: {latest.get('avg_win_rate', 0.0):.3f}")
        
        if 'population_diversity' in latest:
            print(f"\n🌈 Algorithm Status:")
            print(f"   🔀 Population Diversity: {latest.get('population_diversity', 0.0):.3f}")
        
        # System performance
        print(f"\n💻 System Performance:")
        if 'gpu_memory_allocated' in latest:
            print(f"   🎮 GPU Memory: {latest['gpu_memory_allocated']:.2f}GB")
        if 'system_memory_gb' in latest:
            print(f"   💾 System Memory: {latest['system_memory_gb']:.2f}GB")
        
        # Statistics
        print(f"\n📊 Training Statistics:")
        print(f"   📝 Total Records: {total_records}")
        
        if total_records >= 2:
            # Calculate training speed
            first_time = data[0].get('generation_time', 0)
            recent_times = [d.get('generation_time', 0) for d in data[-10:]]
            avg_time = sum(recent_times) / len(recent_times) if recent_times else 0
            
            # Calculate improvement
            first_fitness = data[0].get('best_fitness', 0)
            current_fitness = latest.get('best_fitness', 0)
            improvement = current_fitness - first_fitness
            
            print(f"   ⚡ Avg Generation Time: {avg_time:.2f}s")
            print(f"   📈 Fitness Improvement: {improvement:.6f}")
            
            # Recent trend
            if total_records >= 5:
                recent_fitness = [d.get('best_fitness', 0) for d in data[-5:]]
                trend = "📈 Rising" if recent_fitness[-1] > recent_fitness[0] else "📉 Falling"
                print(f"   📊 Recent Trend: {trend}")
        
        print("=" * 80)
        print(f"🕒 Update Time: {time.strftime('%H:%M:%S')}")
        print("Press Ctrl+C to stop monitoring")
        print("=" * 80)
    
    def monitor(self):
        """Start monitoring"""
        print("🚀 Starting quick monitoring mode")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                if self.log_file.exists():
                    current_size = self.log_file.stat().st_size
                    if current_size != self.last_file_size:
                        data = self.load_latest_data()
                        self.display_status(data)
                        self.last_file_size = current_size
                    else:
                        # File not updated, show waiting status
                        print(f"\r⏳ Waiting for updates... {time.strftime('%H:%M:%S')}", end="", flush=True)
                else:
                    print(f"\r⏳ Waiting for log file... {time.strftime('%H:%M:%S')}", end="", flush=True)
                
                time.sleep(2)  # Check every 2 seconds
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped")

def main():
    print("🚀 Quick Enhanced Training Monitor")
    print("=" * 40)
    
    try:
        monitor = QuickEnhancedMonitor()
        monitor.monitor()
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()