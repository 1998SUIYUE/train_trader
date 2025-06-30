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
                print(f"🎯 找到日志文件: {path}")
                return True
        
        print("❌ 未找到训练日志文件")
        print("请先启动训练:")
        print("  - 增强版: python core/main_enhanced_cuda.py")
        print("  - 普通版: python core/main_cuda.py")
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
            print(f"⚠️  读取文件错误: {e}")
        
        return data
    
    def display_status(self, data):
        """显示训练状态"""
        if not data:
            print("📝 暂无训练数据")
            return
        
        latest = data[-1]
        total_records = len(data)
        
        # 清屏
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 80)
        print("           🚀 增强版CUDA遗传算法训练监控 (快速版)")
        print("=" * 80)
        
        # 基础信息
        print(f"📈 当前代数: {latest.get('generation', 0)}")
        print(f"🎯 最佳适应度: {latest.get('best_fitness', 0):.6f}")
        print(f"📊 平均适应度: {latest.get('avg_fitness', 0):.6f}")
        print(f"⏱️  代数时间: {latest.get('generation_time', 0):.2f}s")
        print(f"🔄 无改进次数: {latest.get('no_improvement_count', 0)}")
        
        # 增强版特有信息
        if 'data_ratio' in latest:
            print(f"\n🔥 数据退火:")
            print(f"   📊 数据使用比例: {latest.get('data_ratio', 1.0):.3f}")
            print(f"   🎯 复杂度得分: {latest.get('complexity_score', 1.0):.3f}")
            print(f"   📈 退火策略: {latest.get('annealing_strategy', 'none')}")
        
        if 'pareto_front_size' in latest:
            print(f"\n🎯 多目标优化:")
            print(f"   📊 帕累托前沿: {latest.get('pareto_front_size', 0)}")
            print(f"   📈 超体积: {latest.get('hypervolume', 0.0):.4f}")
        
        if 'avg_sharpe_ratio' in latest:
            print(f"\n💰 交易性能:")
            print(f"   📈 夏普比率: {latest.get('avg_sharpe_ratio', 0.0):.3f}")
            print(f"   📉 最大回撤: {latest.get('avg_max_drawdown', 0.0):.3f}")
            print(f"   💵 总收益率: {latest.get('avg_total_return', 0.0):.3f}")
            print(f"   🎯 胜率: {latest.get('avg_win_rate', 0.0):.3f}")
        
        if 'population_diversity' in latest:
            print(f"\n🌈 算法状态:")
            print(f"   🔀 种群多样性: {latest.get('population_diversity', 0.0):.3f}")
        
        # 系统性能
        print(f"\n💻 系统性能:")
        if 'gpu_memory_allocated' in latest:
            print(f"   🎮 GPU内存: {latest['gpu_memory_allocated']:.2f}GB")
        if 'system_memory_gb' in latest:
            print(f"   💾 系统内存: {latest['system_memory_gb']:.2f}GB")
        
        # 统计信息
        print(f"\n📊 训练统计:")
        print(f"   📝 总记录数: {total_records}")
        
        if total_records >= 2:
            # 计算训练速度
            first_time = data[0].get('generation_time', 0)
            recent_times = [d.get('generation_time', 0) for d in data[-10:]]
            avg_time = sum(recent_times) / len(recent_times) if recent_times else 0
            
            # 计算改进情况
            first_fitness = data[0].get('best_fitness', 0)
            current_fitness = latest.get('best_fitness', 0)
            improvement = current_fitness - first_fitness
            
            print(f"   ⚡ 平均代数时间: {avg_time:.2f}s")
            print(f"   📈 适应度改进: {improvement:.6f}")
            
            # 最近趋势
            if total_records >= 5:
                recent_fitness = [d.get('best_fitness', 0) for d in data[-5:]]
                trend = "📈 上升" if recent_fitness[-1] > recent_fitness[0] else "📉 下降"
                print(f"   📊 最近趋势: {trend}")
        
        print("=" * 80)
        print(f"🕒 更新时间: {time.strftime('%H:%M:%S')}")
        print("按 Ctrl+C 停止监控")
        print("=" * 80)
    
    def monitor(self):
        """开始监控"""
        print("🚀 启动快速监控模式")
        print("按 Ctrl+C 停止\n")
        
        try:
            while True:
                if self.log_file.exists():
                    current_size = self.log_file.stat().st_size
                    if current_size != self.last_file_size:
                        data = self.load_latest_data()
                        self.display_status(data)
                        self.last_file_size = current_size
                    else:
                        # 文件没有更新，显示等待状态
                        print(f"\r⏳ 等待更新... {time.strftime('%H:%M:%S')}", end="", flush=True)
                else:
                    print(f"\r⏳ 等待日志文件... {time.strftime('%H:%M:%S')}", end="", flush=True)
                
                time.sleep(2)  # 每2秒检查一次
                
        except KeyboardInterrupt:
            print("\n\n⏹️  监控已停止")

def main():
    print("🚀 快速增强版训练监控器")
    print("=" * 40)
    
    try:
        monitor = QuickEnhancedMonitor()
        monitor.monitor()
    except KeyboardInterrupt:
        print("\n⏹️  监控被中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")

if __name__ == "__main__":
    main()