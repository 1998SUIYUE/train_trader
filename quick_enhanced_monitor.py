#!/usr/bin/env python3
"""
快速增强版训练监控器
Quick Enhanced Training Monitor - 轻量级版本
"""

import json
import time
import os
from pathlib import Path

class QuickEnhancedMonitor:
    def __init__(self):
        self.log_file = self.find_log_file()
        if not self.log_file:
            print("❌ 未找到增强版训练日志文件")
            exit(1)
    
    def find_log_file(self):
        """查找日志文件"""
        paths = [
            Path("results/enhanced_training_history.jsonl"),
            Path("../results/enhanced_training_history.jsonl"),
            Path("enhanced_training_history.jsonl"),
            Path("results/enhanced_training_history.jsonl.backup"),
            Path("../results/enhanced_training_history.jsonl.backup"),
        ]
        
        for path in paths:
            if path.exists():
                print(f"🎯 找到日志文件: {path}")
                return path
        return None
    
    def get_latest_data(self):
        """获取最新数据"""
        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if not lines:
                    return None
                
                # 获取最后一行有效数据
                for line in reversed(lines):
                    line = line.strip()
                    if line:
                        try:
                            return json.loads(line)
                        except:
                            continue
                return None
        except Exception as e:
            print(f"❌ 读取文件错误: {e}")
            return None
    
    def display_status(self, data):
        """显示状态"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🚀 增强版CUDA遗传算法 - 快速状态监控")
        print("=" * 60)
        
        if not data:
            print("❌ 无数据")
            return
        
        # 基础信息
        gen = data.get('generation', 0)
        best_fit = data.get('best_fitness', 0)
        avg_fit = data.get('avg_fitness', 0)
        gen_time = data.get('generation_time', 0)
        
        print(f"📈 代数: {gen:>8}")
        print(f"🎯 最佳适应度: {best_fit:>12.6f}")
        print(f"📊 平均适应度: {avg_fit:>12.6f}")
        print(f"⏱️  代数时间: {gen_time:>10.2f}s")
        
        # 增强功能状态
        print("\n🔥 增强功能状态:")
        data_ratio = data.get('data_ratio', 1.0)
        complexity = data.get('complexity_score', 1.0)
        pareto_size = data.get('pareto_front_size', 0)
        diversity = data.get('population_diversity', 0.0)
        
        print(f"   数据使用比例: {data_ratio:>8.3f}")
        print(f"   复杂度得分: {complexity:>10.3f}")
        print(f"   帕累托前沿: {pareto_size:>10}")
        print(f"   种群多样性: {diversity:>10.3f}")
        
        # 交易性能
        print("\n💰 交易性能:")
        sharpe = data.get('avg_sharpe_ratio', 0.0)
        drawdown = data.get('avg_max_drawdown', 0.0)
        returns = data.get('avg_total_return', 0.0)
        winrate = data.get('avg_win_rate', 0.0)
        
        print(f"   夏普比率: {sharpe:>12.3f}")
        print(f"   最大回撤: {drawdown:>12.3f}")
        print(f"   总收益率: {returns:>12.3f}")
        print(f"   胜率: {winrate:>16.3f}")
        
        # 系统资源
        print("\n💻 系统资源:")
        gpu_mem = data.get('gpu_memory_allocated', 0.0)
        sys_mem = data.get('system_memory_gb', 0.0)
        
        print(f"   GPU内存: {gpu_mem:>13.2f}GB")
        print(f"   系统内存: {sys_mem:>11.2f}GB")
        
        print("\n" + "=" * 60)
        print(f"🕒 更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("按 Ctrl+C 停止监控")
    
    def run(self):
        """运行监控"""
        print("🚀 启动快速监控...")
        print("按 Ctrl+C 停止\n")
        
        try:
            while True:
                data = self.get_latest_data()
                self.display_status(data)
                time.sleep(5)  # 每5秒更新一次
                
        except KeyboardInterrupt:
            print("\n⏹️  监控已停止")

def main():
    monitor = QuickEnhancedMonitor()
    monitor.run()

if __name__ == "__main__":
    main()