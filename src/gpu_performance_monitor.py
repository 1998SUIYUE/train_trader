"""
GPU性能监控模块
监控GPU使用情况和性能指标
"""

import time
import torch
import psutil
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: float
    gpu_memory_used: float
    system_memory_used: float
    cpu_usage: float
    operation_name: str
    duration: Optional[float] = None

class GPUPerformanceMonitor:
    """GPU性能监控器"""
    
    def __init__(self, gpu_manager, log_file: Optional[Path] = None):
        """
        初始化性能监控器
        
        Args:
            gpu_manager: GPU管理器
            log_file: 性能日志文件路径
        """
        self.gpu_manager = gpu_manager
        self.log_file = log_file
        self.metrics_history: List[PerformanceMetrics] = []
        self.operation_start_time = None
        self.current_operation = None
        
    def start_operation(self, operation_name: str):
        """开始监控操作"""
        self.current_operation = operation_name
        self.operation_start_time = time.time()
        
        # 记录开始时的性能指标
        metrics = self._collect_metrics(operation_name + "_start")
        self.metrics_history.append(metrics)
        
    def end_operation(self):
        """结束监控操作"""
        if self.operation_start_time is None:
            return
            
        duration = time.time() - self.operation_start_time
        
        # 记录结束时的性能指标
        metrics = self._collect_metrics(self.current_operation + "_end", duration)
        self.metrics_history.append(metrics)
        
        # 保存到日志文件
        if self.log_file:
            self._save_metrics(metrics)
            
        self.operation_start_time = None
        self.current_operation = None
        
        return duration
        
    def _collect_metrics(self, operation_name: str, duration: Optional[float] = None) -> PerformanceMetrics:
        """收集当前性能指标"""
        # GPU内存使用
        gpu_memory_used, _ = self.gpu_manager.get_memory_usage()
        
        # 系统内存使用
        memory = psutil.virtual_memory()
        system_memory_used = (memory.total - memory.available) / 1e9
        
        # CPU使用率
        cpu_usage = psutil.cpu_percent()
        
        return PerformanceMetrics(
            timestamp=time.time(),
            gpu_memory_used=gpu_memory_used,
            system_memory_used=system_memory_used,
            cpu_usage=cpu_usage,
            operation_name=operation_name,
            duration=duration
        )
    
    def _save_metrics(self, metrics: PerformanceMetrics):
        """保存指标到文件"""
        if not self.log_file:
            return
            
        try:
            # 确保目录存在
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为字典
            metrics_dict = {
                'timestamp': metrics.timestamp,
                'gpu_memory_used_gb': metrics.gpu_memory_used,
                'system_memory_used_gb': metrics.system_memory_used,
                'cpu_usage_percent': metrics.cpu_usage,
                'operation_name': metrics.operation_name,
                'duration_seconds': metrics.duration
            }
            
            # 追加到JSONL文件
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics_dict) + '\n')
                
        except Exception as e:
            print(f"保存性能指标失败: {e}")
    
    def get_operation_summary(self, operation_name: str) -> Dict:
        """获取特定操作的性能摘要"""
        operation_metrics = [
            m for m in self.metrics_history 
            if m.operation_name.startswith(operation_name) and m.duration is not None
        ]
        
        if not operation_metrics:
            return {}
            
        durations = [m.duration for m in operation_metrics]
        gpu_memory = [m.gpu_memory_used for m in operation_metrics]
        
        return {
            'operation': operation_name,
            'count': len(operation_metrics),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'avg_gpu_memory': sum(gpu_memory) / len(gpu_memory),
            'max_gpu_memory': max(gpu_memory)
        }
    
    def get_overall_summary(self) -> Dict:
        """获取整体性能摘要"""
        if not self.metrics_history:
            return {}
            
        # 按操作类型分组
        operations = set(m.operation_name.split('_')[0] for m in self.metrics_history)
        summaries = {}
        
        for op in operations:
            summaries[op] = self.get_operation_summary(op)
            
        return summaries
    
    def print_summary(self):
        """打印性能摘要"""
        summary = self.get_overall_summary()
        
        print("\n=== GPU性能监控摘要 ===")
        for operation, stats in summary.items():
            if stats:
                print(f"\n{operation}操作:")
                print(f"  执行次数: {stats['count']}")
                print(f"  平均耗时: {stats['avg_duration']:.3f}秒")
                print(f"  最短耗时: {stats['min_duration']:.3f}秒")
                print(f"  最长耗时: {stats['max_duration']:.3f}秒")
                print(f"  平均GPU内存: {stats['avg_gpu_memory']:.2f}GB")
                print(f"  峰值GPU内存: {stats['max_gpu_memory']:.2f}GB")
        print("========================\n")

class PerformanceContext:
    """性能监控上下文管理器"""
    
    def __init__(self, monitor: GPUPerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        
    def __enter__(self):
        self.monitor.start_operation(self.operation_name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = self.monitor.end_operation()
        return False

# 便捷函数
def monitor_gpu_operation(monitor: GPUPerformanceMonitor, operation_name: str):
    """装饰器：监控GPU操作性能"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceContext(monitor, operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # 测试性能监控器
    from gpu_utils import get_windows_gpu_manager
    
    gpu_manager = get_windows_gpu_manager()
    monitor = GPUPerformanceMonitor(gpu_manager, Path("performance_test.jsonl"))
    
    # 模拟一些GPU操作
    with PerformanceContext(monitor, "test_operation"):
        # 创建一些GPU张量
        x = torch.randn(1000, 1000, device=gpu_manager.device)
        y = torch.randn(1000, 1000, device=gpu_manager.device)
        z = torch.mm(x, y)
        time.sleep(0.1)  # 模拟计算时间
    
    # 打印摘要
    monitor.print_summary()