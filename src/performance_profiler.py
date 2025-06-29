#!/usr/bin/env python3
"""
性能分析器
用于监控遗传算法各部分的运行时间和性能瓶颈
"""

import time
import torch
import psutil
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    name: str
    total_time: float = 0.0
    call_count: int = 0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_time: float = 0.0
    
    def update(self, elapsed_time: float):
        """更新性能指标"""
        self.total_time += elapsed_time
        self.call_count += 1
        self.avg_time = self.total_time / self.call_count
        self.min_time = min(self.min_time, elapsed_time)
        self.max_time = max(self.max_time, elapsed_time)
        self.last_time = elapsed_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'total_time': self.total_time,
            'call_count': self.call_count,
            'avg_time': self.avg_time,
            'min_time': self.min_time if self.min_time != float('inf') else 0.0,
            'max_time': self.max_time,
            'last_time': self.last_time
        }


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        """
        初始化性能分析器
        
        Args:
            enable_gpu_monitoring: 是否启用GPU监控
        """
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        self.start_time = time.time()
        
        # GPU内存监控
        self.gpu_memory_history = deque(maxlen=1000)
        self.system_memory_history = deque(maxlen=1000)
        
        # 监控线程
        self._monitoring = False
        self._monitor_thread = None
        
        # 嵌套计时器栈
        self._timer_stack = []
        
        print(f"🔍 性能分析器已启动")
        print(f"   GPU监控: {'启用' if self.enable_gpu_monitoring else '禁用'}")
    
    @contextmanager
    def timer(self, name: str, category: str = "general"):
        """
        计时器上下文管理器
        
        Args:
            name: 计时器名称
            category: 分类
        """
        full_name = f"{category}.{name}" if category != "general" else name
        
        # GPU同步（如果需要）
        if self.enable_gpu_monitoring:
            torch.cuda.synchronize()
        
        start_time = time.time()
        self._timer_stack.append((full_name, start_time))
        
        try:
            yield
        finally:
            # GPU同步（如果需要）
            if self.enable_gpu_monitoring:
                torch.cuda.synchronize()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 移除栈顶
            self._timer_stack.pop()
            
            # 更新指标
            if full_name not in self.metrics:
                self.metrics[full_name] = PerformanceMetrics(full_name)
            
            self.metrics[full_name].update(elapsed_time)
            
            # 记录内存使用
            self._record_memory_usage()
    
    def start_monitoring(self, interval: float = 1.0):
        """
        开始后台监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        print(f"📊 后台监控已启动 (间隔: {interval}s)")
    
    def stop_monitoring(self):
        """停止后台监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        print("📊 后台监控已停止")
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self._monitoring:
            self._record_memory_usage()
            time.sleep(interval)
    
    def _record_memory_usage(self):
        """记录内存使用情况"""
        timestamp = time.time() - self.start_time
        
        # 系统内存
        system_memory = psutil.virtual_memory()
        self.system_memory_history.append({
            'timestamp': timestamp,
            'used_gb': system_memory.used / 1e9,
            'percent': system_memory.percent
        })
        
        # GPU内存
        if self.enable_gpu_monitoring:
            gpu_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_reserved = torch.cuda.memory_reserved() / 1e9
            self.gpu_memory_history.append({
                'timestamp': timestamp,
                'allocated_gb': gpu_allocated,
                'reserved_gb': gpu_reserved
            })
    
    def get_summary(self, sort_by: str = "total_time") -> Dict[str, Any]:
        """
        获取性能总结
        
        Args:
            sort_by: 排序方式 ('total_time', 'avg_time', 'call_count')
        
        Returns:
            性能总结字典
        """
        # 按类别分组
        categories = defaultdict(list)
        for name, metrics in self.metrics.items():
            if '.' in name:
                category, method = name.split('.', 1)
            else:
                category, method = 'general', name
            categories[category].append(metrics)
        
        # 排序
        for category in categories:
            categories[category].sort(key=lambda x: getattr(x, sort_by), reverse=True)
        
        # 计算总时间
        total_runtime = time.time() - self.start_time
        
        # 内存统计
        memory_stats = self._get_memory_stats()
        
        return {
            'total_runtime': total_runtime,
            'categories': {cat: [m.to_dict() for m in metrics] for cat, metrics in categories.items()},
            'memory_stats': memory_stats,
            'top_bottlenecks': self._get_top_bottlenecks(5)
        }
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计"""
        stats = {
            'system_memory': {
                'current_gb': psutil.virtual_memory().used / 1e9,
                'peak_gb': max([m['used_gb'] for m in self.system_memory_history], default=0),
                'history_length': len(self.system_memory_history)
            }
        }
        
        if self.enable_gpu_monitoring and self.gpu_memory_history:
            stats['gpu_memory'] = {
                'current_allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'current_reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'peak_allocated_gb': max([m['allocated_gb'] for m in self.gpu_memory_history], default=0),
                'peak_reserved_gb': max([m['reserved_gb'] for m in self.gpu_memory_history], default=0),
                'history_length': len(self.gpu_memory_history)
            }
        
        return stats
    
    def _get_top_bottlenecks(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """获取性能瓶颈"""
        sorted_metrics = sorted(
            self.metrics.values(), 
            key=lambda x: x.total_time, 
            reverse=True
        )
        
        return [m.to_dict() for m in sorted_metrics[:top_n]]
    
    def print_summary(self, detailed: bool = True):
        """
        打印性能总结
        
        Args:
            detailed: 是否显示详细信息
        """
        summary = self.get_summary()
        total_time = summary['total_runtime']
        
        print("\n" + "="*80)
        print("🔍 性能分析报告")
        print("="*80)
        print(f"总运行时间: {total_time:.2f}秒")
        
        # 内存统计
        memory_stats = summary['memory_stats']
        print(f"\n📊 内存使用:")
        print(f"   系统内存: {memory_stats['system_memory']['current_gb']:.2f}GB (峰值: {memory_stats['system_memory']['peak_gb']:.2f}GB)")
        
        if 'gpu_memory' in memory_stats:
            gpu_mem = memory_stats['gpu_memory']
            print(f"   GPU内存:  {gpu_mem['current_allocated_gb']:.2f}GB / {gpu_mem['current_reserved_gb']:.2f}GB (峰值: {gpu_mem['peak_allocated_gb']:.2f}GB / {gpu_mem['peak_reserved_gb']:.2f}GB)")
        
        # 性能瓶颈
        print(f"\n🚨 性能瓶颈 (Top 5):")
        print(f"{'名称':<40} {'总时间':<10} {'调用次数':<8} {'平均时间':<10} {'占比':<8}")
        print("-" * 80)
        
        for bottleneck in summary['top_bottlenecks']:
            percentage = (bottleneck['total_time'] / total_time) * 100
            print(f"{bottleneck['name']:<40} {bottleneck['total_time']:<10.3f} {bottleneck['call_count']:<8} {bottleneck['avg_time']:<10.3f} {percentage:<8.1f}%")
        
        if detailed:
            # 按类别详细显示
            print(f"\n📋 详细分析:")
            for category, metrics in summary['categories'].items():
                print(f"\n🔸 {category.upper()}:")
                print(f"{'方法':<30} {'总时间':<10} {'调用次数':<8} {'平均时间':<10} {'最小':<8} {'最大':<8}")
                print("-" * 80)
                
                for metric in metrics:
                    print(f"{metric['name'].split('.')[-1]:<30} {metric['total_time']:<10.3f} {metric['call_count']:<8} {metric['avg_time']:<10.3f} {metric['min_time']:<8.3f} {metric['max_time']:<8.3f}")
        
        print("="*80)
    
    def save_report(self, filepath: Path):
        """
        保存性能报告到文件
        
        Args:
            filepath: 文件路径
        """
        summary = self.get_summary()
        
        # 添加时间戳
        summary['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        summary['memory_history'] = {
            'system': list(self.system_memory_history),
            'gpu': list(self.gpu_memory_history) if self.enable_gpu_monitoring else []
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📄 性能报告已保存: {filepath}")
    
    def reset(self):
        """重置所有统计数据"""
        self.metrics.clear()
        self.gpu_memory_history.clear()
        self.system_memory_history.clear()
        self.start_time = time.time()
        print("🔄 性能分析器已重置")


# 全局性能分析器实例
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """获取全局性能分析器"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile(name: str, category: str = "general"):
    """
    性能分析装饰器
    
    Args:
        name: 方法名称
        category: 分类
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with get_profiler().timer(name, category):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# 便捷函数
def timer(name: str, category: str = "general"):
    """获取计时器上下文管理器"""
    return get_profiler().timer(name, category)


def start_monitoring(interval: float = 1.0):
    """开始性能监控"""
    get_profiler().start_monitoring(interval)


def stop_monitoring():
    """停止性能监控"""
    get_profiler().stop_monitoring()


def print_summary(detailed: bool = True):
    """打印性能总结"""
    get_profiler().print_summary(detailed)


def save_report(filepath: Path):
    """保存性能报告"""
    get_profiler().save_report(filepath)


def reset_profiler():
    """重置性能分析器"""
    get_profiler().reset()


if __name__ == "__main__":
    # 测试性能分析器
    print("=== 性能分析器测试 ===")
    
    profiler = PerformanceProfiler()
    profiler.start_monitoring(0.5)
    
    # 模拟一些操作
    with profiler.timer("test_operation", "test"):
        time.sleep(0.1)
        
        with profiler.timer("sub_operation", "test"):
            time.sleep(0.05)
    
    with profiler.timer("another_operation", "test"):
        time.sleep(0.2)
    
    # 模拟GPU操作（如果可用）
    if torch.cuda.is_available():
        with profiler.timer("gpu_operation", "gpu"):
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.matmul(x, x.T)
            torch.cuda.synchronize()
    
    time.sleep(1)  # 让监控收集一些数据
    
    profiler.stop_monitoring()
    profiler.print_summary()
    
    # 保存报告
    report_path = Path("performance_test_report.json")
    profiler.save_report(report_path)
    
    print("性能分析器测试完成！")