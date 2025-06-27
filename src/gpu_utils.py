"""
Windows版GPU工具函数模块
使用DirectML支持AMD GPU加速
"""

import torch
import torch_directml
import numpy as np
from typing import Optional, Tuple, Union, List
import gc
import psutil

class WindowsGPUManager:
    """Windows GPU资源管理器 (DirectML)"""
    
    def __init__(self, device_id: Optional[int] = None):
        """
        初始化GPU管理器
        
        Args:
            device_id: GPU设备ID (DirectML中通常为0)
        """
        self.device = self._initialize_device()
        self.memory_pool = {}
        self._log_gpu_info()
        print("--- WindowsGPUManager初始化 ---")
        print(f"设备: {self.device}")
        print("-----------------------------")
    
    def _initialize_device(self) -> torch.device:
        """初始化DirectML设备"""
        try:
            # 尝试使用DirectML
            device = torch_directml.device()
            print(f"DirectML设备初始化成功: {device}")
            return device
        except Exception as e:
            print(f"DirectML不可用: {e}")
            print("使用CPU设备")
            return torch.device('cpu')
    
    def _log_gpu_info(self):
        """记录GPU信息"""
        if self.device.type == 'privateuseone':  # DirectML设备类型
            print("使用DirectML GPU加速")
            
            # 获取系统GPU信息
            try:
                import subprocess
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                      capture_output=True, text=True)
                gpu_names = [line.strip() for line in result.stdout.split('\n') 
                           if 'AMD' in line or 'Radeon' in line]
                if gpu_names:
                    print(f"检测到GPU: {gpu_names[0]}")
            except:
                print("GPU信息获取失败")
        else:
            print("使用CPU进行计算")
    
    def to_gpu(self, data: Union[np.ndarray, torch.Tensor], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        将数据转移到GPU
        
        Args:
            data: 输入数据
            dtype: 数据类型
            
        Returns:
            GPU上的张量
        """
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).to(dtype)
        elif isinstance(data, torch.Tensor):
            tensor = data.to(dtype)
        else:
            tensor = torch.tensor(data, dtype=dtype)
        
        print(f"数据已转移到GPU: {tensor.shape}, {tensor.dtype}")
        return tensor.to(self.device)
    
    def to_cpu(self, tensor: torch.Tensor) -> np.ndarray:
        """
        将GPU张量转移到CPU并转换为NumPy数组
        
        Args:
            tensor: GPU张量
            
        Returns:
            NumPy数组
        """
        print(f"数据已转移到CPU: {tensor.shape}, {tensor.dtype}")
        return tensor.detach().cpu().numpy()
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """
        获取内存使用情况 (Windows系统内存)
        
        Returns:
            (已使用内存GB, 总内存GB)
        """
        memory = psutil.virtual_memory()
        used_gb = (memory.total - memory.available) / 1e9
        total_gb = memory.total / 1e9
        return used_gb, total_gb
    
    def clear_cache(self):
        """清理内存缓存"""
        gc.collect()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()  # DirectML也支持此调用
    
    def create_memory_pool(self, name: str, size: Tuple[int, ...], dtype: torch.dtype = torch.float32):
        """
        创建内存池
        
        Args:
            name: 内存池名称
            size: 张量大小
            dtype: 数据类型
        """
        self.memory_pool[name] = torch.zeros(size, dtype=dtype, device=self.device)
        print(f"创建内存池 '{name}': {size}, {dtype}")
    
    def get_from_pool(self, name: str) -> torch.Tensor:
        """从内存池获取张量"""
        return self.memory_pool.get(name)


class WindowsGPUBatchProcessor:
    """Windows GPU批处理器"""
    
    def __init__(self, gpu_manager: WindowsGPUManager, batch_size: int = 500):
        """
        初始化批处理器
        
        Args:
            gpu_manager: GPU管理器
            batch_size: 批处理大小 (Windows上建议较小)
        """
        self.gpu_manager = gpu_manager
        self.batch_size = batch_size
        self.device = gpu_manager.device
    
    def process_in_batches(self, data: np.ndarray, process_func, **kwargs) -> np.ndarray:
        """
        分批处理大数据
        
        Args:
            data: 输入数据
            process_func: 处理函数
            **kwargs: 处理函数的额外参数
            
        Returns:
            处理结果
        """
        n_samples = len(data)
        results = []
        
        for i in range(0, n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, n_samples)
            batch = data[i:end_idx]
            
            # 转移到GPU
            batch_gpu = self.gpu_manager.to_gpu(batch)
            
            # 处理
            result_gpu = process_func(batch_gpu, **kwargs)
            
            # 转回CPU
            result_cpu = self.gpu_manager.to_cpu(result_gpu)
            results.append(result_cpu)
            
            # 清理中间结果
            del batch_gpu, result_gpu
            self.gpu_manager.clear_cache()
        
        return np.concatenate(results, axis=0)


def check_windows_gpu_compatibility() -> dict:
    """
    检查Windows GPU兼容性
    
    Returns:
        GPU信息字典
    """
    info = {
        'pytorch_version': torch.__version__,
        'directml_available': False,
        'system_memory_gb': psutil.virtual_memory().total / 1e9,
        'cpu_cores': psutil.cpu_count(),
    }
    
    try:
        import torch_directml
        device = torch_directml.device()
        info['directml_available'] = True
        info['directml_device'] = str(device)
        
        # 获取GPU信息
        try:
            import subprocess
            result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name,AdapterRAM'], 
                                  capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'AMD' in line or 'Radeon' in line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            vram_bytes = int(parts[-1])
                            info['gpu_memory_gb'] = vram_bytes / 1e9
                        except:
                            pass
                    info['gpu_name'] = ' '.join(parts[:-1]) if len(parts) > 1 else line.strip()
                    break
        except:
            info['gpu_name'] = 'Unknown AMD GPU'
            
    except ImportError:
        info['directml_error'] = 'torch_directml not installed'
    except Exception as e:
        info['directml_error'] = str(e)
    
    return info


    def optimize_windows_gpu_settings():
        """优化Windows GPU设置"""
        try:
            # 设置环境变量优化DirectML性能
            import os
            os.environ['PYTORCH_DIRECTML_DEBUG'] = '0'  # 关闭调试模式
            
            # 清理内存
            gc.collect()
            
            print("Windows GPU设置已优化")
        except Exception as e:
            print(f"GPU设置优化失败: {e}")


def estimate_windows_memory_usage(population_size: int, feature_dim: int, n_samples: int) -> dict:
    """
    估算Windows系统内存使用量
    
    Args:
        population_size: 种群大小
        feature_dim: 特征维度
        n_samples: 样本数量
        
    Returns:
        内存使用估算
    """
    # 每个float32占4字节
    bytes_per_float = 4
    
    # 主要数据结构的内存使用
    population_memory = population_size * (feature_dim + 5) * bytes_per_float
    features_memory = n_samples * feature_dim * bytes_per_float
    scores_memory = population_size * n_samples * bytes_per_float
    
    # Windows上的额外开销
    windows_overhead = (population_memory + features_memory + scores_memory) * 0.3
    
    total_memory = population_memory + features_memory + scores_memory + windows_overhead
    
    # 获取当前系统内存使用
    memory = psutil.virtual_memory()
    
    return {
        'population_mb': population_memory / 1e6,
        'features_mb': features_memory / 1e6,
        'scores_mb': scores_memory / 1e6,
        'windows_overhead_mb': windows_overhead / 1e6,
        'total_mb': total_memory / 1e6,
        'total_gb': total_memory / 1e9,
        'system_total_gb': memory.total / 1e9,
        'system_available_gb': memory.available / 1e9,
        'estimated_usage_percent': (total_memory / memory.total) * 100
    }


# 全局GPU管理器实例
_global_gpu_manager = None

def get_windows_gpu_manager() -> WindowsGPUManager:
    """获取全局Windows GPU管理器实例"""
    global _global_gpu_manager
    if _global_gpu_manager is None:
        _global_gpu_manager = WindowsGPUManager()
    return _global_gpu_manager


if __name__ == "__main__":
    # 测试Windows GPU功能
    print("=== Windows GPU兼容性检查 ===")
    gpu_info = check_windows_gpu_compatibility()
    for key, value in gpu_info.items():
        print(f"{key}: {value}")
    
    print("\n=== 内存使用估算 ===")
    memory_usage = estimate_windows_memory_usage(
        population_size=500,  # Windows上建议较小的种群
        feature_dim=1400,
        n_samples=50000
    )
    for key, value in memory_usage.items():
        if 'gb' in key:
            print(f"{key}: {value:.2f} GB")
        elif 'mb' in key:
            print(f"{key}: {value:.1f} MB")
        elif 'percent' in key:
            print(f"{key}: {value:.1f}%")
    
    print("\n=== Windows GPU管理器测试 ===")
    gpu_manager = get_windows_gpu_manager()
    
    # 测试数据转移
    test_data = np.random.randn(100, 50).astype(np.float32)
    gpu_tensor = gpu_manager.to_gpu(test_data)
    cpu_result = gpu_manager.to_cpu(gpu_tensor)
    
    print(f"数据转移测试: {'通过' if np.allclose(test_data, cpu_result) else '失败'}")
    
    # 显示内存使用
    used, total = gpu_manager.get_memory_usage()
    print(f"系统内存使用: {used:.2f} GB / {total:.2f} GB")