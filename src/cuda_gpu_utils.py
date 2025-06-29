"""
CUDA版GPU工具函数模块
支持NVIDIA GPU CUDA加速
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union, List
import gc
import psutil

class CudaGPUManager:
    """CUDA GPU资源管理器"""
    
    def __init__(self, device_id: Optional[int] = None):
        """
        初始化GPU管理器
        
        Args:
            device_id: GPU设备ID (默认使用第一个可用GPU)
        """
        self.device = self._initialize_device(device_id)
        self.memory_pool = {}
        self._log_gpu_info()
        
        # 启用GPU优化
        self.optimize_tensor_operations()
        
        print("--- CudaGPUManager初始化 ---")
        print(f"设备: {self.device}")
        print("-----------------------------")
    
    def _initialize_device(self, device_id: Optional[int] = None) -> torch.device:
        """初始化CUDA设备"""
        try:
            if not torch.cuda.is_available():
                print("CUDA不可用，使用CPU设备")
                return torch.device('cpu')
            
            if device_id is None:
                device_id = 0  # 使用第一个GPU
            
            if device_id >= torch.cuda.device_count():
                print(f"GPU设备ID {device_id} 不存在，使用GPU 0")
                device_id = 0
            
            device = torch.device(f'cuda:{device_id}')
            
            # 测试GPU可用性
            test_tensor = torch.tensor([1.0]).to(device)
            del test_tensor
            torch.cuda.empty_cache()
            
            print(f"CUDA设备初始化成功: {device}")
            return device
            
        except Exception as e:
            print(f"CUDA初始化失败: {e}")
            print("使用CPU设备")
            return torch.device('cpu')
    
    def _log_gpu_info(self):
        """记录GPU信息"""
        if self.device.type == 'cuda':
            gpu_id = self.device.index
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
            
            print(f"使用CUDA GPU加速")
            print(f"GPU名称: {gpu_name}")
            print(f"GPU内存: {gpu_memory:.2f} GB")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"cuDNN版本: {torch.backends.cudnn.version()}")
            
            # 显示当前GPU内存使用情况
            allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
            cached = torch.cuda.memory_reserved(gpu_id) / 1e9
            print(f"GPU内存使用: {allocated:.2f} GB (已分配) / {cached:.2f} GB (已缓存)")
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
        
        return tensor.to(self.device, non_blocking=True)
    
    def to_cpu(self, tensor: torch.Tensor) -> np.ndarray:
        """
        将GPU张量转移到CPU并转换为NumPy数组
        
        Args:
            tensor: GPU张量
            
        Returns:
            NumPy数组
        """
        if tensor.device.type == 'cpu':
            return tensor.detach().numpy()
        return tensor.detach().cpu().numpy()
    
    def get_memory_usage(self) -> Tuple[float, float, float, float]:
        """
        获取GPU和系统内存使用情况
        
        Returns:
            (GPU已分配内存GB, GPU总内存GB, 系统已使用内存GB, 系统总内存GB)
        """
        if self.device.type == 'cuda':
            gpu_id = self.device.index
            allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
            total_gpu = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
        else:
            allocated = 0.0
            total_gpu = 0.0
        
        # 系统内存
        memory = psutil.virtual_memory()
        used_sys = (memory.total - memory.available) / 1e9
        total_sys = memory.total / 1e9
        
        return allocated, total_gpu, used_sys, total_sys
    
    def clear_cache(self):
        """清理GPU和系统内存缓存"""
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def optimize_tensor_operations(self):
        """优化GPU张量操作设置"""
        try:
            if self.device.type == 'cuda':
                # 启用cuDNN优化
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True  # 自动寻找最优算法
                torch.backends.cudnn.deterministic = False  # 允许非确定性算法以提高性能
                
                # 启用TensorFloat-32 (TF32) 用于更快的训练（Ampere架构及以上）
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                print("CUDA优化已启用")
                print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
                print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
            
        except Exception as e:
            print(f"GPU优化设置失败: {e}")
    
    def batch_to_gpu(self, data_list: List[np.ndarray]) -> List[torch.Tensor]:
        """批量将数据转移到GPU"""
        return [self.to_gpu(data) for data in data_list]
    
    def batch_to_cpu(self, tensor_list: List[torch.Tensor]) -> List[np.ndarray]:
        """批量将张量转移到CPU"""
        return [self.to_cpu(tensor) for tensor in tensor_list]
    
    def create_memory_pool(self, name: str, size: Tuple[int, ...], dtype: torch.dtype = torch.float32):
        """
        创建GPU内存池
        
        Args:
            name: 内存池名称
            size: 张量大小
            dtype: 数据类型
        """
        self.memory_pool[name] = torch.zeros(size, dtype=dtype, device=self.device)
        print(f"创建GPU内存池 '{name}': {size}, {dtype}")
    
    def get_from_pool(self, name: str) -> torch.Tensor:
        """从内存池获取张量"""
        return self.memory_pool.get(name)
    
    def set_memory_fraction(self, fraction: float = 0.9):
        """
        设置GPU内存使用比例
        
        Args:
            fraction: 内存使用比例 (0.0-1.0)
        """
        if self.device.type == 'cuda':
            try:
                torch.cuda.set_per_process_memory_fraction(fraction, self.device.index)
                print(f"GPU内存使用限制设置为: {fraction*100:.1f}%")
            except Exception as e:
                print(f"设置GPU内存限制失败: {e}")


class CudaGPUBatchProcessor:
    """CUDA GPU批处理器"""
    
    def __init__(self, gpu_manager: CudaGPUManager, batch_size: int = 1000):
        """
        初始化批处理器
        
        Args:
            gpu_manager: GPU管理器
            batch_size: 批处理大小 (CUDA上可以使用更大的批次)
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
            
            # 定期清理缓存
            if i % (self.batch_size * 10) == 0:
                self.gpu_manager.clear_cache()
        
        return np.concatenate(results, axis=0)


def check_cuda_compatibility() -> dict:
    """
    检查CUDA兼容性
    
    Returns:
        GPU信息字典
    """
    info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'system_memory_gb': psutil.virtual_memory().total / 1e9,
        'cpu_cores': psutil.cpu_count(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['gpu_count'] = torch.cuda.device_count()
        
        # 获取每个GPU的信息
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                'name': props.name,
                'memory_gb': props.total_memory / 1e9,
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessor_count': props.multi_processor_count
            })
        info['gpus'] = gpu_info
        
        # 检查当前GPU
        current_device = torch.cuda.current_device()
        info['current_gpu'] = current_device
        info['current_gpu_name'] = torch.cuda.get_device_name(current_device)
    else:
        info['cuda_error'] = 'CUDA not available'
    
    return info


def optimize_cuda_settings():
    """优化CUDA设置"""
    try:
        if torch.cuda.is_available():
            # 设置CUDA优化环境变量
            import os
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步执行
            os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # 启用cuDNN v8 API
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            print("CUDA设置已优化")
        else:
            print("CUDA不可用，跳过优化")
    except Exception as e:
        print(f"CUDA设置优化失败: {e}")


def estimate_cuda_memory_usage(population_size: int, feature_dim: int, n_samples: int) -> dict:
    """
    估算CUDA内存使用量
    
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
    
    # CUDA上的额外开销（相对较小）
    cuda_overhead = (population_memory + features_memory + scores_memory) * 0.1
    
    total_gpu_memory = population_memory + features_memory + scores_memory + cuda_overhead
    
    # 获取GPU内存信息
    gpu_memory_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1e9
            total_gpu = props.total_memory / 1e9
            gpu_memory_info[f'gpu_{i}'] = {
                'name': props.name,
                'total_gb': total_gpu,
                'allocated_gb': allocated,
                'available_gb': total_gpu - allocated
            }
    
    # 系统内存
    memory = psutil.virtual_memory()
    
    return {
        'population_mb': population_memory / 1e6,
        'features_mb': features_memory / 1e6,
        'scores_mb': scores_memory / 1e6,
        'cuda_overhead_mb': cuda_overhead / 1e6,
        'total_gpu_mb': total_gpu_memory / 1e6,
        'total_gpu_gb': total_gpu_memory / 1e9,
        'system_total_gb': memory.total / 1e9,
        'system_available_gb': memory.available / 1e9,
        'gpu_memory_info': gpu_memory_info
    }


# 全局GPU管理器实例
_global_cuda_gpu_manager = None

def get_cuda_gpu_manager(device_id: Optional[int] = None) -> CudaGPUManager:
    """获取全局CUDA GPU管理器实例"""
    global _global_cuda_gpu_manager
    if _global_cuda_gpu_manager is None:
        _global_cuda_gpu_manager = CudaGPUManager(device_id)
    return _global_cuda_gpu_manager


if __name__ == "__main__":
    # 测试CUDA GPU功能
    print("=== CUDA兼容性检查 ===")
    gpu_info = check_cuda_compatibility()
    for key, value in gpu_info.items():
        if key == 'gpus':
            print(f"{key}:")
            for i, gpu in enumerate(value):
                print(f"  GPU {i}: {gpu}")
        else:
            print(f"{key}: {value}")
    
    print("\n=== 内存使用估算 ===")
    memory_usage = estimate_cuda_memory_usage(
        population_size=1000,  # CUDA上可以使用更大的种群
        feature_dim=1400,
        n_samples=50000
    )
    for key, value in memory_usage.items():
        if key == 'gpu_memory_info':
            print(f"{key}:")
            for gpu_id, info in value.items():
                print(f"  {gpu_id}: {info}")
        elif 'gb' in key:
            print(f"{key}: {value:.2f} GB")
        elif 'mb' in key:
            print(f"{key}: {value:.1f} MB")
    
    print("\n=== CUDA GPU管理器测试 ===")
    gpu_manager = get_cuda_gpu_manager()
    
    # 测试数据转移
    test_data = np.random.randn(1000, 100).astype(np.float32)
    gpu_tensor = gpu_manager.to_gpu(test_data)
    cpu_result = gpu_manager.to_cpu(gpu_tensor)
    
    print(f"数据转移测试: {'通过' if np.allclose(test_data, cpu_result) else '失败'}")
    
    # 显示内存使用
    gpu_alloc, gpu_total, sys_used, sys_total = gpu_manager.get_memory_usage()
    print(f"GPU内存使用: {gpu_alloc:.2f} GB / {gpu_total:.2f} GB")
    print(f"系统内存使用: {sys_used:.2f} GB / {sys_total:.2f} GB")
    
    # 清理
    gpu_manager.clear_cache()
    print("GPU缓存已清理")