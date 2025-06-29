# CUDA 12.9 环境设置指南

本指南将帮助您在CUDA 12.9环境下设置和运行GPU加速的遗传算法交易员训练程序。

## 🚀 快速开始

### 1. 环境要求

- **操作系统**: Windows 10/11 或 Linux
- **GPU**: NVIDIA GPU (支持CUDA)
- **CUDA**: 12.1 或更高版本 (推荐 12.9)
- **Python**: 3.8 或更高版本
- **显存**: 至少 4GB (推荐 8GB+)

### 2. 自动安装 (Windows)

运行自动安装脚本：

```powershell
# 以管理员权限运行PowerShell
powershell -ExecutionPolicy Bypass -File setup/install_cuda129.ps1
```

### 3. 手动安装

如果自动安装失败，可以手动安装：

```bash
# 安装PyTorch (CUDA 12.1版本，兼容CUDA 12.9)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements_cuda129.txt
```

### 4. 验证安装

运行测试脚本验证环境：

```bash
python test_cuda_environment.py
```

### 5. 开始训练

```bash
python core/main_cuda.py
```

## 📋 详细配置

### CUDA版本兼容性

| CUDA版本 | PyTorch索引URL | 兼容性 |
|----------|----------------|--------|
| 12.1     | cu121         | ✅ 推荐 |
| 12.4     | cu121         | ✅ 兼容 |
| 12.9     | cu121         | ✅ 兼容 |

### GPU内存需求

| 种群大小 | 特征维度 | 最小显存 | 推荐显存 |
|----------|----------|----------|----------|
| 500      | 1400     | 2GB      | 4GB      |
| 1000     | 1400     | 4GB      | 6GB      |
| 2000     | 1400     | 6GB      | 8GB      |
| 5000     | 1400     | 12GB     | 16GB     |

## ⚙️ 配置参数

### 基本配置

在 `core/main_cuda.py` 中修改 `TRAINING_CONFIG`：

```python
TRAINING_CONFIG = {
    # 核心参数
    "population_size": 1000,        # 种群大小
    "generations": -1,              # 训练代数 (-1=无限)
    "batch_size": 1000,             # 批处理大小
    
    # GPU设置
    "gpu_device_id": 0,             # GPU设备ID
    "gpu_memory_fraction": 0.9,     # GPU内存使用比例
    
    # 其他参数...
}
```

### 预设配置

选择适合您硬件的预设配置：

```python
# 快速测试 (适合任何GPU)
ACTIVE_CONFIG = QUICK_TEST_CONFIG

# 高性能 (适合RTX 3080/4080等)
ACTIVE_CONFIG = HIGH_PERFORMANCE_CONFIG

# 极限性能 (适合RTX 4090/A100等)
ACTIVE_CONFIG = EXTREME_PERFORMANCE_CONFIG
```

## 🔧 性能优化

### 1. GPU内存优化

```python
# 设置GPU内存使用比例
gpu_manager.set_memory_fraction(0.9)

# 启用内存池
gpu_manager.create_memory_pool("population", (1000, 1400))
```

### 2. 计算优化

```python
# 启用cuDNN优化
torch.backends.cudnn.benchmark = True

# 启用TensorFloat-32 (Ampere架构)
torch.backends.cuda.matmul.allow_tf32 = True
```

### 3. 批处理优化

根据GPU显存调整批处理大小：

- **4GB显存**: batch_size = 500-1000
- **8GB显存**: batch_size = 1000-2000
- **16GB显存**: batch_size = 2000-4000

## 🐛 常见问题

### Q1: CUDA不可用

**症状**: `torch.cuda.is_available()` 返回 `False`

**解决方案**:
1. 检查NVIDIA驱动程序版本
2. 重新安装PyTorch CUDA版本
3. 验证CUDA安装：`nvcc --version`

### Q2: 显存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
1. 减小种群大小：`population_size = 500`
2. 减小批处理大小：`batch_size = 500`
3. 设置内存限制：`gpu_memory_fraction = 0.8`

### Q3: 性能较慢

**可能原因**:
1. 使用了CPU版本的PyTorch
2. 数据传输开销过大
3. GPU利用率不足

**解决方案**:
1. 确认使用CUDA版本：`torch.version.cuda`
2. 增大批处理大小
3. 启用GPU优化选项

### Q4: torch.scan不可用

**症状**: `torch.func.scan` 不存在

**解决方案**:
- 这是正常的，程序会自动使用传统回测方法
- 不影响训练，但可能略微降低性能

## 📊 性能基准

### RTX 4060 (8GB)

- **推荐配置**: 种群1000, 批次1000
- **训练速度**: ~2-3秒/代
- **内存使用**: ~6GB

### RTX 4080 (16GB)

- **推荐配置**: 种群2000, 批次2000
- **训练速度**: ~1-2秒/代
- **内存使用**: ~12GB

### RTX 4090 (24GB)

- **推荐配置**: 种群5000, 批次3000
- **训练速度**: ~0.5-1秒/代
- **内存使用**: ~20GB

## 📁 文件结构

```
├── core/
│   ├── main_cuda.py           # CUDA版主程序
│   └── main_gpu.py            # DirectML版主程序
├── src/
│   ├── cuda_gpu_utils.py      # CUDA GPU工具
│   ├── cuda_accelerated_ga.py # CUDA遗传算法
│   ├── gpu_utils.py           # DirectML GPU工具
│   └── gpu_accelerated_ga.py  # DirectML遗传算法
├── setup/
│   └── install_cuda129.ps1    # CUDA安装脚本
├── requirements_cuda129.txt   # CUDA依赖文件
└── test_cuda_environment.py   # 环境测试脚本
```

## 🚀 开始训练

1. **验证环境**:
   ```bash
   python test_cuda_environment.py
   ```

2. **快速测试**:
   ```bash
   # 修改main_cuda.py中的配置
   ACTIVE_CONFIG = QUICK_TEST_CONFIG
   python core/main_cuda.py
   ```

3. **正式训练**:
   ```bash
   # 选择适合的配置
   ACTIVE_CONFIG = HIGH_PERFORMANCE_CONFIG
   python core/main_cuda.py
   ```

## 📈 监控训练

### 实时监控

```bash
# 监控GPU使用情况
nvidia-smi -l 1

# 监控训练日志
tail -f results/training_history_cuda.jsonl
```

### 训练输出

程序会实时显示：
- 当前代数和适应度
- GPU内存使用情况
- 训练速度和预计完成时间

## 💡 提示

1. **首次运行**: 建议先使用 `QUICK_TEST_CONFIG` 验证环境
2. **长期训练**: 使用 `LONG_TERM_CONFIG` 并启用检查点保存
3. **性能调优**: 根据GPU型号选择合适的种群大小和批次大小
4. **稳定性**: 启用早停机制避免过拟合

## 📞 支持

如果遇到问题：

1. 运行 `test_cuda_environment.py` 诊断环境
2. 检查 `results/training_history_cuda.jsonl` 日志
3. 查看GPU内存使用情况：`nvidia-smi`

---

**祝您训练顺利！** 🎉