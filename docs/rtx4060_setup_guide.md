# RTX 4060 训练环境配置指南

## 🎯 概述

本指南专门为使用 NVIDIA RTX 4060 显卡的用户提供训练环境配置说明。RTX 4060 拥有 8GB 显存，非常适合中等规模的遗传算法训练。

## 📋 系统要求

- **显卡**: NVIDIA RTX 4060 (8GB 显存)
- **操作系统**: Windows 10/11 或 Linux
- **Python**: 3.8 或更高版本
- **显存**: 建议至少 6GB 可用显存

## 🔧 环境配置步骤

### 1. 安装 NVIDIA 驱动
确保安装了最新的 NVIDIA 驱动程序：
- 访问 [NVIDIA 官网](https://www.nvidia.com/drivers/)
- 下载并安装适合 RTX 4060 的最新驱动

### 2. 安装 CUDA Toolkit
```bash
# 检查当前 CUDA 版本
nvidia-smi

# 推荐安装 CUDA 11.8 或 12.x
# 从 NVIDIA 官网下载 CUDA Toolkit
```

### 3. 安装 Python 依赖
```bash
# 安装支持 CUDA 的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install numpy pandas matplotlib tqdm
```

### 4. 验证 CUDA 环境
```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"显卡名称: {torch.cuda.get_device_name(0)}")
print(f"显存容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

## 🚀 运行训练

### 1. 使用 RTX 4060 优化版本
```bash
cd core
python main_cuda.py
```

### 2. RTX 4060 推荐配置
```python
TRAINING_CONFIG = {
    "population_size": 1000,    # RTX 4060 可支持的种群大小
    "generations": 100,         # 训练代数
    "mutation_rate": 0.01,
    "crossover_rate": 0.8,
    "elite_ratio": 0.1,
}
```

## 📊 性能优化建议

### 1. 显存管理
- **种群大小**: 建议 500-1500，根据具体模型调整
- **批处理大小**: 建议 256-512
- **定期清理**: 每 10 代清理一次显存缓存

### 2. 训练参数优化
```python
# RTX 4060 优化参数
{
    "population_size": 1000,        # 充分利用 8GB 显存
    "batch_size": 512,              # 平衡速度和显存使用
    "checkpoint_interval": 20,      # 定期保存检查点
    "memory_cleanup_interval": 10   # 定期清理显存
}
```

### 3. 监控显存使用
```python
# 在训练过程中监控显存
allocated = torch.cuda.memory_allocated() / 1024**3
cached = torch.cuda.memory_reserved() / 1024**3
print(f"显存使用: {allocated:.2f}GB / {cached:.2f}GB")
```

## 🔍 查看训练结果

### 1. 实时查看日志
```bash
cd tools
python view_training_log.py --auto
```

### 2. 绘制训练曲线
```bash
python view_training_log.py --auto --plot
```

### 3. 查看最近训练结果
```bash
python view_training_log.py --auto --tail 20
```

## ⚡ 性能预期

### RTX 4060 性能指标
- **种群大小 1000**: 约 15-20 秒/代
- **显存使用**: 约 4-6GB
- **推荐训练时长**: 2-4 小时 (100-200 代)

### 与其他显卡对比
| 显卡型号 | 显存 | 推荐种群大小 | 性能 |
|---------|------|-------------|------|
| RTX 4060 | 8GB | 1000 | 基准 |
| RTX 4070 | 12GB | 1500 | +30% |
| RTX 4080 | 16GB | 2000 | +60% |

## 🛠️ 故障排除

### 常见问题

1. **CUDA 不可用**
   ```bash
   # 检查 PyTorch 安装
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **显存不足**
   ```python
   # 减少种群大小
   "population_size": 500  # 从 1000 减少到 500
   ```

3. **训练速度慢**
   ```python
   # 检查是否使用了 CUDA
   print(f"当前设备: {torch.cuda.current_device()}")
   ```

## 📈 进阶优化

### 1. 混合精度训练
```python
# 使用 FP16 减少显存使用
with torch.cuda.amp.autocast():
    # 训练代码
```

### 2. 梯度累积
```python
# 模拟更大的批处理大小
accumulation_steps = 4
for i in range(0, len(data), batch_size):
    # 训练步骤
    if (i + 1) % accumulation_steps == 0:
        # 更新参数
```

### 3. 动态批处理
```python
# 根据显存使用动态调整批处理大小
def get_optimal_batch_size():
    memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
    if memory_usage > 0.8:
        return batch_size // 2
    return batch_size
```

## 📞 技术支持

如果遇到问题，请检查：
1. NVIDIA 驱动是否为最新版本
2. CUDA 版本是否与 PyTorch 兼容
3. 显存是否足够
4. Python 环境是否正确配置

## 🎉 开始训练

配置完成后，运行以下命令开始训练：
```bash
cd core
python main_cuda.py
```

训练完成后，使用以下命令查看结果：
```bash
cd tools
python view_training_log.py --auto --plot
```