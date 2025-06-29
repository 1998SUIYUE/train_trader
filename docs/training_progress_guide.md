# 训练进度监控指南

## 概述

CUDA遗传算法训练系统提供了完整的进度监控功能，让您能够实时查看训练状态、分析训练趋势，并监控系统资源使用情况。

## 功能特性

### 🔥 实时进度显示
- **每代进度**: 显示当前代数、最佳适应度、平均适应度
- **性能指标**: 每代训练时间、无改进代数计数
- **资源监控**: GPU内存使用、系统内存使用
- **趋势分析**: 适应度变化趋势、改进率统计

### 📊 详细统计信息
- **训练统计**: 总训练时间、平均每代时间、ETA估算
- **适应度分析**: 历史最佳、最近趋势、改进历史
- **内存监控**: GPU内存分配、系统内存使用
- **进度条**: 可视化训练进度

### 📝 持久化日志
- **自动记录**: 所有训练结果自动保存到日志文件
- **JSON格式**: 结构化数据，便于分析和可视化
- **累积历史**: 多次训练的历史记录累积保存
- **会话管理**: 每次训练有唯一的会话ID

## 使用方法

### 1. 启动训练（自动启用进度监控）

```bash
python core/main_cuda.py
```

训练开始后，您将看到详细的实时进度显示：

```
🚀 CUDA遗传算法训练开始
================================================================================
目标代数: 1000
早停耐心: 100
开始时间: 2024-12-01 14:30:22
================================================================================

🔥 代数    1 (  0.1%) | 最佳: 0.123456 | 平均: 0.098765 | 标准差: 0.023456 | 无改进:   0 | 时间:  1.234s
   代数    2 (  0.2%) | 最佳: 0.134567 | 平均: 0.109876 | 标准差: 0.024567 | 无改进:   0 | 时间:  1.198s
   ...
```

### 2. 实时监控正在进行的训练

在另一个终端窗口中运行：

```bash
# 自动查找最新的训练日志
python tools/watch_training_progress.py --auto

# 或指定具体的日志文件
python tools/watch_training_progress.py results/training_history_cuda.jsonl
```

实时监控界面显示：

```
================================================================================
🚀 CUDA遗传算法训练实时监控
================================================================================
📈 当前代数: 150
🏆 最佳适应度: 0.456789
📊 平均适应度: 0.234567
📉 标准差: 0.123456
⏱️  本代用时: 1.23秒
🔄 无改进代数: 5
🖥️  GPU内存: 2.34GB
💾 系统内存: 8.76GB

📊 训练统计:
   总代数: 150
   历史最佳: 0.456789
   总训练时间: 0.52小时
   平均每代: 1.25秒
   最近趋势: 📈 上升
================================================================================
```

### 3. 查看训练历史

```bash
# 查看完整训练历史
python tools/view_training_log.py --auto

# 生成训练进度图表
python tools/view_training_log.py --auto --plot

# 保存图表到文件
python tools/view_training_log.py --auto --save-plot training_progress.png

# 只查看最近50代
python tools/view_training_log.py --auto --tail 50
```

### 4. 训练进度演示

```bash
python demo_training_progress.py
```

## 日志文件格式

### 训练日志结构

每条日志记录包含以下信息：

```json
{
  "generation": 150,
  "best_fitness": 0.456789,
  "avg_fitness": 0.234567,
  "std_fitness": 0.123456,
  "generation_time": 1.234,
  "no_improvement_count": 5,
  "gpu_memory_allocated": 2.34,
  "gpu_memory_reserved": 3.45,
  "system_memory_gb": 8.76
}
```

### 日志文件位置

- **CUDA训练日志**: `results/training_history_cuda.jsonl`
- **普通训练日志**: `results/training_history.jsonl`
- **检查点文件**: `results/checkpoints/checkpoint_gen_*.pt`
- **最佳个体**: `results/best_individual_*.npy`

## 进度监控配置

### 在训练代码中配置

```python
# 在 core/main_cuda.py 中的配置
results = ga.evolve(
    train_features,
    train_labels,
    show_detailed_progress=True,      # 启用详细进度监控
    progress_update_interval=1.0,     # 每秒更新一次
    generation_log_file=log_file,     # 日志文件路径
    generation_log_interval=1         # 每代都记录日志
)
```

### 进度监控器选项

```python
from training_progress_monitor import TrainingProgressMonitor

monitor = TrainingProgressMonitor(
    log_file=log_file,           # 日志文件路径
    update_interval=1.0          # 更新间隔（秒）
)

# 启动监控
monitor.start_training(
    total_generations=1000,      # 总代数
    early_stop_patience=100      # 早停耐心
)

# 更新进度
monitor.update_generation(generation, stats)

# 显示最终总结
monitor.display_final_summary(results)
```

## 高级功能

### 1. 自定义进度显示

```python
# 使用简化的进度显示器
from training_progress_monitor import SimpleProgressDisplay

display = SimpleProgressDisplay()
display.start_training(total_generations)
display.update_generation(generation, stats)
display.display_final_summary(results)
```

### 2. 训练统计分析

```python
# 获取训练统计信息
stats = monitor.get_statistics()
print(f"当前代数: {stats['current_generation']}")
print(f"最佳适应度: {stats['best_fitness']}")
print(f"改进次数: {stats['improvement_count']}")
```

### 3. 内存监控

训练过程中自动监控：
- GPU内存分配和缓存
- 系统内存使用
- 内存使用趋势

## 故障排除

### 1. 进度监控器不工作

```bash
# 检查依赖是否安装
pip install psutil

# 检查训练进度监控器是否可用
python -c "from training_progress_monitor import TrainingProgressMonitor; print('OK')"
```

### 2. 日志文件未创建

- 确保 `results` 目录存在
- 检查文件写入权限
- 确认日志文件路径正确

### 3. 实时监控无数据

- 确认训练正在运行
- 检查日志文件路径
- 确认日志文件正在更新

## 最佳实践

### 1. 训练前准备

```bash
# 1. 清理旧的日志（可选）
rm -f results/training_history_cuda.jsonl

# 2. 在新终端启动实时监控
python tools/watch_training_progress.py --auto

# 3. 在主终端启动训练
python core/main_cuda.py
```

### 2. 长期训练监控

```bash
# 定期检查训练进度
python tools/view_training_log.py --auto --tail 20

# 生成进度图表
python tools/view_training_log.py --auto --plot
```

### 3. 训练完成后分析

```bash
# 查看完整训练历史
python tools/view_training_log.py --auto

# 生成详细报告
python tools/view_training_log.py --auto --save-plot final_report.png
```

## 示例输出

### 训练过程中的详细进度

```
🔥 代数  150 ( 15.0%) | 最佳: 0.456789 | 平均: 0.234567 | 标准差: 0.123456 | 无改进:   5 | 时间:  1.234s
    📊 统计 | 总时间: 0.52h | 平均每代: 1.25s | ETA: 2.8h
    📈 趋势 | 最近最佳: 0.456789 | 最近平均: 0.234567
    🖥️  GPU内存 | 已分配: 2.34GB | 已缓存: 3.45GB
```

### 最终训练总结

```
================================================================================
🎉 训练完成！
================================================================================
📊 训练统计:
   总代数: 500
   最佳适应度: 0.78901234
   总训练时间: 2.34 小时
   平均每代时间: 16.848 秒
   总改进次数: 45
   最后改进代数: 487
   改进率: 9.0%

⏱️  性能统计:
   平均每代: 16.848s
   最快一代: 12.345s
   最慢一代: 23.456s

📈 适应度统计:
   初始适应度: 0.12345678
   最终适应度: 0.78901234
   总体改进: 0.66555556 (539.2%)
================================================================================
```

这个完整的训练进度监控系统让您能够：
- 实时了解训练状态
- 监控系统资源使用
- 分析训练趋势
- 优化训练参数
- 及时发现问题

现在您可以开始训练，并使用这些工具来监控和分析训练进度！