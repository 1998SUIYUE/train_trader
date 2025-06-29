# CUDA训练进度监控指南

## 概述
本指南介绍如何实时查看CUDA遗传算法训练的进度和统计信息。

## 🚀 快速开始

### 最简单的方式（推荐）
双击运行：`start_monitoring.bat`
- 提供友好的菜单界面
- 支持多种查看模式
- 适合Windows用户

### 命令行方式
```bash
# 查看当前训练状态
python cuda_progress_monitor.py

# 实时监控训练进度
python cuda_progress_monitor.py --watch

# 查看最近20代历史
python cuda_progress_monitor.py --tail 20
```

## 可用工具详解

### 1. 主要监控工具
```bash
python cuda_progress_monitor.py
```
**功能特点：**
- 显示详细的训练状态
- 包含适应度趋势ASCII图表
- 显示交易性能指标
- 显示GPU和系统内存使用
- 支持实时监控模式

**使用示例：**
```bash
# 查看当前进度
python cuda_progress_monitor.py

# 实时监控（3秒刷新）
python cuda_progress_monitor.py --watch

# 实时监控（自定义刷新间隔）
python cuda_progress_monitor.py --watch --interval 5

# 查看最近50代
python cuda_progress_monitor.py --tail 50

# 不显示图表
python cuda_progress_monitor.py --no-chart
```

### 2. 简化版工具
```bash
# 快速查看
python show_progress.py

# 简单实时监控
python watch_progress.py
```

### 3. 批处理启动器（Windows）
```bash
start_monitoring.bat
```
- 菜单驱动界面
- 无需记忆命令参数
- 支持重复使用

## 训练数据说明

训练日志文件位置：`results/training_history.jsonl`

每行包含的信息：
- `generation`: 当前代数
- `best_fitness`: 最佳适应度
- `mean_fitness`: 平均适应度
- `std_fitness`: 适应度标准差
- `generation_time`: 本代训练时间（秒）
- `system_memory_gb`: 系统内存使用（GB）
- `mean_sharpe_ratio`: 平均夏普比率
- `mean_sortino_ratio`: 平均索提诺比率
- `mean_max_drawdown`: 平均最大回撤
- `mean_overall_return`: 平均总回报
- `timestamp`: 时间戳

## 使用建议

### 开始新训练时：
1. 在一个终端启动训练：
   ```bash
   python core/main_cuda.py
   ```

2. 在另一个终端启动实时监控：
   ```bash
   python watch_progress.py
   ```

### 检查训练历史：
```bash
python show_progress.py
```

### 分析训练趋势：
```bash
python view_cuda_progress.py --tail 50
```

## 故障排除

### 如果看不到数据：
1. 确认训练已经开始
2. 检查日志文件是否存在：`results/training_history.jsonl`
3. 确认训练程序正在运行

### 如果显示乱码：
1. 确保终端支持UTF-8编码
2. 在Windows上可以使用批处理文件

### 如果监控停止响应：
1. 按Ctrl+C停止监控
2. 重新启动监控程序

## 性能监控

监控工具会显示：
- 训练进度（当前代数）
- 适应度变化趋势
- 每代训练时间
- 系统内存使用
- 交易策略性能指标

这些信息帮助你：
- 评估训练效果
- 监控系统资源使用
- 决定是否需要调整参数
- 预估训练完成时间