# 🚀 CUDA训练实时监控指南

本指南介绍如何使用实时动态图表来监控你的CUDA训练进度。

## 📋 目录

- [快速开始](#快速开始)
- [监控工具介绍](#监控工具介绍)
- [安装依赖](#安装依赖)
- [使用方法](#使用方法)
- [功能特性](#功能特性)
- [故障排除](#故障排除)

## 🚀 快速开始

### 方法1: 使用启动器（推荐）

```bash
# 启动监控启动器
python start_monitor.py
```

### 方法2: 直接启动

```bash
# 动态图表监控
python real_time_training_dashboard.py --auto

# 简单文本监控
python quick_monitor.py

# Windows批处理启动
start_dashboard.bat
```

## 🛠️ 监控工具介绍

### 1. 动态图表监控 (`real_time_training_dashboard.py`)

**最强大的监控工具**，提供实时更新的多图表面板：

- 📈 **适应度进化曲线**: 实时显示最佳和平均适应度变化
- ⏱️ **训练时间分析**: 监控每代训练时间
- 📊 **夏普比率趋势**: 跟踪交易策略性能指标
- 💾 **内存使用监控**: 实时显示系统内存使用情况
- 📊 **适应度分布**: 显示最近适应度的分布直方图
- 📋 **详细统计信息**: 包含所有关键训练指标

**特点:**
- 🔄 自动实时更新
- 🎨 美观的图形界面
- 📊 多维度数据展示
- 💾 低内存占用（滑动窗口）

### 2. 快速监控 (`quick_monitor.py`)

**最简单易用的监控工具**：

- 🎯 自动查找日志文件
- 📝 清晰的文本界面
- 📈 ASCII图表显示趋势
- ⚡ 快速启动，无依赖

**特点:**
- 🚀 零配置启动
- 📱 轻量级界面
- 🔍 智能文件查找
- 📊 包含基本图表

### 3. 启动器 (`start_monitor.py`)

**统一的监控入口**：

- 🎯 集成所有监控模式
- 🔧 高级配置选项
- 📁 自动检测日志文件
- ⚠️ 依赖检查和提示

## 📦 安装依赖

### 基础依赖（必需）

```bash
# Python 3.7+
python --version
```

### 图形界面依赖（可选，用于动态图表）

```bash
# 安装绘图库
pip install matplotlib pandas

# 或者使用requirements文件
pip install -r requirements.txt
```

### 检查依赖

```bash
# 运行启动器会自动检查依赖
python start_monitor.py
```

## 📖 使用方法

### 1. 启动训练

首先启动你的CUDA训练：

```bash
# 启动CUDA训练
python core/main_cuda.py
```

### 2. 启动监控

在另一个终端窗口中启动监控：

```bash
# 方法1: 使用启动器（推荐）
python start_monitor.py

# 方法2: 直接启动动态图表
python real_time_training_dashboard.py --auto

# 方法3: 快速文本监控
python quick_monitor.py
```

### 3. 监控选项

#### 动态图表监控选项

```bash
# 基本用法
python real_time_training_dashboard.py [日志文件路径]

# 自动查找日志文件
python real_time_training_dashboard.py --auto

# 自定义参数
python real_time_training_dashboard.py --auto --max-points 200 --interval 1000

# 文本模式（无图形界面）
python real_time_training_dashboard.py --auto --text-mode
```

**参数说明:**
- `--auto`: 自动查找训练日志文件
- `--max-points N`: 图表中显示的最大数据点数（默认100）
- `--interval N`: 更新间隔，毫秒（默认2000）
- `--text-mode`: 使用文本模式，无图形界面

#### 快速监控

```bash
# 自动启动
python quick_monitor.py
```

## 🎯 功能特性

### 实时数据监控

- ✅ **适应度指标**: 最佳、平均、标准差
- ✅ **交易指标**: 夏普比率、索提诺比率、最大回撤、总回报
- ✅ **性能指标**: 训练时间、内存使用
- ✅ **进度信息**: 当前代数、预计剩余时间

### 可视化功能

- 📈 **实时曲线图**: 适应度进化趋势
- ⏱️ **时间分析图**: 每代训练时间变化
- 📊 **分布直方图**: 适应度分布情况
- 💾 **资源监控图**: 内存使用趋势

### 智能功能

- 🔍 **自动文件查找**: 智能定位训练日志
- 🔄 **实时更新**: 文件变化自动检测
- 📱 **响应式界面**: 自适应窗口大小
- ⚠️ **错误处理**: 优雅的异常处理

## 📁 日志文件位置

监控工具会自动查找以下位置的日志文件：

```
results/training_history.jsonl          # 标准日志
results/training_history_cuda.jsonl     # CUDA日志
training_history.jsonl                  # 当前目录
../results/training_history.jsonl       # 上级目录
../results/training_history_cuda.jsonl  # 上级目录CUDA日志
```

## 🎨 界面预览

### 动态图表监控界面

```
🚀 CUDA训练实时监控面板
┌─────────────────┬─────────────────┬─────────────────┐
│   适应度进化曲线   │    每代训练时间    │    夏普比率趋势    │
│                │                │                │
│      📈        │      ⏱️         │      📊        │
│                │                │                │
├─────────────────┼─────────────────┼─────────────────┤
│   系统内存使用    │   适应度分布图    │    训练统计信息    │
│                │                │                │
│      💾        │      📊        │      📋        │
│                │                │                │
└─────────────────┴─────────────────┴─────────────────┘
```

### 文本监控界面

```
🚀================================================================================🚀
                            CUDA训练实时监控面板
🚀================================================================================🚀
📊 当前代数:      156
🏆 最佳适应度:    0.234567
📈 平均适应度:    0.198432
📉 标准差:        0.045123
⏱️  本代用时:     12.34 秒

💰 交易指标:
   📈 夏普比率:    1.2345
   📊 索提诺比率:  1.5678
   📉 最大回撤:   -0.0234
   💵 总回报率:    0.1567

🖥️  系统信息:
   💾 系统内存:    8.45 GB
   🎮 GPU内存:     3.21 GB

🎯 训练进度: [████████████████████████████████████████████████] 78.0%
   (156/200 代)

📊 统计信息:
   📝 总记录数: 156
   ⏰ 预计剩余: 0小时9分钟

================================================================================
🕐 最后更新: 2024-01-15 14:30:25
💡 按 Ctrl+C 停止监控
================================================================================
```

## 🔧 故障排除

### 常见问题

#### 1. 找不到日志文件

**问题**: `❌ 未找到训练日志文件`

**解决方案**:
- 确保训练已经开始并生成了日志文件
- 检查是否在正确的目录中运行监控工具
- 使用 `--auto` 参数自动查找日志文件

#### 2. 图形界面无法显示

**问题**: `⚠️ 图形库不可用，使用文本模式`

**解决方案**:
```bash
# 安装图形库
pip install matplotlib pandas

# 或者使用文本模式
python real_time_training_dashboard.py --auto --text-mode
```

#### 3. 权限错误

**问题**: 无法读取日志文件

**解决方案**:
- 检查文件权限
- 确保日志文件没有被其他程序锁定
- 尝试以管理员权限运行

#### 4. 内存不足

**问题**: 监控工具占用过多内存

**解决方案**:
```bash
# 减少最大数据点数
python real_time_training_dashboard.py --auto --max-points 50

# 使用轻量级监控
python quick_monitor.py
```

### 调试模式

如果遇到问题，可以启用详细输出：

```bash
# 添加调试信息
python -v real_time_training_dashboard.py --auto
```

## 💡 使用技巧

### 1. 多窗口监控

```bash
# 终端1: 启动训练
python core/main_cuda.py

# 终端2: 图形监控
python real_time_training_dashboard.py --auto

# 终端3: 文本监控（备用）
python quick_monitor.py
```

### 2. 自定义监控

```bash
# 高频更新（每秒）
python real_time_training_dashboard.py --auto --interval 1000

# 长期监控（更多历史数据）
python real_time_training_dashboard.py --auto --max-points 500

# 最小化资源使用
python real_time_training_dashboard.py --auto --max-points 20 --interval 5000
```

### 3. 保存监控数据

```bash
# 查看完整历史并保存图表
python tools/view_training_log.py --auto --save-plot training_progress.png
```

## 📞 获取帮助

如果你遇到问题或需要更多功能，可以：

1. 查看工具的帮助信息：
   ```bash
   python real_time_training_dashboard.py --help
   python start_monitor.py
   ```

2. 检查日志文件格式是否正确

3. 确认Python和依赖库版本兼容性

---

🎉 **祝你训练顺利！** 使用这些监控工具，你可以实时掌握训练进度，及时发现问题，优化训练效果。