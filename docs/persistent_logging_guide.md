# 持久化训练日志系统使用指南

## 概述

新的持久化日志系统将所有训练结果写入到一个固定的日志文件中（`training_history.jsonl`），这样无论您何时开始训练，数据都会追加到同一个文件里，实现真正的持久化记录。

## 主要特性

### 1. 统一日志文件
- 所有训练会话的数据都写入 `../results/training_history.jsonl`
- 每次训练不再创建新的日志文件，而是追加到现有文件
- 支持跨天、跨周的长期训练记录

### 2. 会话管理
- 每个训练会话都有唯一的会话ID（格式：`YYYYMMDD_HHMMSS`）
- 记录会话开始、每代结果、会话结束三种事件类型
- 可以区分不同时间的训练会话

### 3. 丰富的日志内容
每个日志条目包含：
- `event_type`: 事件类型（`training_session_start`、`generation_result`、`training_session_end`）
- `session_id`: 会话ID
- `timestamp`: 时间戳
- 训练配置、代数结果、性能指标等详细信息

## 使用方法

### 1. 开始训练
正常运行训练程序：
```bash
cd core
python main_gpu.py
```

训练开始时会自动：
- 创建会话ID
- 记录会话开始信息
- 每代训练结果实时写入日志
- 训练结束时记录会话结束信息

### 2. 查看训练日志

#### 基本查看
```bash
cd tools
python view_training_log.py --auto
```

#### 列出所有训练会话
```bash
python view_training_log.py --auto --list-sessions
```

#### 查看特定会话的数据
```bash
python view_training_log.py --auto --session 20241201_140000
```

#### 只显示最近几代的结果
```bash
python view_training_log.py --auto --tail 10
```

#### 绘制训练进度图
```bash
python view_training_log.py --auto --plot
```

#### 保存图表
```bash
python view_training_log.py --auto --save-plot training_progress.png
```

### 3. 手动指定日志文件
如果不使用 `--auto` 参数，可以手动指定日志文件路径：
```bash
python view_training_log.py ../results/training_history.jsonl
```

## 日志文件格式

日志文件采用JSONL格式（每行一个JSON对象），包含三种事件类型：

### 训练会话开始
```json
{
  "event_type": "training_session_start",
  "session_id": "20241201_140000",
  "timestamp": "2024-12-01 14:00:00",
  "config": {
    "population_size": 100,
    "generations": 50,
    "mutation_rate": 0.01
  },
  "data_file": "../data/sample_data.csv",
  "checkpoint_loaded": null,
  "starting_generation": 0
}
```

### 每代训练结果
```json
{
  "event_type": "generation_result",
  "session_id": "20241201_140000",
  "timestamp": "2024-12-01 14:01:00",
  "generation": 1,
  "best_fitness": 0.15,
  "mean_fitness": 0.08,
  "generation_time": 16.0,
  "mean_sharpe_ratio": 0.9,
  "mean_sortino_ratio": 1.12
}
```

### 训练会话结束
```json
{
  "event_type": "training_session_end",
  "session_id": "20241201_140000",
  "timestamp": "2024-12-01 14:05:30",
  "final_generation": 5,
  "best_fitness": 0.35,
  "total_time": 95.5,
  "best_individual_file": "../results/best_individual_20241201_140530.npy"
}
```

## 优势

1. **持久化记录**：所有训练历史都保存在一个文件中，永不丢失
2. **会话管理**：可以清楚地区分不同时间的训练会话
3. **灵活查看**：支持按会话、按时间、按代数等多种方式查看数据
4. **可视化分析**：内置绘图功能，直观展示训练进度
5. **易于分析**：标准的JSONL格式，便于后续数据分析

## 演示

运行演示脚本来体验新的日志系统：
```bash
cd tools
python demo_persistent_logging.py
```

然后使用各种查看命令来探索生成的演示数据。

## 注意事项

- 日志文件会随着训练次数增加而变大，建议定期备份
- 如果需要重新开始记录，可以删除或重命名现有的 `training_history.jsonl` 文件
- 所有时间戳都使用本地时间格式