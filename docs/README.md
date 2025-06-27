# 基于遗传算法的交易员训练系统

## 🎯 项目概述

使用遗传算法训练一个以**夏普率**为核心衡量指标的交易机器人。系统支持GPU加速，可以高效地训练和优化交易策略。

## 🚀 快速开始

### 1. 环境配置

#### 方案A: GPU加速版本（推荐）
```powershell
# 安装Python 3.11 + torch-directml
.\setup\install_python311.ps1

# 测试环境
.\setup\test_environment.ps1
```

#### 方案B: CPU版本（Python 3.13兼容）
```powershell
# 直接使用CPU版本，无需额外配置
python core/main_cpu.py --data_file data/your_data.csv
```

### 2. 准备数据

将您的交易数据（CSV格式）放入 `data/` 目录。数据应包含以下列：
- `Open`: 开盘价
- `High`: 最高价
- `Low`: 最低价
- `Close`: 收盘价
- `Volume`: 成交量

### 3. 开始训练

#### GPU版本
```powershell
py -3.11 core/main_gpu.py --data_file data/your_data.csv --population_size 500 --generations 300
```

#### CPU版本
```powershell
python core/main_cpu.py --data_file data/your_data.csv --population_size 200 --generations 100
```

## 📁 项目结构

```
trading_ai_project/
├── 📁 core/                    # 核心程序
│   ├── main_gpu.py            # GPU版本主程序
│   └── main_cpu.py            # CPU版本主程序
├── 📁 src/                     # 源代码模块
│   ├── gpu_accelerated_ga_windows.py  # GPU遗传算法
│   ├── gpu_utils_windows.py   # GPU工具
│   └── data_processor.py      # 数据处理
├── 📁 setup/                   # 环境配置
│   ├── install_python311.ps1  # Python安装脚本
│   └── test_environment.ps1   # 环境测试脚本
├── 📁 docs/                    # 文档
│   ├── README.md              # 项目说明
│   └── installation_guide.md  # 安装指南
├── 📁 data/                    # 数据目录
├── 📁 results/                 # 结果输出
└── 📁 examples/                # 示例和工具
```

## ⚙️ 参数说明

### 主要参数

| 参数 | 说明 | GPU版本默认值 | CPU版本默认值 |
|------|------|---------------|---------------|
| `--population_size` | 种群大小 | 500 | 200 |
| `--generations` | 进化代数 | 300 | 100 |
| `--window_size` | 滑动窗口大小 | 350 | 350 |
| `--mutation_rate` | 变异率 | 0.01 | 0.01 |
| `--crossover_rate` | 交叉率 | 0.8 | 0.8 |

### 归一化方法

- `relative`: 相对价格归一化（推荐）
- `rolling`: 滚动标准化
- `minmax`: MinMax归一化

## 🔧 技术架构

### 核心算法
- **遗传算法**: 进化优化交易策略
- **特征工程**: 1400维OHLC特征向量
- **风险管理**: 止损、仓位控制、回撤限制
- **适应度函数**: 夏普率为主要指标

### 性能优化
- **GPU加速**: 使用DirectML支持AMD GPU
- **批量计算**: 向量化操作提升效率
- **内存管理**: 智能内存池减少开销

## 📊 性能对比

| 版本 | 硬件要求 | 训练速度 | 种群规模 | 适用场景 |
|------|----------|----------|----------|----------|
| GPU版本 | AMD GPU + Python 3.11 | 🚀🚀🚀🚀🚀 | 500-1000 | 大规模训练 |
| CPU版本 | 任意CPU + Python 3.13 | 🚀🚀🚀 | 100-500 | 快速测试 |

## 📈 结果分析

训练完成后，结果保存在 `results/` 目录：

- `best_individual_*.npy`: 最优个体基因
- `training_history_*.json`: 训练历史记录
- `config_*.json`: 训练配置参数
- `training_*.log`: 详细日志

## 🛠️ 故障排除

### 常见问题

1. **torch-directml安装失败**
   - 确保使用Python 3.11
   - 运行: `py -3.11 -m pip install torch-directml`

2. **GPU不可用**
   - 检查AMD GPU驱动
   - 使用CPU版本作为备选

3. **内存不足**
   - 减少种群大小
   - 减少窗口大小

4. **数据格式错误**
   - 确保CSV包含必要列
   - 检查数据中的缺失值

## 💡 使用建议

1. **首次使用**: 先用小参数测试
2. **数据质量**: 确保数据完整性
3. **参数调优**: 逐步增加复杂度
4. **结果验证**: 使用样本外数据测试

## 📞 技术支持

如遇问题，请：
1. 运行环境测试脚本
2. 查看日志文件
3. 检查数据格式
4. 尝试减少参数规模

---

**开始您的AI交易员训练之旅！** 🚀