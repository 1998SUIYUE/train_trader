# 🧬 遗传算法交易员基因结构与交易机制详解

## 📊 基因结构概览

**注意**: 本文档描述的是旧版本的复杂基因结构。新版本已简化为只有1400个特征权重。详见 [simplified_trading_mechanism.md](simplified_trading_mechanism.md)

每个AI交易员的基因由 **1405个参数** 组成，分为两大类：

```
基因总长度: 1405 (已简化为1400)
├── 特征权重 (1400维): 用于分析市场数据
└── 风险参数 (5维): 控制交易行为和风险管理 (已移除)
```

## 🧬 详细基因结构

### 1. 特征权重部分 (前1400维)

```python
weights = population[:, :1400]  # 形状: (种群大小, 1400)
```

**作用**: 这1400个权重决定了交易员如何解读市场数据
- 每个权重对应一个市场特征（如价格变化、成交量、技术指标等）
- 权重值范围: 通常在 [-1, 1] 之间
- **正权重**: 该特征支持买入决策
- **负权重**: 该特征支持卖出决策
- **权重绝对值大小**: 表示该特征的重要程度

**初始化**:
```python
population[:, :1400] *= 0.1  # 小的初始权重，避免过度拟合
```

### 2. 风险参数部分 (后5维)

#### 参数1-2: 决策阈值 (维度1400-1401)
```python
buy_threshold = population[:, 1400]    # 买入阈值
sell_threshold = population[:, 1401]   # 卖出阈值
```
- **范围**: [0.01, 0.11] (初始化) → [0.001, 0.2] (约束后)
- **作用**: 控制交易信号的敏感度
- **买入条件**: `决策分数 > buy_threshold`
- **卖出条件**: `决策分数 < -sell_threshold`

#### 参数3: 止损比例 (维度1402)
```python
stop_loss = population[:, 1402]
```
- **范围**: [0.02, 0.08] (初始化) → [0.01, 0.1] (约束后)
- **作用**: 当价格下跌超过此比例时强制平仓
- **例子**: 0.05 表示价格下跌5%时止损

#### 参数4: 最大仓位比例 (维度1403)
```python
max_position = population[:, 1403]
```
- **范围**: [0.2, 0.8] (初始化) → [0.1, 1.0] (约束后)
- **作用**: 控制单次交易的最大资金使用比例
- **例子**: 0.5 表示最多使用50%的资金进行单次交易

#### 参数5: 最大回撤限制 (维度1404)
```python
max_drawdown = population[:, 1404]
```
- **范围**: [0.05, 0.15] (初始化) → [0.02, 0.3] (约束后)
- **作用**: 当账户回撤超过此比例时强制平仓所有头寸
- **例子**: 0.1 表示账户回撤10%时强制清仓

## 🔄 交易决策流程

### 第1步: 特征分析
```python
# 计算决策分数 (GPU并行计算)
scores = torch.mm(weights, features.T)  # (种群大小, 时间步数)
```
- 将1400个特征权重与当前市场特征相乘求和
- 得到每个交易员在每个时间点的"决策分数"
- **正分数**: 倾向买入
- **负分数**: 倾向卖出
- **分数绝对值**: 表示信心强度

### 第2步: 信号生成
```python
# 生成交易信号
buy_signals = scores > buy_threshold.unsqueeze(1)   # 买入信号
sell_signals = scores < -sell_threshold.unsqueeze(1) # 卖出信号
```

### 第3步: 交易执行逻辑

#### 买入条件检查
```python
can_buy = (positions == 0) & buy_signals[:, t] & (~force_close)
new_position = torch.where(can_buy, max_position, positions)
```
**买入条件**:
1. 当前无持仓 (`positions == 0`)
2. 买入信号触发 (`buy_signals[:, t]`)
3. 未被强制平仓 (`~force_close`)

#### 卖出条件检查
```python
can_sell = (positions > 0) & sell_signals[:, t]
new_position = torch.where(can_sell, torch.zeros_like(positions), new_position)
```
**卖出条件**:
1. 当前有持仓 (`positions > 0`)
2. 卖出信号触发 (`sell_signals[:, t]`)

#### 风险控制机制

**止损检查**:
```python
stop_loss_trigger = (positions > 0) & (price_change < -stop_loss)
positions = torch.where(stop_loss_trigger, torch.zeros_like(positions), positions)
```

**回撤控制**:
```python
current_drawdown = (peak_equity - equity) / peak_equity
force_close = current_drawdown > max_drawdown
positions = torch.where(force_close, torch.zeros_like(positions), positions)
```

## 📈 收益计算与适应度评估

### 收益计算
```python
# 每个时间步的收益
period_return = positions * price_change
equity += period_return

# 累计统计
sum_returns += period_return
sum_sq_returns += period_return.pow(2)
```

### 适应度指标

#### 1. 夏普比率 (Sharpe Ratio)
```python
mean_returns = sum_returns / n_samples
variance = sum_sq_returns / n_samples - mean_returns.pow(2)
std_returns = torch.sqrt(variance)
sharpe_ratios = mean_returns / (std_returns + 1e-9) * np.sqrt(252)
```
- **衡量**: 风险调整后的收益
- **公式**: (平均收益 - 无风险收益) / 收益标准差
- **年化**: 乘以√252 (交易日数)

#### 2. 索提诺比率 (Sortino Ratio)
```python
downside_variance = downside_sum_sq_returns / n_samples
downside_std = torch.sqrt(downside_variance)
sortino_ratios = mean_returns / (downside_std + 1e-9) * np.sqrt(252)
```
- **衡量**: 只考虑下行风险的收益比率
- **优势**: 不惩罚上行波动

#### 3. 最大回撤 (Maximum Drawdown)
```python
max_drawdowns = (peak_equity - equity) / peak_equity
```
- **衡量**: 从历史最高点到当前的最大损失比例

#### 4. 交易频率 (负向指标)
```python
trade_frequency = torch.sum(position_changes > 0, dim=1).float() / n_samples
normalized_frequency = torch.clamp(trade_frequency, 0.0, 1.0)
```
- **衡量**: 交易活跃度，交易越频繁，适应度越低

### 综合适应度函数
```python
fitness = (0.5 * sharpe_ratios -      # 50%权重：风险调整收益
           0.3 * max_drawdowns -      # 30%权重：回撤控制
           0.2 * normalized_frequency)    # 20%权重：交易频率惩罚
```

## 🧬 进化机制

### 1. 选择 (Tournament Selection)
- 随机选择若干个体进行"锦标赛"
- 适应度最高的个体获胜
- 获胜者有更高概率繁殖后代

### 2. 交叉 (Crossover)
```python
# 均匀交叉
mask = torch.rand(gene_length) < 0.5
child1 = torch.where(mask, parent1, parent2)
child2 = torch.where(mask, parent2, parent1)
```
- 两个父代的基因随机组合
- 产生具有混合特征的后代

### 3. 变异 (Mutation)
```python
mutation_mask = torch.rand(pop_size, gene_length) < mutation_rate
mutation_values = torch.randn(pop_size, gene_length) * 0.01
new_population += mutation_mask * mutation_values
```
- 以小概率随机改变基因
- 引入新的交易策略可能性
- 避免陷入局部最优

### 4. 精英保留 (Elitism)
```python
elite_count = int(pop_size * elite_ratio)
elite_indices = torch.topk(fitness_scores, elite_count).indices
new_population[:elite_count] = population[elite_indices]
```
- 保留当代最优个体
- 确保优秀策略不会丢失

## 🎯 实际交易示例

假设某个交易员的基因参数为：
- `buy_threshold = 0.05`
- `sell_threshold = 0.03`
- `stop_loss = 0.04`
- `max_position = 0.6`
- `max_drawdown = 0.08`

**交易场景**:
1. **市场分析**: 特征权重计算得出决策分数 = 0.07
2. **买入判断**: 0.07 > 0.05 (买入阈值) → 触发买入信号
3. **仓位管理**: 使用60%资金买入
4. **风险控制**: 
   - 如果价格下跌4%，触发止损
   - 如果账户回撤8%，强制清仓
5. **卖出判断**: 当决策分数 < -0.03时，卖出平仓

## 🚀 系统优势

1. **并行计算**: GPU同时评估数百个交易策略
2. **自动优化**: 通过进化自动发现最优参数组合
3. **风险控制**: 内置多层风险管理机制
4. **适应性强**: 能够适应不同市场环境
5. **可解释性**: 每个参数都有明确的交易含义

这套系统通过模拟自然进化过程，让AI交易员在虚拟环境中"学习"和"进化"，最终找到在特定市场条件下表现最佳的交易策略。