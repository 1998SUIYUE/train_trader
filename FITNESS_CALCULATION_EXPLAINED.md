# 适应度计算详解

## 🎯 **适应度计算概述**

当前系统使用**多层次适应度评估**，根据配置不同有两种模式：

### 📊 **1. 单目标模式 (当前ULTRA_FAST_CONFIG)**
- 只关注**夏普比率**作为适应度指标
- 计算公式：`fitness = sharpe_ratio`

### 🎯 **2. 多目标模式 (完整增强版)**
- 综合多个交易指标
- 使用加权组合计算最终适应度

---

## 🔍 **详细计算流程**

### **第一步：个体参数提取**

每个个体包含以下参数：
```python
# 个体结构：[权重(feature_dim), 偏置, 买入阈值, 卖出阈值, 止损, 最大仓位, 最大回撤, 交易仓位]
weights = population[:, :feature_dim]           # 神经网络权重
biases = population[:, feature_dim]             # 偏置
buy_thresholds = population[:, feature_dim + 1] # 买入阈值 [0.55, 0.8]
sell_thresholds = population[:, feature_dim + 2] # 卖出阈值 [0.2, 0.45]
stop_losses = population[:, feature_dim + 3]    # 止损比例 [0.02, 0.08]
max_positions = population[:, feature_dim + 4]  # 最大仓位 [0.5, 1.0]
max_drawdowns = population[:, feature_dim + 5]  # 最大回撤 [0.1, 0.25]
trade_positions = population[:, feature_dim + 6] # 交易仓位 [0.01, 0.81]
```

### **第二步：信号生成**

```python
# 计算预测信号 [种群大小, 样本数]
signals = torch.sigmoid(torch.matmul(weights, features.T) + biases.unsqueeze(1))
```

**解释**：
- 使用神经网络计算每个时间点的交易信号
- Sigmoid函数将输出映射到[0,1]区间
- 0.5以上倾向买入，0.5以下倾向卖出

### **第三步：交易模拟**

#### **3.1 交易决策生成**
```python
# 生成买卖信号
buy_signals = (signals > buy_thresholds).float()   # 信号超过买入阈值
sell_signals = (signals < sell_thresholds).float() # 信号低于卖出阈值
```

#### **3.2 仓位计算**
```python
# 基于信号强度计算仓位
signal_strength = torch.sigmoid((signals - 0.5) * 4)  # 信号强度映射
positions = signal_strength * max_positions           # 实际仓位
```

#### **3.3 收益计算**
```python
# 计算每个时间点的收益
period_returns = labels.unsqueeze(0).expand(population_size, -1)
portfolio_returns = positions * period_returns        # 组合收益
portfolio_values = torch.cumprod(1 + portfolio_returns, dim=1)  # 累积价值
```

### **第四步：性能指标计算**

#### **4.1 夏普比率 (Sharpe Ratio)**
```python
mean_returns = torch.mean(portfolio_returns, dim=1)
std_returns = torch.std(portfolio_returns, dim=1) + 1e-8
sharpe_ratios = mean_returns / std_returns
```

**含义**：风险调整后收益，越高越好
- **计算**：平均收益 / 收益标准差
- **范围**：通常在-3到3之间，>1为良好

#### **4.2 最大回撤 (Maximum Drawdown)**
```python
running_max = torch.cummax(portfolio_values, dim=1)[0]
drawdowns = (running_max - portfolio_values) / running_max
max_drawdowns = torch.max(drawdowns, dim=1)[0]
```

**含义**：最大亏损幅度，越小越好
- **计算**：(历史最高点 - 当前点) / 历史最高点
- **范围**：0到1之间，<0.1为良好

#### **4.3 总收益率 (Total Return)**
```python
total_return = portfolio_values[:, -1] - 1.0  # 最终价值 - 初始价值
```

**含义**：总体收益率，越高越好

#### **4.4 胜率 (Win Rate)**
```python
winning_trades = daily_pnl > 0
win_rate = torch.sum(winning_trades, dim=1).float() / total_trades
```

**含义**：盈利交易占比，越高越好
- **范围**：0到1之间，>0.5为良好

#### **4.5 交易频率 (Trade Frequency)**
```python
position_changes = torch.abs(torch.diff(positions, dim=1))
trade_frequency = torch.mean(position_changes, dim=1)
```

**含义**：交易活跃度，适中为好
- 太高：过度交易，手续费高
- 太低：错失机会

#### **4.6 波动率 (Volatility)**
```python
volatility = torch.std(portfolio_returns, dim=1)
```

**含义**：收益波动性，越小越稳定

#### **4.7 盈亏比 (Profit Factor)**
```python
gross_profit = torch.sum(torch.clamp(daily_pnl, min=0), dim=1)
gross_loss = torch.sum(torch.clamp(daily_pnl, max=0), dim=1)
profit_factor = gross_profit / torch.abs(gross_loss)
```

**含义**：总盈利/总亏损，>1为盈利

---

## ⚖️ **适应度综合计算**

### **单目标模式 (当前配置)**
```python
fitness = sharpe_ratio  # 只关注夏普比率
```

### **多目标模式 (完整版)**
```python
# 加权组合
fitness = (
    0.3 * sharpe_ratio +           # 30% 夏普比率
    0.2 * (-max_drawdown) +        # 20% 最大回撤(最小化)
    0.25 * total_return +          # 25% 总收益率
    0.15 * win_rate +              # 15% 胜率
    0.1 * (-volatility)            # 10% 波动率(最小化)
)
```

**权重说明**：
- **最小化目标**：max_drawdown, volatility (前面加负号)
- **最大化目标**：sharpe_ratio, total_return, win_rate
- **权重和**：必须等于1.0

---

## 📈 **适应度值的含义**

### **夏普比率适应度 (当前模式)**
- **> 2.0**：优秀策略
- **1.0 - 2.0**：良好策略  
- **0.5 - 1.0**：一般策略
- **< 0.5**：较差策略
- **< 0**：亏损策略

### **多目标综合适应度**
- **> 1.5**：优秀策略
- **1.0 - 1.5**：良好策略
- **0.5 - 1.0**：一般策略
- **0 - 0.5**：较差策略
- **< 0**：亏损策略

---

## 🔧 **适应度优化机制**

### **1. 精英保留**
- 保留适应度最高的5%个体
- 确保优秀基因不丢失

### **2. 锦标赛选择**
- 随机选择几个个体比较适应度
- 适应度高的更容易被选中繁殖

### **3. 早停机制**
- 连续50代无改进则停止训练
- 基于平均适应度判断改进

### **4. 数据退火**
- 从简单数据逐渐过渡到复杂数据
- 帮助算法更好地学习交易模式

---

## 🎯 **当前配置的适应度特点**

由于使用`ULTRA_FAST_CONFIG`：

1. **单一目标**：只优化夏普比率
2. **简化计算**：避免复杂的多目标计算
3. **快速收敛**：专注单一指标，收敛更快
4. **风险调整**：夏普比率本身就考虑了风险

**优点**：
- 计算速度快
- 容易理解和调试
- 风险调整收益是交易的核心指标

**缺点**：
- 可能忽略其他重要指标
- 策略可能过于激进或保守

---

## 🔍 **如何解读JSON中的适应度**

在`enhanced_training_history.jsonl`中：

```json
{
  "best_fitness": 1.234567,      // 当前代最佳适应度(夏普比率)
  "avg_fitness": 0.987654,       // 当前代平均适应度
  "avg_sharpe_ratio": 1.234567,  // 种群平均夏普比率
  "avg_max_drawdown": 0.08,      // 种群平均最大回撤
  "avg_total_return": 0.15       // 种群平均总收益率
}
```

**分析要点**：
- `best_fitness`持续上升表示策略在改进
- `avg_fitness`与`best_fitness`差距大表示种群多样性高
- `avg_sharpe_ratio`应该与`best_fitness`接近(单目标模式)

这就是当前系统的完整适应度计算机制！