#!/usr/bin/env python3
"""
演示Tensor与标量比较的机制
"""

import torch
import numpy as np

def demonstrate_tensor_scalar_comparison():
    """演示张量与标量的比较过程"""
    
    print("🔍 Tensor与标量比较演示")
    print("=" * 50)
    
    # 创建一个小的示例scores张量 (类似您的数据)
    scores = torch.tensor([
        [0.7796, 0.8842, 0.8912, 0.0575, 0.0470, 0.1200],  # 个体1的决策分数
        [0.0838, 0.0663, 0.0631, 0.3376, 0.3282, 0.2496],  # 个体2的决策分数
        [0.0135, 0.0183, 0.0183, 0.7564, 0.7166, 0.7921],  # 个体3的决策分数
        [0.9921, 0.9966, 0.9979, 0.0449, 0.0445, 0.0426],  # 个体4的决策分数
    ])
    
    print(f"原始scores张量形状: {scores.shape}")
    print(f"原始scores:\n{scores}")
    print()
    
    # 设置交易阈值
    buy_threshold = 0.6
    sell_threshold = 0.4
    
    print(f"买入阈值: {buy_threshold}")
    print(f"卖出阈值: {sell_threshold}")
    print()
    
    # 生成交易信号
    buy_signals = scores > buy_threshold
    sell_signals = scores < sell_threshold
    
    print("🔥 买入信号 (scores > 0.6):")
    print(f"形状: {buy_signals.shape}")
    print(f"布尔张量:\n{buy_signals}")
    print()
    
    print("🔥 卖出信号 (scores < 0.4):")
    print(f"形状: {sell_signals.shape}")
    print(f"布尔张量:\n{sell_signals}")
    print()
    
    # 统计信号数量
    total_signals = scores.numel()  # 总元素数
    buy_count = torch.sum(buy_signals).item()
    sell_count = torch.sum(sell_signals).item()
    neutral_count = total_signals - buy_count - sell_count
    
    print("📊 信号统计:")
    print(f"总信号数: {total_signals}")
    print(f"买入信号: {buy_count} ({buy_count/total_signals*100:.1f}%)")
    print(f"卖出信号: {sell_count} ({sell_count/total_signals*100:.1f}%)")
    print(f"中性信号: {neutral_count} ({neutral_count/total_signals*100:.1f}%)")
    print()
    
    # 详细分析每个个体
    print("🧬 每个个体的交易信号分析:")
    for i in range(scores.shape[0]):
        individual_scores = scores[i]
        individual_buy = buy_signals[i]
        individual_sell = sell_signals[i]
        
        buy_positions = torch.where(individual_buy)[0].tolist()
        sell_positions = torch.where(individual_sell)[0].tolist()
        
        print(f"个体 {i+1}:")
        print(f"  分数: {individual_scores.tolist()}")
        print(f"  买入时刻: {buy_positions} (分数: {[individual_scores[pos].item() for pos in buy_positions]})")
        print(f"  卖出时刻: {sell_positions} (分数: {[individual_scores[pos].item() for pos in sell_positions]})")
        print()

def demonstrate_broadcasting():
    """演示PyTorch的广播机制"""
    
    print("📡 PyTorch广播机制演示")
    print("=" * 50)
    
    # 创建不同形状的张量
    tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    scalar = 3
    
    print(f"2D张量: {tensor_2d}")
    print(f"标量: {scalar}")
    print()
    
    # 比较操作
    result = tensor_2d > scalar
    print(f"比较结果 (tensor_2d > {scalar}):")
    print(f"{result}")
    print()
    
    # 展示广播过程
    print("广播过程解释:")
    print(f"标量 {scalar} 被广播为: [[{scalar}, {scalar}, {scalar}], [{scalar}, {scalar}, {scalar}]]")
    print("然后进行逐元素比较:")
    for i in range(tensor_2d.shape[0]):
        for j in range(tensor_2d.shape[1]):
            val = tensor_2d[i, j].item()
            comparison = val > scalar
            print(f"  {val} > {scalar} = {comparison}")

def demonstrate_real_trading_logic():
    """演示真实的交易逻辑"""
    
    print("💰 真实交易逻辑演示")
    print("=" * 50)
    
    # 模拟一个交易员在6个时间点的决策分数
    trader_scores = torch.tensor([0.2, 0.7, 0.8, 0.3, 0.1, 0.9])
    buy_threshold = 0.6
    sell_threshold = 0.4
    
    print(f"交易员决策分数: {trader_scores.tolist()}")
    print(f"买入阈值: {buy_threshold}, 卖出阈值: {sell_threshold}")
    print()
    
    # 生成信号
    buy_signals = trader_scores > buy_threshold
    sell_signals = trader_scores < sell_threshold
    
    print("时间点分析:")
    for t in range(len(trader_scores)):
        score = trader_scores[t].item()
        action = "买入" if buy_signals[t] else ("卖出" if sell_signals[t] else "中性")
        print(f"时刻 {t}: 分数={score:.1f} → {action}")

if __name__ == "__main__":
    demonstrate_tensor_scalar_comparison()
    print("\n" + "="*70 + "\n")
    demonstrate_broadcasting()
    print("\n" + "="*70 + "\n")
    demonstrate_real_trading_logic()