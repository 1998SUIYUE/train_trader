"""
测试参数退火功能
Test Parameter Annealing Functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from parameter_annealing_scheduler import (
    ParameterAnnealingScheduler, 
    ParameterAnnealingConfig, 
    ParameterAnnealingStrategy,
    ParameterRange
)

def test_parameter_annealing():
    """测试参数退火功能"""
    print("=== 参数退火功能测试 ===")
    
    # 创建配置
    config = ParameterAnnealingConfig(
        strategy=ParameterAnnealingStrategy.ADAPTIVE,
        total_generations=200,
        warmup_generations=20,
        mutation_rate_range=ParameterRange(
            initial_value=0.03,
            final_value=0.005,
            min_value=0.001,
            max_value=0.1
        ),
        crossover_rate_range=ParameterRange(
            initial_value=0.6,
            final_value=0.9,
            min_value=0.3,
            max_value=0.95
        ),
        elite_ratio_range=ParameterRange(
            initial_value=0.02,
            final_value=0.1,
            min_value=0.01,
            max_value=0.2
        )
    )
    
    scheduler = ParameterAnnealingScheduler(config)
    
    # 模拟训练过程
    generations = []
    mutation_rates = []
    crossover_rates = []
    elite_ratios = []
    fitness_values = []
    
    print("\n--- 模拟训练过程 ---")
    
    for generation in range(150):
        # 模拟适应度变化
        if generation < 30:
            # 早期快速改善
            fitness = 0.1 + generation * 0.02 + np.random.normal(0, 0.01)
        elif generation < 80:
            # 中期缓慢改善
            fitness = 0.7 + (generation - 30) * 0.005 + np.random.normal(0, 0.005)
        elif generation < 120:
            # 后期停滞
            fitness = 0.95 + np.random.normal(0, 0.002)
        else:
            # 最后阶段微调
            fitness = 0.95 + (generation - 120) * 0.0001 + np.random.normal(0, 0.001)
        
        fitness_values.append(fitness)
        
        # 获取退火后的参数
        params = scheduler.get_annealed_parameters(generation, fitness, fitness_values)
        
        generations.append(generation)
        mutation_rates.append(params['mutation_rate'])
        crossover_rates.append(params['crossover_rate'])
        elite_ratios.append(params['elite_ratio'])
        
        # 每20代打印一次
        if generation % 20 == 0:
            print(f"代数 {generation:3d}: 适应度={fitness:.4f}, "
                  f"变异率={params['mutation_rate']:.4f}, "
                  f"交叉率={params['crossover_rate']:.4f}, "
                  f"精英比例={params['elite_ratio']:.4f}")
    
    # 绘制参数变化图
    plt.figure(figsize=(15, 10))
    
    # 适应度变化
    plt.subplot(2, 2, 1)
    plt.plot(generations, fitness_values, 'b-', linewidth=2, label='适应度')
    plt.xlabel('代数')
    plt.ylabel('适应度')
    plt.title('适应度变化')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 变异率变化
    plt.subplot(2, 2, 2)
    plt.plot(generations, mutation_rates, 'r-', linewidth=2, label='变异率')
    plt.xlabel('代数')
    plt.ylabel('变异率')
    plt.title('变异率退火')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 交叉率变化
    plt.subplot(2, 2, 3)
    plt.plot(generations, crossover_rates, 'g-', linewidth=2, label='交叉率')
    plt.xlabel('代数')
    plt.ylabel('交叉率')
    plt.title('交叉率退火')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 精英比例变化
    plt.subplot(2, 2, 4)
    plt.plot(generations, elite_ratios, 'm-', linewidth=2, label='精英比例')
    plt.xlabel('代数')
    plt.ylabel('精英比例')
    plt.title('精英比例退火')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('parameter_annealing_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n参数退火测试完成！图表已保存为 parameter_annealing_test.png")

def test_different_strategies():
    """测试不同的参数退火策略"""
    print("\n=== 测试不同参数退火策略 ===")
    
    strategies = [
        ParameterAnnealingStrategy.LINEAR,
        ParameterAnnealingStrategy.EXPONENTIAL,
        ParameterAnnealingStrategy.COSINE,
        ParameterAnnealingStrategy.STEP,
        ParameterAnnealingStrategy.CYCLIC,
    ]
    
    plt.figure(figsize=(15, 12))
    
    for i, strategy in enumerate(strategies):
        config = ParameterAnnealingConfig(
            strategy=strategy,
            total_generations=100,
            warmup_generations=10,
        )
        
        scheduler = ParameterAnnealingScheduler(config)
        
        generations = list(range(100))
        mutation_rates = []
        
        for gen in generations:
            params = scheduler.get_annealed_parameters(gen, 0.5)
            mutation_rates.append(params['mutation_rate'])
        
        plt.subplot(3, 2, i + 1)
        plt.plot(generations, mutation_rates, linewidth=2, label=f'{strategy.value}')
        plt.xlabel('代数')
        plt.ylabel('变异率')
        plt.title(f'{strategy.value.upper()} 策略')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('parameter_annealing_strategies.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"策略对比测试完成！图表已保存为 parameter_annealing_strategies.png")

def test_adaptive_behavior():
    """测试自适应行为"""
    print("\n=== 测试自适应参数退火行为 ===")
    
    config = ParameterAnnealingConfig(
        strategy=ParameterAnnealingStrategy.ADAPTIVE,
        total_generations=200,
        warmup_generations=20,
        performance_window=10,
        improvement_threshold=0.001,
        adaptation_rate=0.2
    )
    
    scheduler = ParameterAnnealingScheduler(config)
    
    # 模拟不同的性能场景
    scenarios = [
        "快速改善期",
        "缓慢改善期", 
        "性能停滞期",
        "性能下降期",
        "恢复期"
    ]
    
    generations = []
    mutation_rates = []
    fitness_values = []
    
    generation = 0
    
    for scenario in scenarios:
        print(f"\n--- {scenario} ---")
        
        for i in range(40):
            if scenario == "快速改善期":
                fitness = 0.1 + generation * 0.02 + np.random.normal(0, 0.005)
            elif scenario == "缓慢改善期":
                fitness = 0.5 + generation * 0.002 + np.random.normal(0, 0.003)
            elif scenario == "性能停滞期":
                fitness = 0.8 + np.random.normal(0, 0.001)
            elif scenario == "性能下降期":
                fitness = 0.8 - i * 0.005 + np.random.normal(0, 0.002)
            else:  # 恢复期
                fitness = 0.6 + i * 0.01 + np.random.normal(0, 0.003)
            
            fitness_values.append(fitness)
            
            params = scheduler.get_annealed_parameters(generation, fitness, fitness_values)
            
            generations.append(generation)
            mutation_rates.append(params['mutation_rate'])
            
            if i % 10 == 0:
                print(f"  代数 {generation:3d}: 适应度={fitness:.4f}, 变异率={params['mutation_rate']:.4f}")
            
            generation += 1
    
    # 绘制自适应行为
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(generations, fitness_values, 'b-', linewidth=2, label='适应度')
    plt.xlabel('代数')
    plt.ylabel('适应度')
    plt.title('适应度变化（不同性能场景）')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加场景分割线
    for i, scenario in enumerate(scenarios):
        plt.axvline(x=i*40, color='red', linestyle='--', alpha=0.5)
        plt.text(i*40 + 20, max(fitness_values) * 0.9, scenario, 
                rotation=0, ha='center', fontsize=8)
    
    plt.subplot(2, 1, 2)
    plt.plot(generations, mutation_rates, 'r-', linewidth=2, label='变异率')
    plt.xlabel('代数')
    plt.ylabel('变异率')
    plt.title('自适应变异率调整')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加场景分割线
    for i, scenario in enumerate(scenarios):
        plt.axvline(x=i*40, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('adaptive_parameter_annealing.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"自适应行为测试完成！图表已保存为 adaptive_parameter_annealing.png")

if __name__ == "__main__":
    try:
        # 基础功能测试
        test_parameter_annealing()
        
        # 策略对比测试
        test_different_strategies()
        
        # 自适应行为测试
        test_adaptive_behavior()
        
        print("\n🎉 所有参数退火测试完成！")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()