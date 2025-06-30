#!/usr/bin/env python3
"""
测试JSON日志记录功能
Test JSON logging functionality
"""

import json
import time
from pathlib import Path
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from enhanced_monitoring import EnhancedMonitor, MonitoringConfig, PerformanceMetrics

def test_json_logging():
    """测试JSON日志记录"""
    print("=== 测试JSON日志记录功能 ===")
    
    # 创建测试目录
    test_dir = Path("test_json_output")
    test_dir.mkdir(exist_ok=True)
    
    # 配置监控系统
    config = MonitoringConfig(
        log_file=test_dir / "test_training_history",
        save_interval=1,  # 每代都保存
        detailed_logging=True,
        track_diversity=False,  # 禁用多样性跟踪以提高速度
        track_convergence=True,
        export_format="json"
    )
    
    # 初始化监控器
    monitor = EnhancedMonitor(config)
    monitor.start_monitoring(total_generations=10)
    
    print(f"测试日志文件路径: {config.log_file}")
    
    # 模拟10代训练
    for generation in range(10):
        # 模拟基础统计
        basic_stats = {
            'best_fitness': 0.5 + generation * 0.01,
            'avg_fitness': 0.3 + generation * 0.008,
            'std_fitness': 0.1,
            'generation_time': 2.0,
            'no_improvement_count': max(0, 5 - generation),
            'mutation_rate': 0.01,
            'crossover_rate': 0.8,
            'elite_ratio': 0.05,
        }
        
        # 模拟多目标统计
        multi_objective_stats = {
            'pareto_front_size': 20 + generation,
            'hypervolume': 0.1 + generation * 0.01,
            'pareto_ratio': 0.1 + generation * 0.005,
            'objective_stats': {
                'sharpe_ratio': {'mean': 1.0 + generation * 0.01},
                'max_drawdown': {'mean': 0.1 - generation * 0.001},
                'total_return': {'mean': 0.05 + generation * 0.002},
                'win_rate': {'mean': 0.6 + generation * 0.001},
            }
        }
        
        # 模拟数据退火统计
        annealing_stats = {
            'data_ratio': min(1.0, 0.3 + generation * 0.07),
            'complexity_score': generation / 10,
            'strategy': 'progressive',
            'progress': generation / 10,
        }
        
        # 更新监控
        metrics = monitor.update_metrics(
            generation, basic_stats, multi_objective_stats, annealing_stats
        )
        
        print(f"代数 {generation}: 最佳适应度={metrics.best_fitness:.4f}")
        
        # 短暂延迟模拟真实训练
        time.sleep(0.1)
    
    # 检查生成的文件
    json_file = test_dir / "test_training_history.jsonl"
    backup_file = test_dir / "test_training_history.jsonl.backup"
    csv_file = test_dir / "test_training_history.csv"
    simple_file = test_dir / "test_training_history.simple.log"
    
    print(f"\n=== 文件检查结果 ===")
    
    files_to_check = [
        ("主JSON文件", json_file),
        ("备份JSON文件", backup_file),
        ("CSV文件", csv_file),
        ("简单日志文件", simple_file),
    ]
    
    for file_desc, file_path in files_to_check:
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {file_desc}: {file_path} (大小: {size} 字节)")
            
            # 如果是JSON文件，验证内容
            if file_path.suffix == '.jsonl':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    print(f"   - 包含 {len(lines)} 行数据")
                    
                    # 验证第一行和最后一行是否为有效JSON
                    if lines:
                        first_line = json.loads(lines[0])
                        last_line = json.loads(lines[-1])
                        print(f"   - 第一代: generation={first_line.get('generation', 'N/A')}")
                        print(f"   - 最后代: generation={last_line.get('generation', 'N/A')}")
                        print(f"   - 包含字段: {list(first_line.keys())[:5]}...")
                        
                except Exception as e:
                    print(f"   ❌ JSON验证失败: {e}")
        else:
            print(f"❌ {file_desc}: {file_path} (不存在)")
    
    # 生成训练总结
    summary = monitor.get_training_summary()
    print(f"\n=== 训练总结 ===")
    print(f"总代数: {summary['total_generations']}")
    print(f"最佳适应度: {summary['best_fitness_ever']:.4f}")
    print(f"收敛状态: {summary['convergence_achieved']}")
    
    # 导出详细报告
    report_path = test_dir / "detailed_report.json"
    if monitor.export_detailed_report(report_path):
        print(f"✅ 详细报告已导出: {report_path}")
    else:
        print(f"❌ 详细报告导出失败")
    
    print("\n=== 测试完成 ===")
    
    # 返回测试结果
    return {
        'json_file_exists': json_file.exists(),
        'json_file_size': json_file.stat().st_size if json_file.exists() else 0,
        'backup_exists': backup_file.exists(),
        'total_generations': summary['total_generations'],
        'test_dir': str(test_dir)
    }

if __name__ == "__main__":
    result = test_json_logging()
    
    print(f"\n=== 最终测试结果 ===")
    print(f"JSON文件生成: {'✅ 成功' if result['json_file_exists'] else '❌ 失败'}")
    print(f"文件大小: {result['json_file_size']} 字节")
    print(f"记录代数: {result['total_generations']}")
    print(f"测试目录: {result['test_dir']}")