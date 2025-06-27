#!/usr/bin/env python3
"""
持久化日志系统演示脚本
展示如何使用新的训练日志系统
"""

import json
import time
from pathlib import Path

def demo_log_format():
    """演示新的日志格式"""
    
    # 模拟日志文件路径
    log_file = Path("../results/training_history.jsonl")
    
    print("=== 持久化训练日志系统演示 ===\n")
    
    print("1. 新的日志格式特点:")
    print("   - 所有训练结果都写入同一个文件: training_history.jsonl")
    print("   - 每条记录都有event_type字段区分记录类型")
    print("   - 每个训练会话都有唯一的session_id")
    print("   - 支持多次训练的历史记录累积\n")
    
    print("2. 记录类型:")
    print("   - training_session_start: 训练会话开始")
    print("   - generation_result: 每代训练结果")
    print("   - training_session_end: 训练会话结束\n")
    
    print("3. 示例日志记录:")
    
    # 会话开始记录示例
    session_start = {
        "event_type": "training_session_start",
        "session_id": "20241201_143022",
        "timestamp": "2024-12-01 14:30:22",
        "config": {
            "population_size": 500,
            "generations": 100,
            "mutation_rate": 0.01
        },
        "data_file": "../data/latest_data.csv",
        "checkpoint_loaded": None,
        "starting_generation": 0
    }
    
    print("会话开始记录:")
    print(json.dumps(session_start, indent=2, ensure_ascii=False))
    print()
    
    # 代数结果记录示例
    generation_result = {
        "event_type": "generation_result",
        "session_id": "20241201_143022",
        "timestamp": "2024-12-01 14:30:45",
        "generation": 1,
        "best_fitness": 0.1234,
        "mean_fitness": 0.0987,
        "std_fitness": 0.0234,
        "generation_time": 23.45,
        "system_memory_gb": 2.1,
        "mean_sharpe_ratio": 1.23,
        "mean_sortino_ratio": 1.45
    }
    
    print("代数结果记录:")
    print(json.dumps(generation_result, indent=2, ensure_ascii=False))
    print()
    
    # 会话结束记录示例
    session_end = {
        "event_type": "training_session_end",
        "session_id": "20241201_143022",
        "timestamp": "2024-12-01 15:45:30",
        "final_generation": 100,
        "best_fitness": 0.5678,
        "total_time": 4508.23,
        "best_individual_file": "../results/best_individual_20241201_154530.npy"
    }
    
    print("会话结束记录:")
    print(json.dumps(session_end, indent=2, ensure_ascii=False))
    print()
    
    print("4. 使用方法:")
    print("   - 每次运行训练，所有结果都会追加到同一个文件")
    print("   - 明天再训练时，新的结果会继续追加，不会覆盖之前的记录")
    print("   - 可以使用 tools/view_training_log.py 查看和分析历史记录")
    print("   - 支持按会话ID过滤和分析特定训练会话的结果\n")
    
    print("5. 查看日志命令示例:")
    print("   python tools/view_training_log.py ../results/training_history.jsonl")
    print("   python tools/view_training_log.py ../results/training_history.jsonl --plot")
    print("   python tools/view_training_log.py ../results/training_history.jsonl --tail 50")

if __name__ == "__main__":
    demo_log_format()