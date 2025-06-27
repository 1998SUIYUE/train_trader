"""
GPU版遗传算法交易员训练主程序
使用DirectML后端支持AMD GPU
"""

import time
from pathlib import Path
import json
import torch
import torch_directml
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpu_utils import WindowsGPUManager, get_windows_gpu_manager
from gpu_accelerated_ga import WindowsGPUAcceleratedGA, WindowsGAConfig
from data_processor import GPUDataProcessor

# 确保results目录存在
results_dir = Path('../results')
results_dir.mkdir(exist_ok=True)





def main():
    """主函数 - 集成配置与自动化流程"""

    # ==============================================================================
    # ======================= 在这里修改你的训练参数 ============================
    # ==============================================================================
    TRAINING_CONFIG = {
        # --- 数据参数 ---
        "data_directory": "../data",  # 数据文件所在的目录
        "window_size": 350,          # 特征工程的滑动窗口大小
        "normalization": "rolling",  # 归一化方法: 'relative', 'rolling', 'minmax_local', 'hybrid'

        # --- 遗传算法参数 ---
        "population_size": 5,      # 种群大小 (GPU建议500-1000)
        "generations": -1,           # 最大进化代数 (-1表示无限训练)
        "mutation_rate": 0.01,       # 变异率 (建议0.01-0.05)
        "crossover_rate": 0.8,       # 交叉率 (建议0.7-0.9)
        "elite_ratio": 0.1,          # 精英比例 (建议0.05-0.1)

        # --- 检查点参数 ---
        "save_checkpoints": True,    # 是否自动保存检查点
        "checkpoint_interval": 10,   # 每隔多少代保存一次
        "results_dir": "../results", # 所有结果和日志的输出目录
        
        # --- 新增：持续训练参数 ---
        "continuous_training": True, # 是否启用持续训练模式
        "save_generation_results": True,  # 是否每代保存结果
        "generation_log_interval": 1,     # 每隔多少代记录到文件
        "auto_save_best": True,           # 是否自动保存最佳个体
    }
    # ==============================================================================
    # ======================= 参数修改区域结束 ==================================
    # ==============================================================================

    # --- 1. 自动化设置与路径管理 ---
    output_dir = Path(TRAINING_CONFIG["results_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    data_dir = Path(TRAINING_CONFIG["data_directory"])
    
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    print("=== GPU加速遗传算法交易员训练开始 (自动化模式) ===")
    print("\n--- 训练参数 ---")
    for key, value in TRAINING_CONFIG.items():
        print(f"{key}: {value}")
    print("--------------------\n")
    
    # --- 2. 自动发现最新的数据文件 ---
    try:
        data_files = sorted(data_dir.glob("*.csv"), key=os.path.getmtime, reverse=True)
        if not data_files:
            print(f"数据目录 '{data_dir}' 中未找到任何.csv文件。")
            return
        latest_data_file = data_files[0]
        print(f"自动选择最新的数据文件: {latest_data_file}")
    except FileNotFoundError:
        print(f"数据目录 '{data_dir}' 不存在。")
        return

     # --- 3. 自动发现最新的检查点 ---
    load_checkpoint_path = None
    if TRAINING_CONFIG["save_checkpoints"]:
        checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=os.path.getmtime, reverse=True)
        if checkpoints:
            latest_checkpoint = checkpoints[0]
            print(f"发现最新的检查点: {latest_checkpoint}")
            
            # 检查检查点中的参数是否与当前配置匹配
            try:
                ckpt = torch.load(latest_checkpoint, map_location='cpu')
                if ckpt['config'].population_size != TRAINING_CONFIG['population_size']:
                    print(f"警告: 检查点中的种群大小 ({ckpt['config'].population_size}) 与当前配置 ({TRAINING_CONFIG['population_size']}) 不匹配。")
                    print("将忽略检查点，开始新的训练。")
                else:
                    load_checkpoint_path = latest_checkpoint
                    print(f"检查点参数匹配，将从 '{load_checkpoint_path}' 继续训练。")
            except Exception as e:
                print(f"无法加载或解析检查点 '{latest_checkpoint}': {e}")
                print("将忽略检查点，开始新的训练。")
        else:
            print("未发现检查点，将开始新的训练。")

    try:
        # --- 4. 初始化GPU和数据处理器 ---
        print("初始化GPU环境...")
        gpu_manager = get_windows_gpu_manager()
        print(f"✅ GPU加速已{'启用' if gpu_manager.device.type == 'privateuseone' else '禁用'}")

        print("开始数据处理...")
        processor = GPUDataProcessor(
            window_size=TRAINING_CONFIG["window_size"],
            normalization_method=TRAINING_CONFIG["normalization"],
            gpu_manager=gpu_manager
        )
        train_features, train_labels = processor.load_and_process_data(latest_data_file)
        print(f"训练数据形状: {train_features.shape}, 标签数据形状: {train_labels.shape}")

        # --- 5. 配置并初始化遗传算法 ---
        ga_config = WindowsGAConfig(
            population_size=TRAINING_CONFIG["population_size"],
            max_generations=TRAINING_CONFIG["generations"],
            mutation_rate=TRAINING_CONFIG["mutation_rate"],
            crossover_rate=TRAINING_CONFIG["crossover_rate"],
            elite_ratio=TRAINING_CONFIG["elite_ratio"],
            feature_dim=train_features.shape[1]
        )
        print(f"遗传算法配置: {ga_config}")
        ga = WindowsGPUAcceleratedGA(ga_config, gpu_manager)

        # --- 6. 智能加载或初始化种群 ---
        if load_checkpoint_path:
            ga.load_checkpoint(str(load_checkpoint_path))
        else:
            print("初始化新的种群...")
            ga.initialize_population(seed=int(time.time())) # 使用时间戳作为种子

        # --- 7. 开始进化 ---
        print("开始进化过程...")
        
        # 使用固定的日志文件名，所有训练结果都追加到同一个文件
        generation_log_file = output_dir / "training_history.jsonl"
        print(f"📝 训练日志将写入: {generation_log_file}")
        
        results = ga.evolve(
            train_features,
            train_labels,
            save_checkpoints=TRAINING_CONFIG["save_checkpoints"],
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=TRAINING_CONFIG["checkpoint_interval"],
            # 新增参数
            continuous_training=TRAINING_CONFIG["continuous_training"],
            save_generation_results=TRAINING_CONFIG["save_generation_results"],
            generation_log_file=generation_log_file,
            generation_log_interval=TRAINING_CONFIG["generation_log_interval"],
            auto_save_best=TRAINING_CONFIG["auto_save_best"],
            output_dir=output_dir
        )

        # --- 8. 保存最终结果 ---
        print("训练完成，正在保存最终结果...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存最佳个体
        best_individual_path = output_dir / f"best_individual_{timestamp}.npy"
        np.save(best_individual_path, results['best_individual'])
        
        # 训练历史已通过实时日志记录，无需重复保存

        # --- 9. 输出最终报告 ---
        print("="*60)
        print("              GPU加速遗传算法训练完成")
        print("="*60)
        print(f"  - 最佳适应度: {results['best_fitness']:.4f}")
        print(f"  - 总训练时间: {results['total_time']:.2f}秒")
        print(f"  - 最终代数:   {results['final_generation']}")
        print(f"  - 最佳个体:   {best_individual_path}")
        print(f"  - 实时日志:   {generation_log_file}")
        print(f"  - 结果目录:   {output_dir}")
        print("="*60)

    except Exception as e:
        print(f"训练过程中发生严重错误: {e}")
        raise

if __name__ == "__main__":
    main()