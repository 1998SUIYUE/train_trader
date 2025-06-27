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
        # ==================== 核心训练参数 ====================
        
        # --- 数据配置 ---
        "data_directory": "../data",     # 数据文件目录
        "window_size": 350,             # 特征工程窗口大小
        "normalization": "rolling",     # 归一化方法: 'rolling', 'minmax_local', 'hybrid'
        "batch_size":500,
        # --- 遗传算法参数 ---
        "population_size": 500,          # 种群大小 (推荐: 500-2000)
        "generations": -1,              # 训练代数 (-1=无限训练, 推荐: 50-500)
        "mutation_rate": 0.01,           # 变异率 (推荐: 0.005-0.02)
        "crossover_rate": 0.8,           # 交叉率 (推荐: 0.7-0.9)
        "elite_ratio": 0.1,              # 精英保留比例 (推荐: 0.05-0.15)
        "early_stop_patience": 50,       # 无改进停止代数 (推荐: 30-100)
        
        # --- 交易策略参数 (Sigmoid[0,1]区间) ---
        "buy_threshold": 0.6,            # 买入阈值 (>0.5偏向买入, 推荐: 0.55-0.8)
        "sell_threshold": 0.4,           # 卖出阈值 (<0.5偏向卖出, 推荐: 0.2-0.45)
        
        # --- 风险管理参数 ---
        "stop_loss": 0.05,               # 止损比例 (推荐: 0.02-0.08)
        "max_position": 0.2,             # 最大仓位 (推荐: 0.5-1.0)
        "max_drawdown": 0.2,             # 最大回撤限制 (推荐: 0.1-0.3)
        
        # --- 适应度权重 (总和应为1.0) ---
        "sharpe_weight": 0.5,            # 夏普比率权重
        "drawdown_weight": 0.3,          # 回撤惩罚权重
        "stability_weight": 0.2,         # 交易稳定性权重
        
        # ==================== 系统配置 ====================
        
        # --- 保存设置 ---
        "results_dir": "../results",     # 结果输出目录
        "save_checkpoints": True,        # 是否保存检查点
        "checkpoint_interval": 300,       # 检查点保存间隔
        "auto_save_best": True,          # 是否自动保存最佳个体
        
        # --- 日志设置 ---
        "save_generation_results": True, # 是否保存每代结果
        "generation_log_interval": 1,    # 日志记录间隔
    }
    
    # ==============================================================================
    # ======================== 预设配置模板 (可选择使用) =========================
    # ==============================================================================
    
    # 🚀 快速测试配置
    QUICK_TEST_CONFIG = {
        **TRAINING_CONFIG,
        "population_size": 50,
        "generations": 10,
        "checkpoint_interval": 5,
    }
    
    # 💪 高性能配置 (适合高端显卡)
    HIGH_PERFORMANCE_CONFIG = {
        **TRAINING_CONFIG,
        "population_size": 1500,
        "generations": 200,
        "checkpoint_interval": 25,
    }
    
    # 🛡️ 保守交易策略
    CONSERVATIVE_CONFIG = {
        **TRAINING_CONFIG,
        "buy_threshold": 0.7,            # 更严格的买入条件
        "sell_threshold": 0.3,           # 更严格的卖出条件
        "stop_loss": 0.03,               # 更严格的止损
        "max_position": 0.6,             # 较小的仓位
        "max_drawdown": 0.15,            # 较小的回撤容忍
        "sharpe_weight": 0.6,            # 更重视风险调整收益
        "drawdown_weight": 0.4,
        "stability_weight": 0.0,
    }
    
    # ⚡ 激进交易策略
    AGGRESSIVE_CONFIG = {
        **TRAINING_CONFIG,
        "buy_threshold": 0.55,           # 更宽松的买入条件
        "sell_threshold": 0.45,          # 更宽松的卖出条件
        "stop_loss": 0.08,               # 更宽松的止损
        "max_position": 1.0,             # 满仓交易
        "max_drawdown": 0.3,             # 更大的回撤容忍
        "sharpe_weight": 0.3,            # 更重视收益
        "drawdown_weight": 0.2,
        "stability_weight": 0.5,         # 重视交易频率
    }
    
    # 🔄 长期训练配置
    LONG_TERM_CONFIG = {
        **TRAINING_CONFIG,
        "generations": -1,               # 无限训练
        "early_stop_patience": 100,      # 更长的耐心
        "checkpoint_interval": 50,       # 更长的保存间隔
    }
    
    # ==============================================================================
    # =================== 选择要使用的配置 (修改这里) ===========================
    # ==============================================================================
    
    # 选择配置 (取消注释想要使用的配置)
    ACTIVE_CONFIG = TRAINING_CONFIG           # 默认配置
    # ACTIVE_CONFIG = QUICK_TEST_CONFIG       # 快速测试
    # ACTIVE_CONFIG = HIGH_PERFORMANCE_CONFIG # 高性能
    # ACTIVE_CONFIG = CONSERVATIVE_CONFIG     # 保守策略
    # ACTIVE_CONFIG = AGGRESSIVE_CONFIG       # 激进策略
    # ACTIVE_CONFIG = LONG_TERM_CONFIG        # 长期训练
    
    # ==============================================================================
    # ======================= 参数修改区域结束 ==================================
    # ==============================================================================

    # --- 1. 自动化设置与路径管理 ---
    output_dir = Path(ACTIVE_CONFIG["results_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    data_dir = Path(ACTIVE_CONFIG["data_directory"])
    
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    print("=== GPU加速遗传算法交易员训练开始 (自动化模式) ===")
    print("\n--- 训练参数 ---")
    for key, value in ACTIVE_CONFIG.items():
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
    if ACTIVE_CONFIG["save_checkpoints"]:
        checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=os.path.getmtime, reverse=True)
        if checkpoints:
            latest_checkpoint = checkpoints[0]
            print(f"发现最新的检查点: {latest_checkpoint}")
            
            # 检查检查点中的参数是否与当前配置匹配
            try:
                ckpt = torch.load(latest_checkpoint, map_location='cpu')
                if ckpt['config'].population_size != ACTIVE_CONFIG['population_size']:
                    print(f"警告: 检查点中的种群大小 ({ckpt['config'].population_size}) 与当前配置 ({ACTIVE_CONFIG['population_size']}) 不匹配。")
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
            window_size=ACTIVE_CONFIG["window_size"],
            normalization_method=ACTIVE_CONFIG["normalization"],
            gpu_manager=gpu_manager
        )
        train_features, train_labels = processor.load_and_process_data(latest_data_file)
        print(f"训练数据形状: {train_features.shape}, 标签数据形状: {train_labels.shape}")

        # --- 5. 配置并初始化遗传算法 ---
        ga_config = WindowsGAConfig(
            population_size=ACTIVE_CONFIG["population_size"],
            max_generations=ACTIVE_CONFIG["generations"],
            mutation_rate=ACTIVE_CONFIG["mutation_rate"],
            crossover_rate=ACTIVE_CONFIG["crossover_rate"],
            elite_ratio=ACTIVE_CONFIG["elite_ratio"],
            feature_dim=train_features.shape[1],
            # 交易策略参数
            buy_threshold=ACTIVE_CONFIG["buy_threshold"],
            sell_threshold=ACTIVE_CONFIG["sell_threshold"],
            # 风险管理参数
            stop_loss=ACTIVE_CONFIG["stop_loss"],
            max_position=ACTIVE_CONFIG["max_position"],
            max_drawdown=ACTIVE_CONFIG["max_drawdown"],
            # 适应度函数权重
            sharpe_weight=ACTIVE_CONFIG["sharpe_weight"],
            drawdown_weight=ACTIVE_CONFIG["drawdown_weight"],
            stability_weight=ACTIVE_CONFIG["stability_weight"],
            # GPU优化参数
            batch_size=ACTIVE_CONFIG["batch_size"],
            early_stop_patience=ACTIVE_CONFIG["early_stop_patience"]
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
            save_checkpoints=ACTIVE_CONFIG["save_checkpoints"],
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=ACTIVE_CONFIG["checkpoint_interval"],
            save_generation_results=ACTIVE_CONFIG["save_generation_results"],
            generation_log_file=generation_log_file,
            generation_log_interval=ACTIVE_CONFIG["generation_log_interval"],
            auto_save_best=ACTIVE_CONFIG["auto_save_best"],
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