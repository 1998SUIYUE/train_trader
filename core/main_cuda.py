"""
CUDA-accelerated Genetic Algorithm Trading Agent Training
Supports NVIDIA GPU CUDA acceleration
"""

import time
from pathlib import Path
import json
import torch
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cuda_gpu_utils import CudaGPUManager, get_cuda_gpu_manager, check_cuda_compatibility, optimize_cuda_settings
from cuda_accelerated_ga import CudaGPUAcceleratedGA, CudaGAConfig
from data_processor import GPUDataProcessor

# 确保results目录存在
results_dir = Path('../results')
results_dir.mkdir(exist_ok=True)


def main():
    """Main function - CUDA version integrated configuration and automated workflow"""

    # ==============================================================================
    # ======================= 在这里修改你的训练参数 ============================
    # ==============================================================================
    TRAINING_CONFIG = {
        # ==================== 核心训练参数 ====================
        
        # --- 数据配置 ---
        "data_directory": "../data",     # 数据文件目录
        "window_size": 350,             # 特征工程窗口大小
        "normalization": "rolling",     # 归一化方法: 'rolling', 'minmax_local', 'hybrid'
        "batch_size": 1000,             # CUDA上可以使用更大的批次
        
        # --- 遗传算法参数 ---
        "population_size": 1000,         # 种群大小 (CUDA上推荐: 1000-5000)
        "generations": -1,              # 训练代数 (-1=无限训练, 推荐: 100-1000)
        "mutation_rate": 0.01,           # 变异率 (推荐: 0.005-0.02)
        "crossover_rate": 0.8,           # 交叉率 (推荐: 0.7-0.9)
        "elite_ratio": 0.1,              # 精英保留比例 (推荐: 0.05-0.15)
        "early_stop_patience": 100,      # 无改进停止代数 (推荐: 50-200)
        "use_torch_scan": True,          # 使用torch.scan优化回测 (推荐: True)
        
        # --- 交易策略参数 (Sigmoid[0,1]区间) ---
        "buy_threshold": 0.6,            # 买入阈值 (>0.5偏向买入, 推荐: 0.55-0.8)
        "sell_threshold": 0.4,           # 卖出阈值 (<0.5偏向卖出, 推荐: 0.2-0.45)
        
        # --- 风险管理参数 ---
        "stop_loss": 0.05,               # 止损比例 (推荐: 0.02-0.08)
        "max_position": 0.8,             # 最大仓位 (推荐: 0.5-1.0)
        "max_drawdown": 0.2,             # 最大回撤限制 (推荐: 0.1-0.3)
        
        # --- 适应度权重 (总和应为1.0) ---
        "sharpe_weight": 0.5,            # 夏普比率权重
        "drawdown_weight": 0.3,          # 回撤惩罚权重
        "stability_weight": 0.2,         # 交易稳定性权重
        
        # ==================== CUDA专用配置 ====================
        
        # --- GPU设置 ---
        "gpu_device_id": 0,              # GPU设备ID (0为第一个GPU)
        "gpu_memory_fraction": 0.9,      # GPU内存使用比例 (0.0-1.0)
        "mixed_precision": False,        # 是否使用混合精度训练 (实验性)
        
        # ==================== 系统配置 ====================
        
        # --- 保存设置 ---
        "results_dir": "../results",     # 结果输出目录
        "save_checkpoints": True,        # 是否保存检查点
        "checkpoint_interval": 100,      # 检查点保存间隔 (CUDA上可以更长)
        "auto_save_best": True,          # 是否自动保存最佳个体
        
        # --- 日志设置 ---
        "save_generation_results": True, # 是否保存每代结果
        "generation_log_interval": 1,    # 日志记录间隔
    }
    
    # ==============================================================================
    # ======================== 预设配置模板 (可选择使用) =========================
    # ==============================================================================
    
    # 🚀 快速测试配置 (CUDA版)
    QUICK_TEST_CONFIG = {
        **TRAINING_CONFIG,
        "population_size": 200,
        "generations": 20,
        "checkpoint_interval": 10,
        "batch_size": 500,
    }
    
    # 💪 高性能配置 (适合高端NVIDIA GPU)
    HIGH_PERFORMANCE_CONFIG = {
        **TRAINING_CONFIG,
        "population_size": 3000,
        "generations": 500,
        "checkpoint_interval": 50,
        "batch_size": 2000,
        "early_stop_patience": 150,
    }
    
    # 🔥 极限性能配置 (RTX 4090/A100等)
    EXTREME_PERFORMANCE_CONFIG = {
        **TRAINING_CONFIG,
        "population_size": 5000,
        "generations": 1000,
        "checkpoint_interval": 25,
        "batch_size": 3000,
        "early_stop_patience": 200,
        "gpu_memory_fraction": 0.95,
    }
    
    # 🛡️ 保守交易策略 (CUDA版)
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
        "population_size": 1500,         # 更大的种群以提高稳定性
    }
    
    # ⚡ 激进交易策略 (CUDA版)
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
        "population_size": 2000,         # 更大的种群以探索更多策略
    }
    
    # 🔄 长期训练配置 (CUDA版)
    LONG_TERM_CONFIG = {
        **TRAINING_CONFIG,
        "generations": -1,               # 无限训练
        "early_stop_patience": 200,      # 更长的耐心
        "checkpoint_interval": 100,      # 更长的保存间隔
        "population_size": 2000,         # 更大的种群
    }
    
    # 🧪 实验性配置 (使用最新CUDA特性)
    EXPERIMENTAL_CONFIG = {
        **TRAINING_CONFIG,
        "mixed_precision": True,         # 混合精度训练
        "use_torch_scan": True,          # 使用最新的torch.scan
        "population_size": 4000,
        "batch_size": 2500,
        "gpu_memory_fraction": 0.95,
    }
    
    # ==============================================================================
    # =================== 选择要使用的配置 (修改这里) ===========================
    # ==============================================================================
    
    # 选择配置 (取消注释想要使用的配置)
    ACTIVE_CONFIG = TRAINING_CONFIG              # 默认配置
    # ACTIVE_CONFIG = QUICK_TEST_CONFIG          # 快速测试
    # ACTIVE_CONFIG = HIGH_PERFORMANCE_CONFIG    # 高性能
    # ACTIVE_CONFIG = EXTREME_PERFORMANCE_CONFIG # 极限性能
    # ACTIVE_CONFIG = CONSERVATIVE_CONFIG        # 保守策略
    # ACTIVE_CONFIG = AGGRESSIVE_CONFIG          # 激进策略
    # ACTIVE_CONFIG = LONG_TERM_CONFIG           # 长期训练
    # ACTIVE_CONFIG = EXPERIMENTAL_CONFIG        # 实验性配置
    
    # ==============================================================================
    # ======================= 参数修改区域结束 ==================================
    # ==============================================================================

    print("=== CUDA GPU加速遗传算法交易员训练开始 ===")
    
    # --- 1. CUDA环境检查与优化 ---
    print("\n--- CUDA环境检查 ---")
    cuda_info = check_cuda_compatibility()
    for key, value in cuda_info.items():
        if key == 'gpus':
            print(f"可用GPU:")
            for i, gpu in enumerate(value):
                print(f"  GPU {i}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
        elif key not in ['gpus']:
            print(f"{key}: {value}")
    
    if not cuda_info['cuda_available']:
        print("❌ CUDA不可用，请检查CUDA安装")
        return
    
    print("✅ CUDA环境检查通过")
    
    # 优化CUDA设置
    optimize_cuda_settings()
    
    # --- 2. 自动化设置与路径管理 ---
    output_dir = Path(ACTIVE_CONFIG["results_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    data_dir = Path(ACTIVE_CONFIG["data_directory"])
    
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    print("\n--- 训练参数 ---")
    for key, value in ACTIVE_CONFIG.items():
        print(f"{key}: {value}")
    print("--------------------\n")
    
    # --- 3. 自动发现最新的数据文件 ---
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

    # --- 4. 自动发现最新的检查点 ---
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
        # --- 5. 初始化CUDA GPU管理器 ---
        print("初始化CUDA GPU环境...")
        gpu_manager = get_cuda_gpu_manager(device_id=ACTIVE_CONFIG.get("gpu_device_id", 0))
        
        # 设置GPU内存使用限制
        if "gpu_memory_fraction" in ACTIVE_CONFIG:
            gpu_manager.set_memory_fraction(ACTIVE_CONFIG["gpu_memory_fraction"])
        
        print(f"✅ CUDA GPU加速已{'启用' if gpu_manager.device.type == 'cuda' else '禁用'}")
        
        # 显示GPU内存使用情况
        gpu_alloc, gpu_total, sys_used, sys_total = gpu_manager.get_memory_usage()
        print(f"GPU内存: {gpu_alloc:.2f}GB / {gpu_total:.2f}GB")
        print(f"系统内存: {sys_used:.2f}GB / {sys_total:.2f}GB")

        # --- 6. 数据处理 ---
        print("\n开始数据处理...")
        
        # 注意：这里我们需要修改GPUDataProcessor以支持CUDA
        # 暂时使用原有的处理器，但需要确保数据能正确转移到CUDA GPU
        processor = GPUDataProcessor(
            window_size=ACTIVE_CONFIG["window_size"],
            normalization_method=ACTIVE_CONFIG["normalization"],
            gpu_manager=gpu_manager  # 传入CUDA GPU管理器
        )
        
        train_features, train_labels = processor.load_and_process_data(latest_data_file)
        print(f"训练数据形状: {train_features.shape}, 标签数据形状: {train_labels.shape}")

        # --- 7. 配置并初始化CUDA遗传算法 ---
        ga_config = CudaGAConfig(
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
            early_stop_patience=ACTIVE_CONFIG["early_stop_patience"],
            use_torch_scan=ACTIVE_CONFIG["use_torch_scan"]
        )
        print(f"CUDA遗传算法配置: {ga_config}")
        ga = CudaGPUAcceleratedGA(ga_config, gpu_manager)

        # --- 8. 智能加载或初始化种群 ---
        if load_checkpoint_path:
            ga.load_checkpoint(str(load_checkpoint_path))
        else:
            print("初始化新的种群...")
            ga.initialize_population(seed=int(time.time())) # 使用时间戳作为种子

        # --- 9. 开始进化 ---
        print("开始CUDA加速进化过程...")
        
        # 使用固定的日志文件名，所有训练结果都追加到同一个文件
        generation_log_file = output_dir / "training_history_cuda.jsonl"
        print(f"📝 训练日志将写入: {generation_log_file}")
        
        # 启用混合精度训练（实验性）
        if ACTIVE_CONFIG.get("mixed_precision", False):
            print("🧪 启用混合精度训练（实验性功能）")
            # 这里可以添加混合精度训练的代码
        
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

        # --- 10. 保存最终结果 ---
        print("训练完成，正在保存最终结果...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存最佳个体
        best_individual_path = output_dir / f"best_individual_cuda_{timestamp}.npy"
        np.save(best_individual_path, results['best_individual'])
        
        # 保存训练配置
        config_path = output_dir / f"training_config_cuda_{timestamp}.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(ACTIVE_CONFIG, f, indent=2, ensure_ascii=False)

        # --- 11. 输出最终报告 ---
        print("="*70)
        print("              CUDA GPU加速遗传算法训练完成")
        print("="*70)
        print(f"  - 使用GPU:     {gpu_manager.device}")
        if gpu_manager.device.type == 'cuda':
            print(f"  - GPU名称:     {torch.cuda.get_device_name(gpu_manager.device.index)}")
        print(f"  - 最佳适应度:   {results['best_fitness']:.6f}")
        print(f"  - 总训练时间:   {results['total_time']:.2f}秒")
        print(f"  - 最终代数:     {results['final_generation']}")
        print(f"  - 种群大小:     {ACTIVE_CONFIG['population_size']}")
        print(f"  - 最佳个体:     {best_individual_path}")
        print(f"  - 训练配置:     {config_path}")
        print(f"  - 实时日志:     {generation_log_file}")
        print(f"  - 结果目录:     {output_dir}")
        print("="*70)
        
        # 显示最终GPU内存使用情况
        gpu_alloc, gpu_total, sys_used, sys_total = gpu_manager.get_memory_usage()
        print(f"最终GPU内存使用: {gpu_alloc:.2f}GB / {gpu_total:.2f}GB")
        print(f"最终系统内存使用: {sys_used:.2f}GB / {sys_total:.2f}GB")

    except Exception as e:
        print(f"训练过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 清理GPU缓存
        if 'gpu_manager' in locals():
            gpu_manager.clear_cache()
            print("GPU缓存已清理")


if __name__ == "__main__":
    main()