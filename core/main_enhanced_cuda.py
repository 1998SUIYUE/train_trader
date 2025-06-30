"""
增强版CUDA-accelerated Genetic Algorithm Trading Agent Training
Enhanced CUDA-accelerated Genetic Algorithm with:
1. Data Annealing
2. Multi-Objective Optimization  
3. Enhanced Monitoring
"""

import time
from pathlib import Path
import json
import torch
import numpy as np
import sys
import os
from numpy._core.multiarray import _reconstruct as numpy_reconstruct
from numpy import ndarray as numpy_ndarray

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cuda_gpu_utils import CudaGPUManager, get_cuda_gpu_manager, check_cuda_compatibility, optimize_cuda_settings
from enhanced_cuda_ga import EnhancedCudaGA, EnhancedGAConfig
from data_annealing_scheduler import AnnealingStrategy
from data_processor import GPUDataProcessor

# 性能分析
try:
    from performance_profiler import get_profiler, start_monitoring, stop_monitoring, print_summary, save_report, timer
    PERFORMANCE_PROFILER_AVAILABLE = True
    print("🔍 性能分析器已启用")
except ImportError:
    PERFORMANCE_PROFILER_AVAILABLE = False
    print("⚠️  性能分析器不可用")
    # 创建空的上下文管理器
    class timer:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    def start_monitoring(*args, **kwargs):
        pass
    def stop_monitoring():
        pass
    def print_summary(*args, **kwargs):
        pass
    def save_report(*args, **kwargs):
        pass

# 确保results目录存在
results_dir = Path('../results')
results_dir.mkdir(exist_ok=True)


def main():
    """Main function - Enhanced CUDA version with advanced features"""
    # Allowlist numpy._core.multiarray._reconstruct and numpy.ndarray for torch.load with weights_only=True
    torch.serialization.add_safe_globals([numpy_reconstruct, numpy_ndarray])

    # ==============================================================================
    # ======================= 增强版训练参数配置 ============================
    # ==============================================================================
    ENHANCED_TRAINING_CONFIG = {
        # ==================== 核心训练参数 ====================
        
        # --- 数据配置 ---
        "data_directory": "../data",     # 数据文件目录
        "window_size": 350,             # 特征工程窗口大小
        "normalization": "rolling",     # 归一化方法: 'rolling', 'minmax_local', 'hybrid'
        "batch_size": 1000,             # CUDA上可以使用更大的批次
        
        # --- 遗传算法参数 ---
        "population_size": 3000,         # 种群大小 (增强版推荐: 2000-5000)
        "generations": -1,              # 训练代数 (-1=无限训练, 推荐: 200-2000)
        "mutation_rate": 0.01,           # 变异率 (推荐: 0.005-0.02)
        "crossover_rate": 0.8,           # 交叉率 (推荐: 0.7-0.9)
        "elite_ratio": 0.05,              # 精英保留比例 (推荐: 0.05-0.15)
        "early_stop_patience": 150,      # 无改进停止代数 (推荐: 100-300)
        "use_torch_scan": True,          # 使用torch.scan优化回测 (推荐: True)
        
        # ==================== 增强功能配置 ====================
        
        # --- 数据退火配置 ---
        "enable_data_annealing": True,           # 启用数据退火
        "annealing_strategy": "progressive",    # 退火策略: 'temporal', 'volatility', 'market_regime', 'feature_complexity', 'progressive'
        "annealing_rate": 0.1,                  # 退火速度 (0.05-0.2)
        "min_data_ratio": 0.3,                  # 最小数据使用比例 (0.2-0.5)
        "max_data_ratio": 1.0,                  # 最大数据使用比例 (通常为1.0)
        "warmup_generations": 50,               # 预热代数 (20-100)
        
        # --- 多目标优化配置 ---
        "enable_multi_objective": True,         # 启用多目标优化
        "pareto_front_size": 100,               # 帕累托前沿大小 (50-200)
        "enable_hypervolume": True,             # 启用超体积计算
        "objective_weights": {                  # 目标权重 (总和应为1.0)
            "sharpe_ratio": 0.25,               # 夏普比率 (最大化)
            "max_drawdown": 0.20,               # 最大回撤 (最小化)
            "total_return": 0.25,               # 总收益率 (最大化)
            "win_rate": 0.15,                   # 胜率 (最大化)
            "volatility": 0.10,                 # 波动率 (最小化)
            "profit_factor": 0.05,              # 盈亏比 (最大化)
        },
        
        # --- 增强监控配置 ---
        "enable_enhanced_monitoring": True,     # 启用增强监控
        "monitoring_save_interval": 10,         # 监控保存间隔
        "detailed_logging": True,               # 详细日志记录
        "track_diversity": True,                # 跟踪种群多样性
        "track_convergence": True,              # 跟踪收敛性
        "export_format": "both",                # 导出格式: 'json', 'csv', 'both'
        
        # ==================== CUDA专用配置 ====================
        
        # --- GPU设置 ---
        "gpu_device_id": 0,              # GPU设备ID (0为第一个GPU)
        "gpu_memory_fraction": 0.9,      # GPU内存使用比例 (0.0-1.0)
        "mixed_precision": False,        # 是否使用混合精度训练 (实验性)
        
        # ==================== 系统配置 ====================
        
        # --- 保存设置 ---
        "results_dir": "../results",     # 结果输出目录
        "save_checkpoints": True,        # 是否保存检查点
        "checkpoint_interval": 50,       # 检查点保存间隔
        "auto_save_best": True,          # 是否自动保存最佳个体
        "save_best_interval": 100,       # 每隔多少代保存最优个体
        
        # --- 日志设置 ---
        "save_generation_results": True, # 是否保存每代结果
        "generation_log_interval": 1,    # 日志记录间隔
    }
    
    # ==============================================================================
    # ======================== 预设配置模板 (增强版) =========================
    # ==============================================================================
    
    # 🚀 快速测试配置 (增强版)
    QUICK_TEST_CONFIG = {
        **ENHANCED_TRAINING_CONFIG,
        "population_size": 500,
        "generations": 30,
        "checkpoint_interval": 10,
        "batch_size": 500,
        "warmup_generations": 5,
        "pareto_front_size": 50,
        "monitoring_save_interval": 5,
    }
    
    # 💪 高性能配置 (适合高端NVIDIA GPU)
    HIGH_PERFORMANCE_CONFIG = {
        **ENHANCED_TRAINING_CONFIG,
        "population_size": 4000,
        "generations": 1000,
        "checkpoint_interval": 100,
        "batch_size": 2000,
        "early_stop_patience": 200,
        "warmup_generations": 100,
        "pareto_front_size": 150,
    }
    
    # 🔥 极限性能配置 (RTX 4090/A100等)
    EXTREME_PERFORMANCE_CONFIG = {
        **ENHANCED_TRAINING_CONFIG,
        "population_size": 6000,
        "generations": 2000,
        "checkpoint_interval": 50,
        "batch_size": 3000,
        "early_stop_patience": 300,
        "gpu_memory_fraction": 0.95,
        "warmup_generations": 150,
        "pareto_front_size": 200,
    }
    
    # 🛡️ 保守交易策略 (增强版) - 注重风险控制
    CONSERVATIVE_CONFIG = {
        **ENHANCED_TRAINING_CONFIG,
        "objective_weights": {
            "sharpe_ratio": 0.30,           # 更重视风险调整收益
            "max_drawdown": 0.35,           # 更重视回撤控制
            "total_return": 0.15,           # 适度重视收益
            "volatility": 0.15,             # 重视波动率控制
            "win_rate": 0.05,               # 适度重视胜率
        },
        "population_size": 2000,
        "annealing_strategy": "volatility", # 从低波动数据开始
        "min_data_ratio": 0.4,              # 使用更多数据
    }
    
    # ⚡ 激进交易策略 (增强版) - 追求高收益
    AGGRESSIVE_CONFIG = {
        **ENHANCED_TRAINING_CONFIG,
        "objective_weights": {
            "total_return": 0.40,           # 重视总收益
            "profit_factor": 0.25,          # 重视盈亏比
            "win_rate": 0.20,               # 重视胜率
            "sharpe_ratio": 0.10,           # 适度重视风险调整
            "max_drawdown": 0.05,           # 较少重视回撤
        },
        "population_size": 3000,
        "annealing_strategy": "market_regime", # 适应不同市场状态
        "early_stop_patience": 100,         # 更激进的早停
    }
    
    # 🔄 长期训练配置 (增强版)
    LONG_TERM_CONFIG = {
        **ENHANCED_TRAINING_CONFIG,
        "generations": -1,               # 无限训练
        "early_stop_patience": 300,      # 更长的耐心
        "checkpoint_interval": 200,      # 更长的保存间隔
        "population_size": 4000,         # 更大的种群
        "warmup_generations": 200,       # 更长的预热期
        "annealing_strategy": "progressive", # 渐进式策略
    }
    
    # 🧪 实验性配置 (使用最新增强特性)
    EXPERIMENTAL_CONFIG = {
        **ENHANCED_TRAINING_CONFIG,
        "mixed_precision": True,         # 混合精度训练
        "annealing_strategy": "feature_complexity", # 特征复杂度退火
        "annealing_rate": 0.05,          # 更慢的退火速度
        "population_size": 5000,
        "batch_size": 2500,
        "gpu_memory_fraction": 0.95,
        "pareto_front_size": 200,
        "track_diversity": True,
        "track_convergence": True,
    }
    
    # 📊 数据退火专项测试配置
    DATA_ANNEALING_TEST_CONFIG = {
        **ENHANCED_TRAINING_CONFIG,
        "population_size": 1000,
        "generations": 100,
        "enable_multi_objective": False,  # 专注于数据退火效果
        "annealing_strategy": "progressive",
        "min_data_ratio": 0.2,
        "warmup_generations": 20,
        "monitoring_save_interval": 5,
    }
    
    # 🎯 多目标优化专项测试配置
    MULTI_OBJECTIVE_TEST_CONFIG = {
        **ENHANCED_TRAINING_CONFIG,
        "population_size": 2000,
        "generations": 200,
        "enable_data_annealing": False,   # 专注于多目标优化效果
        "pareto_front_size": 100,
        "enable_hypervolume": True,
        "objective_weights": {
            "sharpe_ratio": 0.2,
            "max_drawdown": 0.2,
            "total_return": 0.2,
            "win_rate": 0.2,
            "volatility": 0.1,
            "profit_factor": 0.1,
        },
    }
    
    # ==============================================================================
    # =================== 选择要使用的配置 (修改这里) ===========================
    # ==============================================================================
    
    # 选择配置 (取消注释想要使用的配置)
    ACTIVE_CONFIG = ENHANCED_TRAINING_CONFIG     # 默认增强配置
    # ACTIVE_CONFIG = QUICK_TEST_CONFIG          # 快速测试
    # ACTIVE_CONFIG = HIGH_PERFORMANCE_CONFIG    # 高性能
    # ACTIVE_CONFIG = EXTREME_PERFORMANCE_CONFIG # 极限性能
    # ACTIVE_CONFIG = CONSERVATIVE_CONFIG        # 保守策略
    # ACTIVE_CONFIG = AGGRESSIVE_CONFIG          # 激进策略
    # ACTIVE_CONFIG = LONG_TERM_CONFIG           # 长期训练
    # ACTIVE_CONFIG = EXPERIMENTAL_CONFIG        # 实验性配置
    # ACTIVE_CONFIG = DATA_ANNEALING_TEST_CONFIG # 数据退火测试
    # ACTIVE_CONFIG = MULTI_OBJECTIVE_TEST_CONFIG # 多目标优化测试
    
    # ==============================================================================
    # ======================= 参数修改区域结束 ==================================
    # ==============================================================================

    print("=== 增强版CUDA GPU加速遗传算法交易员训练开始 ===")
    
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

    print("\n--- 增强版训练参数 ---")
    for key, value in ACTIVE_CONFIG.items():
        if key == "objective_weights" and isinstance(value, dict):
            print(f"{key}:")
            for obj_name, weight in value.items():
                print(f"  {obj_name}: {weight}")
        else:
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

    # --- 4. 自动发现最新的增强检查点 ---
    load_checkpoint_path = None
    if ACTIVE_CONFIG["save_checkpoints"]:
        checkpoints = sorted(checkpoint_dir.glob("*enhanced*.pt"), key=os.path.getmtime, reverse=True)
        if not checkpoints:
            # 如果没有增强检查点，查找普通检查点
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
        # --- 5. 启动性能监控 ---
        if PERFORMANCE_PROFILER_AVAILABLE:
            start_monitoring(interval=2.0)  # 每2秒记录一次内存使用
            print("🔍 性能监控已启动")
        
        # --- 6. 初始化CUDA GPU管理器 ---
        with timer("gpu_initialization", "setup"):
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

        # --- 7. 数据处理 ---
        with timer("data_processing", "setup"):
            print("\n开始数据处理...")
            
            processor = GPUDataProcessor(
                window_size=ACTIVE_CONFIG["window_size"],
                normalization_method=ACTIVE_CONFIG["normalization"],
                gpu_manager=gpu_manager
            )
            
            train_features, train_labels = processor.load_and_process_data(latest_data_file)
            print(f"训练数据形状: {train_features.shape}, 标签数据形状: {train_labels.shape}")

        # --- 8. 配置并初始化增强版CUDA遗传算法 ---
        with timer("enhanced_ga_initialization", "setup"):
            # 转换退火策略
            annealing_strategy_map = {
                "temporal": AnnealingStrategy.TEMPORAL,
                "volatility": AnnealingStrategy.VOLATILITY,
                "market_regime": AnnealingStrategy.MARKET_REGIME,
                "feature_complexity": AnnealingStrategy.FEATURE_COMPLEXITY,
                "progressive": AnnealingStrategy.PROGRESSIVE,
            }
            
            enhanced_config = EnhancedGAConfig(
                # 基础遗传算法参数
                population_size=ACTIVE_CONFIG["population_size"],
                max_generations=ACTIVE_CONFIG["generations"],
                mutation_rate=ACTIVE_CONFIG["mutation_rate"],
                crossover_rate=ACTIVE_CONFIG["crossover_rate"],
                elite_ratio=ACTIVE_CONFIG["elite_ratio"],
                feature_dim=train_features.shape[1],
                batch_size=ACTIVE_CONFIG["batch_size"],
                early_stop_patience=ACTIVE_CONFIG["early_stop_patience"],
                use_torch_scan=ACTIVE_CONFIG["use_torch_scan"],
                
                # 增强功能参数
                enable_data_annealing=ACTIVE_CONFIG["enable_data_annealing"],
                annealing_strategy=annealing_strategy_map[ACTIVE_CONFIG["annealing_strategy"]],
                annealing_rate=ACTIVE_CONFIG["annealing_rate"],
                min_data_ratio=ACTIVE_CONFIG["min_data_ratio"],
                max_data_ratio=ACTIVE_CONFIG["max_data_ratio"],
                warmup_generations=ACTIVE_CONFIG["warmup_generations"],
                
                enable_multi_objective=ACTIVE_CONFIG["enable_multi_objective"],
                pareto_front_size=ACTIVE_CONFIG["pareto_front_size"],
                enable_hypervolume=ACTIVE_CONFIG["enable_hypervolume"],
                objective_weights=ACTIVE_CONFIG["objective_weights"],
                
                enable_enhanced_monitoring=ACTIVE_CONFIG["enable_enhanced_monitoring"],
                monitoring_save_interval=ACTIVE_CONFIG["monitoring_save_interval"],
                detailed_logging=ACTIVE_CONFIG["detailed_logging"],
                track_diversity=ACTIVE_CONFIG["track_diversity"],
                track_convergence=ACTIVE_CONFIG["track_convergence"],
                export_format=ACTIVE_CONFIG["export_format"],
            )
            
            print(f"增强版CUDA遗传算法配置: {enhanced_config}")
            ga = EnhancedCudaGA(enhanced_config, gpu_manager)

        # --- 9. 智能加载或初始化种群 ---
        if load_checkpoint_path:
            with timer("load_checkpoint", "setup"):
                if "enhanced" in str(load_checkpoint_path):
                    ga.load_checkpoint_enhanced(str(load_checkpoint_path))
                else:
                    ga.load_checkpoint(str(load_checkpoint_path))
        else:
            print("初始化新的种群...")
            ga.initialize_population(seed=int(time.time()))

        # --- 10. 开始增强版进化 ---
        with timer("enhanced_evolution_process", "training"):
            print("开始增强版CUDA加速进化过程...")
            
            # 使用固定的日志文件名
            generation_log_file = output_dir / "enhanced_training_history.jsonl"
            print(f"📝 增强版训练日志将写入: {generation_log_file}")
            
            # 启用混合精度训练（实验性）
            if ACTIVE_CONFIG.get("mixed_precision", False):
                print("🧪 启用混合精度训练（实验性功能）")
            
            results = ga.evolve_enhanced(
                train_features,
                train_labels,
                save_checkpoints=ACTIVE_CONFIG["save_checkpoints"],
                checkpoint_dir=checkpoint_dir,
                
                save_generation_results=ACTIVE_CONFIG["save_generation_results"],
                generation_log_file=generation_log_file,
                generation_log_interval=ACTIVE_CONFIG["generation_log_interval"],
                auto_save_best=ACTIVE_CONFIG["auto_save_best"],
                output_dir=output_dir,
            )

        # --- 11. 保存最终结果 ---
        print("训练完成，正在保存最终结果...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存最佳个体
        best_individual_path = output_dir / f"best_individual_enhanced_{timestamp}.npy"
        np.save(best_individual_path, results['best_individual'])
        
        # 保存训练配置
        config_path = output_dir / f"enhanced_training_config_{timestamp}.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            # 转换不可序列化的对象
            serializable_config = ACTIVE_CONFIG.copy()
            
            # 将Path对象转换为字符串
            for key, value in serializable_config.items():
                if isinstance(value, Path):
                    serializable_config[key] = str(value)
            
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)

        # --- 12. 输出最终报告 ---
        print("="*80)
        print("              增强版CUDA GPU加速遗传算法训练完成")
        print("="*80)
        print(f"  - 使用GPU:           {gpu_manager.device}")
        if gpu_manager.device.type == 'cuda':
            print(f"  - GPU名称:           {torch.cuda.get_device_name(gpu_manager.device.index)}")
        print(f"  - 最佳适应度:         {results['best_fitness']:.6f}")
        print(f"  - 总训练时间:         {results['total_time']:.2f}秒")
        print(f"  - 最终代数:           {results['final_generation']}")
        print(f"  - 种群大小:           {ACTIVE_CONFIG['population_size']}")
        
        # 增强功能报告
        if ACTIVE_CONFIG["enable_data_annealing"]:
            annealing_progress = results.get('final_annealing_progress', {})
            print(f"  - 数据退火策略:       {ACTIVE_CONFIG['annealing_strategy']}")
            print(f"  - 最终数据复杂度:     {annealing_progress.get('complexity_score', 0.0):.3f}")
        
        if ACTIVE_CONFIG["enable_multi_objective"]:
            print(f"  - 多目标优化:         已启用 ({len(ACTIVE_CONFIG['objective_weights'])}个目标)")
            print(f"  - 帕累托前沿大小:     {ACTIVE_CONFIG['pareto_front_size']}")
        
        if ACTIVE_CONFIG["enable_enhanced_monitoring"]:
            training_summary = results.get('training_summary', {})
            print(f"  - 收敛状态:           {'已收敛' if training_summary.get('convergence_achieved', False) else '未收敛'}")
            print(f"  - 平均代数时间:       {training_summary.get('avg_generation_time', 0.0):.2f}秒")
        
        print(f"  - 最佳个体:           {best_individual_path}")
        print(f"  - 训练配置:           {config_path}")
        print(f"  - 实时日志:           {generation_log_file}")
        print(f"  - 结果目录:           {output_dir}")
        
        # 增强报告文件
        if 'detailed_report_path' in results:
            print(f"  - 详细报告:           {results['detailed_report_path']}")
        if 'progress_plot_path' in results:
            print(f"  - 进度图表:           {results['progress_plot_path']}")
        
        print("="*80)
        
        # 显示最终GPU内存使用情况
        gpu_alloc, gpu_total, sys_used, sys_total = gpu_manager.get_memory_usage()
        print(f"最终GPU内存使用: {gpu_alloc:.2f}GB / {gpu_total:.2f}GB")
        print(f"最终系统内存使用: {sys_used:.2f}GB / {sys_total:.2f}GB")
        
        # --- 13. 性能分析报告 ---
        if PERFORMANCE_PROFILER_AVAILABLE:
            stop_monitoring()
            print("\n" + "="*80)
            print("🔍 增强版性能分析报告")
            print("="*80)
            print_summary(detailed=True)
            
            # 保存详细的性能报告
            performance_report_path = output_dir / f"enhanced_performance_report_{timestamp}.json"
            save_report(performance_report_path)
            print(f"📊 详细性能报告已保存: {performance_report_path}")
            print("="*80)

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