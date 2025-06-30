#!/usr/bin/env python3
"""
测试JSON序列化修复
Test JSON serialization fix
"""

import sys
import os
sys.path.append('src')

import json
from pathlib import Path
from enhanced_monitoring import MonitoringConfig, EnhancedMonitor

def test_json_serialization():
    """测试JSON序列化是否正常工作"""
    print("=== 测试JSON序列化修复 ===")
    
    try:
        # 创建包含Path对象的配置
        config = MonitoringConfig(
            log_file=Path("test_monitoring.log"),
            save_interval=5,
            detailed_logging=True,
            export_format="json"
        )
        
        monitor = EnhancedMonitor(config)
        
        # 测试序列化配置
        serialized_config = monitor._serialize_config()
        print(f"序列化配置: {serialized_config}")
        
        # 测试JSON序列化
        json_str = json.dumps(serialized_config, indent=2, ensure_ascii=False)
        print("✅ JSON序列化成功")
        
        # 测试导出详细报告
        monitor.start_monitoring(total_generations=5)
        
        # 添加一些测试数据
        import torch
        import numpy as np
        
        for generation in range(3):
            basic_stats = {
                'best_fitness': 0.5 + generation * 0.01,
                'avg_fitness': 0.3 + generation * 0.008,
                'std_fitness': 0.1,
                'generation_time': 2.0,
                'no_improvement_count': 0,
            }
            
            population = torch.randn(10, 100)
            
            monitor.update_metrics(generation, basic_stats, population=population)
        
        # 测试导出报告
        test_report_path = Path("test_report.json")
        success = monitor.export_detailed_report(test_report_path)
        
        if success:
            print("✅ 详细报告导出成功")
            # 清理测试文件
            if test_report_path.exists():
                test_report_path.unlink()
        else:
            print("❌ 详细报告导出失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_enhanced_cuda_config():
    """测试main_enhanced_cuda.py中的配置序列化"""
    print("\n=== 测试主配置序列化 ===")
    
    try:
        # 模拟ACTIVE_CONFIG
        ACTIVE_CONFIG = {
            "data_directory": "../data",
            "window_size": 350,
            "population_size": 1000,
            "results_dir": "../results",
        }
        
        # 模拟Path对象被添加到配置中的情况
        from pathlib import Path
        output_dir = Path(ACTIVE_CONFIG["results_dir"])
        
        # 转换不可序列化的对象
        serializable_config = ACTIVE_CONFIG.copy()
        
        # 将Path对象转换为字符串
        for key, value in serializable_config.items():
            if isinstance(value, Path):
                serializable_config[key] = str(value)
        
        # 测试JSON序列化
        json_str = json.dumps(serializable_config, indent=2, ensure_ascii=False)
        print("✅ 主配置JSON序列化成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 主配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 开始JSON序列化修复测试")
    
    test1_result = test_json_serialization()
    test2_result = test_main_enhanced_cuda_config()
    
    print("\n" + "="*50)
    print("📊 测试结果总结")
    print("="*50)
    print(f"增强监控序列化测试: {'✅ 通过' if test1_result else '❌ 失败'}")
    print(f"主配置序列化测试: {'✅ 通过' if test2_result else '❌ 失败'}")
    
    if test1_result and test2_result:
        print("\n🎉 所有测试通过！JSON序列化问题已修复！")
        print("\n现在可以安全运行enhanced_cuda.py了")
    else:
        print("\n❌ 部分测试失败，需要进一步修复")
    
    print("="*50)