#!/usr/bin/env python3
"""
测试最终修复的脚本
"""

import sys
import os
import subprocess

def test_final_fix():
    """测试最终修复"""
    print("=== 测试最终修复 ===")
    
    # 检查数据文件
    data_file = "XAUUSD_M1_202503142037_202506261819.csv"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    
    print(f"✅ 数据文件存在: {data_file}")
    
    # 运行最小训练测试
    cmd = [
        sys.executable, "core/main_gpu.py",
        "--data_file", data_file,
        "--population_size", "5",
        "--generations", "2", 
        "--window_size", "20"
    ]
    
    print(f"🚀 运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("📊 输出信息:")
        print("=" * 50)
        print(result.stdout)
        print("=" * 50)
        
        if result.stderr:
            print("⚠️ 错误信息:")
            print("=" * 50)
            print(result.stderr)
            print("=" * 50)
        
        if result.returncode == 0:
            print("✅ 训练成功完成!")
            return True
        else:
            print(f"❌ 训练失败，返回码: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 训练超时（300秒）")
        return False
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        return False

if __name__ == "__main__":
    success = test_final_fix()
    if success:
        print("\n🎉 所有修复都成功了！")
    else:
        print("\n💥 仍有问题需要解决")
    
    input("\n按回车键继续...")