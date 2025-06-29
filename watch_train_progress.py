#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的训练进度监控启动器
直接运行即可开始监控，无需任何参数
"""

import os
import sys
from pathlib import Path

def main():
    """主函数 - 启动训练进度监控"""
    
    print("🚀 启动训练进度监控...")
    print("=" * 50)
    
    # 检查是否存在tools目录中的监控工具
    tools_watcher = Path("tools/watch_training_progress.py")
    current_watcher = Path("view_cuda_progress.py")
    quick_monitor = Path("quick_monitor.py")
    
    # 优先使用最强大的监控工具
    if tools_watcher.exists():
        print("🎯 使用完整版监控工具...")
        os.system(f"python {tools_watcher}")
    elif current_watcher.exists():
        print("🎯 使用CUDA进度查看器...")
        os.system(f"python {current_watcher} --watch")
    elif quick_monitor.exists():
        print("🎯 使用快速监控器...")
        os.system(f"python {quick_monitor}")
    else:
        print("❌ 未找到监控工具")
        print("请确保以下文件之一存在:")
        print("  - tools/watch_training_progress.py")
        print("  - view_cuda_progress.py")
        print("  - quick_monitor.py")
        return
    
    print("\n👋 监控已结束")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 用户中断，监控已停止")
    except Exception as e:
        print(f"\n❌ 出现错误: {e}")
        input("按回车键退出...")