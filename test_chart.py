#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

def test_file_reading():
    log_file = Path("results/training_history.jsonl")
    
    if not log_file.exists():
        print("日志文件不存在")
        return
    
    print(f"文件大小: {log_file.stat().st_size} 字节")
    
    # 尝试不同编码
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(log_file, 'r', encoding=encoding) as f:
                lines = f.readlines()
            
            print(f"✅ {encoding}: 成功读取 {len(lines)} 行")
            
            # 尝试解析第一行JSON
            if lines:
                try:
                    data = json.loads(lines[0].strip())
                    print(f"   JSON解析成功: {list(data.keys())[:5]}")
                    return encoding, lines
                except json.JSONDecodeError as e:
                    print(f"   JSON解析失败: {e}")
            
        except Exception as e:
            print(f"❌ {encoding}: {e}")
    
    return None, []

if __name__ == "__main__":
    encoding, lines = test_file_reading()
    if encoding:
        print(f"\n推荐使用编码: {encoding}")
    else:
        print("\n无法读取文件")