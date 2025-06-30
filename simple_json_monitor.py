#!/usr/bin/env python3
"""
简化的JSON监控器 - 专门用于确保JSON文件生成
Simplified JSON Monitor - Specifically for ensuring JSON file generation
"""

import json
import time
from pathlib import Path
from typing import Dict, Any
import logging

class SimpleJSONMonitor:
    """简化的JSON监控器，专注于可靠的文件生成"""
    
    def __init__(self, log_file_path: Path):
        """
        初始化简化监控器
        
        Args:
            log_file_path: 日志文件路径
        """
        self.log_file = Path(log_file_path)
        self.logger = logging.getLogger(__name__)
        
        # 确保目录存在
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化文件
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """初始化日志文件"""
        try:
            # 创建空文件或验证现有文件
            with open(self.log_file, 'a', encoding='utf-8') as f:
                pass
            print(f"✅ JSON监控器初始化成功: {self.log_file}")
        except Exception as e:
            print(f"❌ JSON监控器初始化失败: {e}")
            # 使用备用路径
            self.log_file = Path.cwd() / "emergency_training_log.jsonl"
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"使用备用路径: {self.log_file}")
    
    def log_generation(self, generation_data: Dict[str, Any]):
        """
        记录一代的数据
        
        Args:
            generation_data: 代数据字典
        """
        try:
            # 添加时间戳
            generation_data['timestamp'] = time.time()
            generation_data['timestamp_str'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # 写入主文件
            success = self._write_to_file(self.log_file, generation_data)
            
            if not success:
                # 尝试备份文件
                backup_file = self.log_file.with_suffix('.jsonl.backup')
                success = self._write_to_file(backup_file, generation_data)
                
                if success:
                    print(f"已写入备份文件: {backup_file}")
                else:
                    # 最后的应急措施
                    emergency_file = self.log_file.parent / "emergency_log.txt"
                    with open(emergency_file, 'a', encoding='utf-8') as f:
                        f.write(f"Gen {generation_data.get('generation', 0)}: {generation_data.get('best_fitness', 0.0):.6f}\n")
                        f.flush()
                    print(f"已写入应急文件: {emergency_file}")
            
        except Exception as e:
            self.logger.error(f"记录代数据失败: {e}")
    
    def _write_to_file(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """
        写入数据到文件
        
        Args:
            file_path: 文件路径
            data: 要写入的数据
            
        Returns:
            是否成功写入
        """
        try:
            with open(file_path, 'a', encoding='utf-8', buffering=1) as f:
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')
                f.flush()
            
            # 验证写入
            if file_path.exists() and file_path.stat().st_size > 0:
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.debug(f"写入文件失败 {file_path}: {e}")
            return False
    
    def get_log_file_info(self) -> Dict[str, Any]:
        """获取日志文件信息"""
        try:
            if self.log_file.exists():
                size = self.log_file.stat().st_size
                
                # 计算行数
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                return {
                    'file_path': str(self.log_file),
                    'exists': True,
                    'size_bytes': size,
                    'line_count': len(lines),
                    'readable': True
                }
            else:
                return {
                    'file_path': str(self.log_file),
                    'exists': False,
                    'size_bytes': 0,
                    'line_count': 0,
                    'readable': False
                }
        except Exception as e:
            return {
                'file_path': str(self.log_file),
                'exists': False,
                'size_bytes': 0,
                'line_count': 0,
                'readable': False,
                'error': str(e)
            }

def test_simple_monitor():
    """测试简化监控器"""
    print("=== 测试简化JSON监控器 ===")
    
    # 创建测试目录
    test_dir = Path("test_simple_monitor")
    test_dir.mkdir(exist_ok=True)
    
    # 初始化监控器
    monitor = SimpleJSONMonitor(test_dir / "simple_training_log.jsonl")
    
    # 模拟5代训练
    for generation in range(5):
        data = {
            'generation': generation,
            'best_fitness': 0.5 + generation * 0.01,
            'avg_fitness': 0.3 + generation * 0.008,
            'generation_time': 2.0,
            'no_improvement_count': max(0, 3 - generation),
        }
        
        monitor.log_generation(data)
        print(f"记录代数 {generation}")
        time.sleep(0.1)
    
    # 检查结果
    info = monitor.get_log_file_info()
    print(f"\n=== 监控器测试结果 ===")
    print(f"文件路径: {info['file_path']}")
    print(f"文件存在: {info['exists']}")
    print(f"文件大小: {info['size_bytes']} 字节")
    print(f"记录行数: {info['line_count']}")
    print(f"可读性: {info['readable']}")
    
    if 'error' in info:
        print(f"错误: {info['error']}")
    
    return info

if __name__ == "__main__":
    test_simple_monitor()