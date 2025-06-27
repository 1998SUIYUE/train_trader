#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print('=== Testing Fixed Imports ===')

# Test basic packages
try:
    import numpy as np
    print('SUCCESS: numpy imported')
except Exception as e:
    print(f'FAILED: numpy - {e}')

try:
    import pandas as pd
    print('SUCCESS: pandas imported')
except Exception as e:
    print(f'FAILED: pandas - {e}')

try:
    import torch
    print('SUCCESS: torch imported')
except Exception as e:
    print(f'FAILED: torch - {e}')

# Test our modules
try:
    from normalization_strategies import DataNormalizer
    print('SUCCESS: normalization_strategies imported')
except Exception as e:
    print(f'FAILED: normalization_strategies - {e}')

try:
    from gpu_utils import WindowsGPUManager, get_windows_gpu_manager
    print('SUCCESS: gpu_utils imported')
except Exception as e:
    print(f'FAILED: gpu_utils - {e}')

try:
    from data_processor import GPUDataProcessor
    print('SUCCESS: data_processor imported')
except Exception as e:
    print(f'FAILED: data_processor - {e}')

try:
    from gpu_accelerated_ga import WindowsGPUAcceleratedGA, WindowsGAConfig
    print('SUCCESS: gpu_accelerated_ga imported')
except Exception as e:
    print(f'FAILED: gpu_accelerated_ga - {e}')

print('\n=== Import Test Complete ===')

# Test creating instances
print('\n=== Testing Instance Creation ===')

try:
    normalizer = DataNormalizer('relative', 350)
    print('SUCCESS: DataNormalizer created')
except Exception as e:
    print(f'FAILED: DataNormalizer creation - {e}')

try:
    gpu_manager = get_windows_gpu_manager()
    print(f'SUCCESS: WindowsGPUManager created, device: {gpu_manager.device}')
except Exception as e:
    print(f'FAILED: WindowsGPUManager creation - {e}')

try:
    processor = GPUDataProcessor()
    print('SUCCESS: GPUDataProcessor created')
except Exception as e:
    print(f'FAILED: GPUDataProcessor creation - {e}')

print('\n=== All Tests Complete ===')