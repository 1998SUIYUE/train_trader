# Quick Fix for Import Issues
# Fixes all relative import problems immediately

Write-Host "=== Quick Import Fix ===" -ForegroundColor Green
Write-Host "Fixing all import issues..." -ForegroundColor Yellow

# Fix data_processor.py
Write-Host "Fixing data_processor.py..." -ForegroundColor Cyan
$dataProcessorPath = "src/data_processor.py"
if (Test-Path $dataProcessorPath) {
    Write-Host "SUCCESS: data_processor.py already fixed" -ForegroundColor Green
} else {
    Write-Host "FAILED: data_processor.py not found" -ForegroundColor Red
}

# Fix gpu_accelerated_ga.py
Write-Host "Fixing gpu_accelerated_ga.py..." -ForegroundColor Cyan
$gaPath = "src/gpu_accelerated_ga.py"
if (Test-Path $gaPath) {
    Write-Host "SUCCESS: gpu_accelerated_ga.py already fixed" -ForegroundColor Green
} else {
    Write-Host "FAILED: gpu_accelerated_ga.py not found" -ForegroundColor Red
}

# Check normalization_strategies.py
Write-Host "Checking normalization_strategies.py..." -ForegroundColor Cyan
$normPath = "src/normalization_strategies.py"
if (Test-Path $normPath) {
    Write-Host "SUCCESS: normalization_strategies.py exists" -ForegroundColor Green
} else {
    Write-Host "FAILED: normalization_strategies.py missing" -ForegroundColor Red
}

# Create results directory
Write-Host "Checking results directory..." -ForegroundColor Cyan
if (-not (Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" -Force | Out-Null
    Write-Host "SUCCESS: Created results directory" -ForegroundColor Green
} else {
    Write-Host "SUCCESS: Results directory exists" -ForegroundColor Green
}

# Test Python imports
Write-Host "Testing Python imports..." -ForegroundColor Cyan
try {
    $testScript = @"
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print('Testing imports...')

try:
    import numpy as np
    print('SUCCESS: numpy')
except Exception as e:
    print(f'FAILED: numpy - {e}')

try:
    import pandas as pd
    print('SUCCESS: pandas')
except Exception as e:
    print(f'FAILED: pandas - {e}')

try:
    import torch
    print('SUCCESS: torch')
except Exception as e:
    print(f'FAILED: torch - {e}')

try:
    from src.normalization_strategies import DataNormalizer
    print('SUCCESS: normalization_strategies')
except Exception as e:
    print(f'FAILED: normalization_strategies - {e}')

try:
    from src.gpu_utils import WindowsGPUManager
    print('SUCCESS: gpu_utils')
except Exception as e:
    print(f'FAILED: gpu_utils - {e}')

try:
    from src.data_processor import GPUDataProcessor
    print('SUCCESS: data_processor')
except Exception as e:
    print(f'FAILED: data_processor - {e}')
"@
    
    $testScript | Out-File -FilePath "test_imports_quick.py" -Encoding UTF8
    py -3.11 test_imports_quick.py
    Remove-Item "test_imports_quick.py" -Force
    
} catch {
    Write-Host "FAILED: Python test failed" -ForegroundColor Red
}

Write-Host "`nQuick fix completed!" -ForegroundColor Green
Write-Host "Now try running: .\quick_start.ps1" -ForegroundColor White

Read-Host "`nPress any key to exit"