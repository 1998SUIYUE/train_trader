# Fix Python Module Import Issues
# Resolves relative import problems

Write-Host "=== Fixing Import Issues ===" -ForegroundColor Green

# Check and fix relative imports in data_processor.py
Write-Host "Checking data_processor.py..." -ForegroundColor Cyan

$dataProcessorPath = "src/data_processor.py"
if (Test-Path $dataProcessorPath) {
    $content = Get-Content $dataProcessorPath -Raw -Encoding UTF8
    
    # Check for relative import issues
    if ($content -match "from \.") {
        Write-Host "Found relative imports, fixing..." -ForegroundColor Yellow
        
        # Fix relative imports
        $content = $content -replace "from \.gpu_utils", "try:`n    from .gpu_utils`nexcept ImportError:`n    from gpu_utils"
        
        # Save fixed file
        $content | Out-File -FilePath $dataProcessorPath -Encoding UTF8
        Write-Host "SUCCESS: data_processor.py fixed" -ForegroundColor Green
    } else {
        Write-Host "SUCCESS: data_processor.py no fixes needed" -ForegroundColor Green
    }
} else {
    Write-Host "FAILED: data_processor.py not found" -ForegroundColor Red
}

# Check results directory
Write-Host "`nChecking results directory..." -ForegroundColor Cyan
if (-not (Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" -Force | Out-Null
    Write-Host "SUCCESS: Created results directory" -ForegroundColor Green
} else {
    Write-Host "SUCCESS: Results directory exists" -ForegroundColor Green
}

# Check Python environment
Write-Host "`nChecking Python environment..." -ForegroundColor Cyan
try {
    $pythonCmd = "py -3.11"
    py -3.11 --version | Out-Null
    Write-Host "SUCCESS: Python 3.11 available" -ForegroundColor Green
    
    # Test imports
    Write-Host "Testing key module imports..." -ForegroundColor White
    
    $testScript = @"
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import numpy as np
    print('SUCCESS: numpy OK')
except ImportError as e:
    print(f'FAILED: numpy - {e}')

try:
    import pandas as pd
    print('SUCCESS: pandas OK')
except ImportError as e:
    print(f'FAILED: pandas - {e}')

try:
    import torch
    print('SUCCESS: torch OK')
except ImportError as e:
    print(f'FAILED: torch - {e}')

try:
    import torch_directml
    print('SUCCESS: torch_directml OK')
except ImportError as e:
    print(f'FAILED: torch_directml - {e}')
"@
    
    $testScript | Out-File -FilePath "test_imports.py" -Encoding UTF8
    py -3.11 test_imports.py
    Remove-Item "test_imports.py" -Force
    
} catch {
    Write-Host "FAILED: Python 3.11 not available" -ForegroundColor Red
}

Write-Host "`nFix completed!" -ForegroundColor Green
Write-Host "You can now try running the training scripts" -ForegroundColor White

Read-Host "`nPress any key to exit"