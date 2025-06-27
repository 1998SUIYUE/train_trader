# Complete Setup Script for AI Trading System
# Installs all dependencies and fixes import issues

Write-Host "=== AI Trading System - Complete Setup ===" -ForegroundColor Green
Write-Host "This script will install all required dependencies and fix import issues" -ForegroundColor Yellow

# Check Python 3.11
Write-Host "`nStep 1: Checking Python 3.11..." -ForegroundColor Cyan
$pythonCmd = ""
try {
    py -3.11 --version | Out-Null


    $pythonCmd = "py -3.11"
    $version = py -3.11 --version
    Write-Host "SUCCESS: Found Python 3.11: $version" -ForegroundColor Green
} catch {
    try {
        & "C:\Python311\python.exe" --version | Out-Null
        $pythonCmd = "C:\Python311\python.exe"
        $version = & "C:\Python311\python.exe" --version
        Write-Host "SUCCESS: Found Python 3.11: $version" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Python 3.11 not found!" -ForegroundColor Red
        Write-Host "Please install Python 3.11 first by running: .\setup\install_python311.ps1" -ForegroundColor Yellow
        Read-Host "Press any key to exit"
        exit 1
    }
}

# Install packages
Write-Host "`nStep 2: Installing required packages..." -ForegroundColor Cyan
Write-Host "This may take several minutes..." -ForegroundColor Yellow

# All required packages
$allPackages = @(
    "numpy",
    "pandas", 
    "scikit-learn",
    "matplotlib",
    "seaborn",

    "tqdm",
    "psutil",
    "torch-directml"
)

$failedPackages = @()

foreach ($pkg in $allPackages) {
    Write-Host "Installing $pkg..." -ForegroundColor White
    try {
        if ($pythonCmd -eq "py -3.11") {
            py -3.11 -m pip install $pkg --quiet
        } else {
            & $pythonCmd -m pip install $pkg --quiet
        }
        Write-Host "SUCCESS: $pkg installed" -ForegroundColor Green
    } catch {
        Write-Host "FAILED: $pkg installation failed" -ForegroundColor Red
        $failedPackages += $pkg
    }
}

# Fix import issues
Write-Host "`nStep 3: Fixing import issues..." -ForegroundColor Cyan

# Create results directory
if (-not (Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" -Force | Out-Null
    Write-Host "Created results directory" -ForegroundColor Green
}

# Fix relative imports in gpu_accelerated_ga.py if needed
$gaPath = "src/gpu_accelerated_ga.py"
if (Test-Path $gaPath) {
    $content = Get-Content $gaPath -Raw -Encoding UTF8
    if ($content -notmatch "try:\s*from \.gpu_utils") {
        Write-Host "Fixing imports in gpu_accelerated_ga.py..." -ForegroundColor Yellow
        $content = $content -replace "from \.gpu_utils import", "try:`n    from .gpu_utils import`nexcept ImportError:`n    from gpu_utils import"
        $content | Out-File -FilePath $gaPath -Encoding UTF8
        Write-Host "Import fixes applied" -ForegroundColor Green
    }
}

# Test installations
Write-Host "`nStep 4: Testing installations..." -ForegroundColor Cyan

$testResults = @{}
$testPackages = @("numpy", "pandas", "torch", "torch_directml")

foreach ($pkg in $testPackages) {
    try {
        if ($pythonCmd -eq "py -3.11") {
            $result = py -3.11 -c "import $pkg; print('OK')" 2>&1
        } else {
            $result = & $pythonCmd -c "import $pkg; print('OK')" 2>&1
        }
        
        if ($result -match "OK") {
            Write-Host "SUCCESS: $pkg working" -ForegroundColor Green
            $testResults[$pkg] = $true
        } else {
            Write-Host "FAILED: $pkg test failed" -ForegroundColor Red
            $testResults[$pkg] = $false
        }
    } catch {
        Write-Host "FAILED: $pkg test error" -ForegroundColor Red
        $testResults[$pkg] = $false
    }
}

# Test GPU acceleration
if ($testResults["torch"] -and $testResults["torch_directml"]) {
    Write-Host "`nTesting GPU acceleration..." -ForegroundColor White
    try {
        if ($pythonCmd -eq "py -3.11") {
            $gpuTest = py -3.11 -c "import torch_directml; device = torch_directml.device(); print(f'GPU Device: {device}')" 2>&1
        } else {
            $gpuTest = & $pythonCmd -c "import torch_directml; device = torch_directml.device(); print(f'GPU Device: {device}')" 2>&1
        }
        
        if ($gpuTest -match "GPU Device:") {
            Write-Host "SUCCESS: $gpuTest" -ForegroundColor Green
            Write-Host "GPU acceleration is available!" -ForegroundColor Green
        } else {
            Write-Host "WARNING: GPU acceleration not available, will use CPU mode" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "WARNING: GPU test failed, will use CPU mode" -ForegroundColor Yellow
    }
}

# Summary
Write-Host "`n" + "="*60 -ForegroundColor Green
Write-Host "Setup Complete - Summary" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green

if ($failedPackages.Count -eq 0) {
    Write-Host "SUCCESS: All packages installed successfully!" -ForegroundColor Green
} else {
    Write-Host "WARNING: Some packages failed to install:" -ForegroundColor Yellow
    foreach ($pkg in $failedPackages) {
        Write-Host "  - $pkg" -ForegroundColor Red
    }
}

Write-Host "`nWhat you can do now:" -ForegroundColor Cyan
Write-Host "1. Quick start:     .\simple_start.ps1" -ForegroundColor Yellow
Write-Host "2. Full training:   .\start_training_en.ps1" -ForegroundColor Yellow
Write-Host "3. Small test:      .\quick_start.ps1" -ForegroundColor Yellow

if ($testResults["numpy"] -and $testResults["pandas"]) {
    Write-Host "`nData generation: READY" -ForegroundColor Green
} else {
    Write-Host "`nData generation: NOT READY (missing numpy/pandas)" -ForegroundColor Red
}

if ($testResults["torch"]) {
    Write-Host "GPU training: READY" -ForegroundColor Green
} else {
    Write-Host "GPU training: NOT READY (missing PyTorch)" -ForegroundColor Yellow
}

Write-Host "CPU training: READY (uses built-in algorithms)" -ForegroundColor Green

Write-Host "`nManual commands:" -ForegroundColor Cyan
Write-Host "Generate data: $pythonCmd examples/sample_data_generator.py --samples 3000" -ForegroundColor White
Write-Host "GPU training:  $pythonCmd core/main_gpu.py --data_file XAUUSD_M1_202503142037_202506261819.csv" -ForegroundColor White
Write-Host "CPU training:  python core/main_cpu.py --data_file XAUUSD_M1_202503142037_202506261819.csv" -ForegroundColor White

Read-Host "`nPress any key to exit"