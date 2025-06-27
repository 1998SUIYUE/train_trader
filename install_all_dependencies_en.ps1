# Install All Dependencies for AI Trading System
# Complete dependency installation script

Write-Host "=== AI Trading System - Complete Dependency Installation ===" -ForegroundColor Green
Write-Host "Installing All Required Libraries" -ForegroundColor Yellow

# Check Python 3.11
$pythonCmd = ""
try {
    py -3.11 --version | Out-Null
    $pythonCmd = "py -3.11"
    $version = py -3.11 --version
    Write-Host "Found Python 3.11: $version" -ForegroundColor Green
} catch {
    try {
        & "C:\Python311\python.exe" --version | Out-Null
        $pythonCmd = "C:\Python311\python.exe"
        $version = & "C:\Python311\python.exe" --version
        Write-Host "Found Python 3.11: $version" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Python 3.11 not found!" -ForegroundColor Red
        Write-Host "Please run: .\setup\install_python311.ps1" -ForegroundColor Yellow
        Read-Host "Press any key to exit"
        exit 1
    }
}

Write-Host "`nStarting dependency installation..." -ForegroundColor Cyan
Write-Host "This may take several minutes, please wait..." -ForegroundColor Yellow

# Upgrade pip
Write-Host "`n1. Upgrading pip..." -ForegroundColor Cyan
try {
    if ($pythonCmd -eq "py -3.11") {
        py -3.11 -m pip install --upgrade pip
    } else {
        & $pythonCmd -m pip install --upgrade pip
    }
    Write-Host "SUCCESS: pip upgraded" -ForegroundColor Green
} catch {
    Write-Host "WARNING: pip upgrade failed, continuing..." -ForegroundColor Yellow
}

# Basic data processing libraries
Write-Host "`n2. Installing basic data processing libraries..." -ForegroundColor Cyan
$basicPackages = @(
    "numpy",
    "pandas", 
    "scikit-learn"
)

foreach ($pkg in $basicPackages) {
    Write-Host "Installing $pkg..." -ForegroundColor White
    try {
        if ($pythonCmd -eq "py -3.11") {
            py -3.11 -m pip install $pkg
        } else {
            & $pythonCmd -m pip install $pkg
        }
        Write-Host "SUCCESS: $pkg installed" -ForegroundColor Green
    } catch {
        Write-Host "FAILED: $pkg installation failed" -ForegroundColor Red
    }
}

# Visualization libraries
Write-Host "`n3. Installing visualization libraries..." -ForegroundColor Cyan
$visualPackages = @(
    "matplotlib",
    "seaborn"
)

foreach ($pkg in $visualPackages) {
    Write-Host "Installing $pkg..." -ForegroundColor White
    try {
        if ($pythonCmd -eq "py -3.11") {
            py -3.11 -m pip install $pkg
        } else {
            & $pythonCmd -m pip install $pkg
        }
        Write-Host "SUCCESS: $pkg installed" -ForegroundColor Green
    } catch {
        Write-Host "FAILED: $pkg installation failed" -ForegroundColor Red
    }
}

# Utility libraries
Write-Host "`n4. Installing utility libraries..." -ForegroundColor Cyan
$utilPackages = @(
    "tqdm",
    "psutil"
)

foreach ($pkg in $utilPackages) {
    Write-Host "Installing $pkg..." -ForegroundColor White
    try {
        if ($pythonCmd -eq "py -3.11") {
            py -3.11 -m pip install $pkg
        } else {
            & $pythonCmd -m pip install $pkg
        }
        Write-Host "SUCCESS: $pkg installed" -ForegroundColor Green
    } catch {
        Write-Host "FAILED: $pkg installation failed" -ForegroundColor Red
    }
}

# PyTorch with DirectML (GPU acceleration)
Write-Host "`n5. Installing PyTorch with DirectML (GPU acceleration)..." -ForegroundColor Cyan
Write-Host "This is the most important library for GPU accelerated training..." -ForegroundColor Yellow

try {
    if ($pythonCmd -eq "py -3.11") {
        py -3.11 -m pip install torch-directml
    } else {
        & $pythonCmd -m pip install torch-directml
    }
    Write-Host "SUCCESS: PyTorch with DirectML installed" -ForegroundColor Green
} catch {
    Write-Host "FAILED: PyTorch with DirectML installation failed" -ForegroundColor Red
    Write-Host "Trying to install basic torch..." -ForegroundColor Yellow
    
    try {
        if ($pythonCmd -eq "py -3.11") {
            py -3.11 -m pip install torch
        } else {
            & $pythonCmd -m pip install torch
        }
        Write-Host "SUCCESS: Basic PyTorch installed" -ForegroundColor Green
    } catch {
        Write-Host "FAILED: PyTorch installation failed" -ForegroundColor Red
    }
}

# Test installations
Write-Host "`n6. Testing installation results..." -ForegroundColor Cyan

# Test basic packages
$testPackages = @("numpy", "pandas", "matplotlib")
foreach ($pkg in $testPackages) {
    try {
        if ($pythonCmd -eq "py -3.11") {
            $result = py -3.11 -c "import $pkg; print('$pkg: OK')" 2>&1
        } else {
            $result = & $pythonCmd -c "import $pkg; print('$pkg: OK')" 2>&1
        }
        
        if ($result -match "OK") {
            Write-Host "SUCCESS: $pkg working" -ForegroundColor Green
        } else {
            Write-Host "FAILED: $pkg test failed" -ForegroundColor Red
        }
    } catch {
        Write-Host "FAILED: $pkg test error" -ForegroundColor Red
    }
}

# Test PyTorch
Write-Host "`nTesting PyTorch..." -ForegroundColor White
try {
    if ($pythonCmd -eq "py -3.11") {
        $torchTest = py -3.11 -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>&1
    } else {
        $torchTest = & $pythonCmd -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>&1
    }
    
    if ($torchTest -match "PyTorch version:") {
        Write-Host "SUCCESS: $torchTest" -ForegroundColor Green
        
        # Test DirectML
        try {
            if ($pythonCmd -eq "py -3.11") {
                $directmlTest = py -3.11 -c "import torch_directml; device = torch_directml.device(); print(f'GPU Device: {device}')" 2>&1
            } else {
                $directmlTest = & $pythonCmd -c "import torch_directml; device = torch_directml.device(); print(f'GPU Device: {device}')" 2>&1
            }
            
            if ($directmlTest -match "GPU Device:") {
                Write-Host "SUCCESS: $directmlTest" -ForegroundColor Green
                Write-Host "GPU acceleration available!" -ForegroundColor Green
            } else {
                Write-Host "WARNING: DirectML not available, will use CPU mode" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "WARNING: DirectML test failed, will use CPU mode" -ForegroundColor Yellow
        }
    } else {
        Write-Host "FAILED: PyTorch test failed: $torchTest" -ForegroundColor Red
    }
} catch {
    Write-Host "FAILED: PyTorch not installed or test failed" -ForegroundColor Red
}

# Installation summary
Write-Host "`n" + "="*60 -ForegroundColor Green
Write-Host "Installation Summary" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green

Write-Host "`nInstalled packages:" -ForegroundColor Cyan
Write-Host "- numpy (data processing)" -ForegroundColor White
Write-Host "- pandas (data analysis)" -ForegroundColor White  
Write-Host "- scikit-learn (machine learning)" -ForegroundColor White
Write-Host "- matplotlib (plotting)" -ForegroundColor White
Write-Host "- seaborn (advanced plotting)" -ForegroundColor White
Write-Host "- tqdm (progress bars)" -ForegroundColor White
Write-Host "- psutil (system monitoring)" -ForegroundColor White
Write-Host "- torch + torch-directml (GPU acceleration)" -ForegroundColor White

Write-Host "`nNow you can run:" -ForegroundColor Cyan
Write-Host "1. .\simple_start.ps1           (quick start)" -ForegroundColor Yellow
Write-Host "2. .\start_training_en.ps1      (full training)" -ForegroundColor Yellow
Write-Host "3. .\quick_start.ps1            (small scale test)" -ForegroundColor Yellow

Write-Host "`nManual commands:" -ForegroundColor Cyan
Write-Host "GPU training: $pythonCmd core/main_gpu.py --data_file XAUUSD_M1_202503142037_202506261819.csv" -ForegroundColor White
Write-Host "CPU training: python core/main_cpu.py --data_file XAUUSD_M1_202503142037_202506261819.csv" -ForegroundColor White

Write-Host "`nIf you encounter import errors, run the fix script" -ForegroundColor Yellow

Read-Host "`nPress any key to exit"