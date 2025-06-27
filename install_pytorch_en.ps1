# Install PyTorch with DirectML for GPU acceleration
# Quick PyTorch installation script

Write-Host "=== Installing PyTorch with DirectML ===" -ForegroundColor Green

# Check Python 3.11
$pythonCmd = ""
try {
    py -3.11 --version | Out-Null
    $pythonCmd = "py -3.11"
    Write-Host "Found Python 3.11" -ForegroundColor Green
} catch {
    try {
        & "C:\Python311\python.exe" --version | Out-Null
        $pythonCmd = "C:\Python311\python.exe"
        Write-Host "Found Python 3.11 (direct path)" -ForegroundColor Green
    } catch {
        Write-Host "Python 3.11 not found!" -ForegroundColor Red
        Read-Host "Press any key to exit"
        exit 1
    }
}

Write-Host "Installing PyTorch with DirectML for GPU acceleration..." -ForegroundColor Cyan
Write-Host "This may take a few minutes..." -ForegroundColor Yellow

try {
    if ($pythonCmd -eq "py -3.11") {
        py -3.11 -m pip install torch-directml
    } else {
        & $pythonCmd -m pip install torch-directml
    }
    
    Write-Host "PyTorch installation completed!" -ForegroundColor Green
    
    # Test the installation
    Write-Host "`nTesting PyTorch installation..." -ForegroundColor Cyan
    
    if ($pythonCmd -eq "py -3.11") {
        $testResult = py -3.11 -c "import torch; import torch_directml; device = torch_directml.device(); print(f'GPU Device: {device}'); print('PyTorch with DirectML is working!')" 2>&1
    } else {
        $testResult = & $pythonCmd -c "import torch; import torch_directml; device = torch_directml.device(); print(f'GPU Device: {device}'); print('PyTorch with DirectML is working!')" 2>&1
    }
    
    if ($testResult -match "PyTorch with DirectML is working") {
        Write-Host $testResult -ForegroundColor Green
        Write-Host "`nSUCCESS: GPU acceleration is ready!" -ForegroundColor Green
        Write-Host "You can now run GPU training with:" -ForegroundColor White
        Write-Host ".\simple_start.ps1" -ForegroundColor Cyan
    } else {
        Write-Host "PyTorch installed but GPU test failed:" -ForegroundColor Yellow
        Write-Host $testResult -ForegroundColor Red
        Write-Host "CPU training will still work" -ForegroundColor White
    }
    
} catch {
    Write-Host "Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "You can try manual installation:" -ForegroundColor Yellow
    Write-Host "$pythonCmd -m pip install torch-directml" -ForegroundColor White
}

Read-Host "`nPress any key to exit"