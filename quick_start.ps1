# Quick Start Script for AI Trading System

Write-Host "=== AI Trading System Quick Start ===" -ForegroundColor Green

# Check Python 3.11
$python311 = $false
$pythonCmd = ""

try {
    $version = py -3.11 --version 2>&1
    if ($version -match "Python 3\.11") {
        Write-Host "Found Python 3.11: $version" -ForegroundColor Green
        $python311 = $true
        $pythonCmd = "py -3.11"
    }
} catch {
    try {
        $version = & "C:\Python311\python.exe" --version 2>&1
        if ($version -match "Python 3\.11") {
            Write-Host "Found Python 3.11: $version" -ForegroundColor Green
            $python311 = $true
            $pythonCmd = "C:\Python311\python.exe"
        }
    } catch {
        Write-Host "Python 3.11 not found" -ForegroundColor Red
    }
}

if (-not $python311) {
    Write-Host "Need to install Python 3.11 environment" -ForegroundColor Yellow
    Write-Host "Run: .\setup\install_python311.ps1" -ForegroundColor White
    Read-Host "Press any key to exit"
    exit 1
}

# Check data files
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data" -Force | Out-Null
}

$csvFiles = Get-ChildItem "data\*.csv" -ErrorAction SilentlyContinue
if ($csvFiles.Count -eq 0) {
    Write-Host "Generating sample data..." -ForegroundColor Cyan
    if ($pythonCmd -eq "py -3.11") {
        py -3.11 examples/sample_data_generator.py --samples 3000 --pattern mixed --output XAUUSD_M1_202503142037_202506261819.csv
    } else {
        & $pythonCmd examples/sample_data_generator.py --samples 3000 --pattern mixed --output XAUUSD_M1_202503142037_202506261819.csv
    }
    $csvFiles = Get-ChildItem "data\*.csv" -ErrorAction SilentlyContinue
}

if ($csvFiles.Count -eq 0) {
    Write-Host "Failed to generate data files" -ForegroundColor Red
    Read-Host "Press any key to exit"
    exit 1
}

$dataFile = $csvFiles[0]
Write-Host "Using data file: $($dataFile.Name)" -ForegroundColor Green

# Quick training with small parameters
Write-Host "`nStarting quick training (small scale test)..." -ForegroundColor Cyan
Write-Host "Parameters: Population 100, Generations 50" -ForegroundColor White

$command = "$pythonCmd core/main_gpu.py --data_file `"$($dataFile.FullName)`" --population_size 100 --generations 50"

Write-Host "`nExecuting: $command" -ForegroundColor Yellow

try {
    Invoke-Expression $command
    Write-Host "`nTraining completed!" -ForegroundColor Green
    Write-Host "Results saved in results/ directory" -ForegroundColor White
} catch {
    Write-Host "`nTraining failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Trying CPU version..." -ForegroundColor Yellow
    
    $cpuCommand = "python core/main_cpu.py --data_file `"$($dataFile.FullName)`" --population_size 50 --generations 30"
    Write-Host "Executing: $cpuCommand" -ForegroundColor Yellow
    
    try {
        Invoke-Expression $cpuCommand
        Write-Host "`nCPU training completed!" -ForegroundColor Green
    } catch {
        Write-Host "`nCPU training also failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Read-Host "`nPress any key to exit"