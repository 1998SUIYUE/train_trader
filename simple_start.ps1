# Simple Start Script - No Chinese Characters

Write-Host "AI Trading System - Simple Start" -ForegroundColor Green

# Check Python 3.11
$pythonCmd = ""
try
{
    py -3.11 --version | Out-Null
    $pythonCmd = "py -3.11"
    Write-Host "Found Python 3.11" -ForegroundColor Green
}
catch
{
    try
    {
        & "C:\Python311\python.exe" --version | Out-Null
        $pythonCmd = "C:\Python311\python.exe"
        Write-Host "Found Python 3.11 (direct path)" -ForegroundColor Green
    }
    catch
    {
        Write-Host "Python 3.11 not found. Please install first." -ForegroundColor Red
        Write-Host "Run: .\setup\install_python311.ps1" -ForegroundColor Yellow
        pause
        exit
    }
}

# Create data directory
if (-not (Test-Path "data"))
{
    New-Item -ItemType Directory -Path "data" -Force | Out-Null
}

# Check for data files
$dataFiles = Get-ChildItem "data\*.csv" -ErrorAction SilentlyContinue
if ($dataFiles.Count -eq 0)
{
    Write-Host "Generating sample data..." -ForegroundColor Yellow
    if ($pythonCmd -eq "py -3.11")
    {
        py -3.11 examples/sample_data_generator.py --samples 3000 --output XAUUSD_M1_202503142037_202506261819.csv
    }
    else
    {
        & $pythonCmd examples/sample_data_generator.py --samples 3000 --output XAUUSD_M1_202503142037_202506261819.csv
    }
    $dataFiles = Get-ChildItem "data\*.csv" -ErrorAction SilentlyContinue
}

if ($dataFiles.Count -eq 0)
{
    Write-Host "Failed to create data file" -ForegroundColor Red
    pause
    exit
}

# Use first data file
$dataFile = $dataFiles[0].FullName
Write-Host "Using data file: $( $dataFiles[0].Name )" -ForegroundColor Green

# Start training
Write-Host "Starting training..." -ForegroundColor Cyan
Write-Host "Parameters: Population=200, Generations=100" -ForegroundColor White

# Try GPU training first
Write-Host "Attempting GPU training..." -ForegroundColor Cyan
if ($pythonCmd -eq "py -3.11")
{
    $gpuResult = py -3.11 core/main_gpu.py --data_file $dataFile --population_size 200 --generations 100 2>&1
}
else
{
    $gpuResult = & $pythonCmd core/main_gpu.py --data_file $dataFile --population_size 200 --generations 100 2>&1
}

if ($LASTEXITCODE -eq 0)
{
    Write-Host "GPU training completed successfully!" -ForegroundColor Green
}
else
{
    Write-Host "GPU training failed, trying CPU..." -ForegroundColor Yellow
    Write-Host "GPU Error: $( $gpuResult | Select-String 'Error|ModuleNotFoundError|ImportError' | Select-Object -First 1 )" -ForegroundColor Red

    # Try CPU training
    $cpuResult = python core/main_cpu.py --data_file $dataFile --population_size 100 --generations 50 2>&1

    if ($LASTEXITCODE -eq 0)
    {
        Write-Host "CPU training completed successfully!" -ForegroundColor Green
    }
    else
    {
        Write-Host "CPU training also failed!" -ForegroundColor Red
        Write-Host "CPU Error: $( $cpuResult | Select-String 'Error|ModuleNotFoundError|ImportError' | Select-Object -First 1 )" -ForegroundColor Red
        Write-Host "`nTo fix GPU training, install PyTorch:" -ForegroundColor Yellow
        Write-Host "py -3.11 -m pip install torch-directml" -ForegroundColor White
    }
}

Write-Host "Results saved in results/ directory" -ForegroundColor White
pause