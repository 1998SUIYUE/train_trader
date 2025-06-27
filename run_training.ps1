# AI Trading System Launcher (English Version)

Write-Host "=== AI Trading System ===" -ForegroundColor Green
Write-Host "Genetic Algorithm Based Trading Bot Training" -ForegroundColor Yellow

# Check Python environment
Write-Host "`nChecking Python environment..." -ForegroundColor Cyan

$python311Available = $false
$pythonCmd = ""

# Check Python 3.11
try {
    $py311Version = py -3.11 --version 2>&1
    if ($py311Version -match "Python 3\.11") {
        Write-Host "Python 3.11: $py311Version" -ForegroundColor Green
        $python311Available = $true
        $pythonCmd = "py -3.11"
    }
} catch {
    try {
        $direct311Version = & "C:\Python311\python.exe" --version 2>&1
        if ($direct311Version -match "Python 3\.11") {
            Write-Host "Python 3.11: $direct311Version" -ForegroundColor Green
            $python311Available = $true
            $pythonCmd = "C:\Python311\python.exe"
        }
    } catch {
        Write-Host "Python 3.11 not found" -ForegroundColor Red
    }
}

# Check Python 3.13
$python313Available = $false
try {
    $python313Version = python --version 2>&1
    if ($python313Version -match "Python 3\.13") {
        Write-Host "Python 3.13: $python313Version" -ForegroundColor Green
        $python313Available = $true
    }
} catch {
    Write-Host "Python 3.13 not found" -ForegroundColor Yellow
}

# Select training mode
Write-Host "`nSelect training mode:" -ForegroundColor Cyan

if ($python311Available) {
    Write-Host "1. GPU Accelerated Training (Python 3.11 + DirectML) - Recommended" -ForegroundColor Green
}
if ($python313Available) {
    Write-Host "2. CPU Training (Python 3.13) - Good Compatibility" -ForegroundColor Yellow
}
if (-not $python311Available -and -not $python313Available) {
    Write-Host "No suitable Python environment found" -ForegroundColor Red
    Write-Host "Please run: .\setup\install_python311.ps1" -ForegroundColor Yellow
    Read-Host "Press any key to exit"
    exit 1
}

# Set non-interactive defaults
$choice = "1"
$fileChoice = "1" # Default to the first data file
$continueChoice = "Y" # Default to not continuing from checkpoint

# Check data files
Write-Host "`nChecking data files..." -ForegroundColor Cyan

$dataFiles = @()
if (Test-Path "data") {
    $dataFiles = Get-ChildItem "data\*.csv" -ErrorAction SilentlyContinue
}

if ($dataFiles.Count -eq 0) {
    Write-Host "No data files found. Please add data to the 'data' directory." -ForegroundColor Red
    exit 1
}

# Select data file
$selectedFile = $dataFiles[$fileChoice - 1]
Write-Host "Using data file: $($selectedFile.Name)" -ForegroundColor Green

# Configure training parameters
Write-Host "`nConfigure training parameters:" -ForegroundColor Cyan

$mode_name = "" # Initialize mode_name

if ($choice -eq "1" -and $python311Available) {
    # GPU training parameters
    Write-Host "GPU Training Mode - Recommended parameters:" -ForegroundColor Green
    $defaultPopSize = 500
    $defaultGenerations = 300
    $trainScript = "core/main_gpu.py"
    $pythonCommand = $pythonCmd
    $mode_name = "GPU Accelerated"
} else {
    # CPU training parameters
    Write-Host "CPU Training Mode - Recommended parameters:" -ForegroundColor Yellow
    $defaultPopSize = 200
    $defaultGenerations = 100
    $trainScript = "core/main_cpu.py"
    $pythonCommand = "python"
    $mode_name = "CPU"
}

# Set default parameters non-interactively
$popSize = $defaultPopSize
$generations = $defaultGenerations
$normalization = "rolling"

# Check for existing checkpoints
$checkpointDir = "D:/auto_test/train_trader/results/checkpoints"
if (-not (Test-Path $checkpointDir)) {
    New-Item -ItemType Directory -Path $checkpointDir -Force | Out-Null
}
$latestCheckpoint = Get-ChildItem -Path $checkpointDir -Filter "*.pt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

$loadCheckpointPath = ""
if ($latestCheckpoint -and $continueChoice -eq 'y') {
    $loadCheckpointPath = $latestCheckpoint.FullName
}

Write-Host "`n=== Start Training ==="
Write-Host "Training mode: $mode_name"
Write-Host "Data file: $($selectedFile.Name)"
Write-Host "Population size: $popSize"
Write-Host "Generations: $generations"
Write-Host "Normalization: $normalization"
if ($loadCheckpointPath) {
    Write-Host "Loading from checkpoint: $loadCheckpointPath"
}

Write-Host "`nStarting training..."

$command = "$pythonCommand $trainScript --data_file `"$($selectedFile.FullName)`" --population_size $popSize --generations $generations --normalization $normalization"
if ($loadCheckpointPath) {
    $command += " --load_checkpoint `"$($loadCheckpointPath)`""
}

# 添加保存检查点的参数
$command += " --save_checkpoints --checkpoint_interval 10"

Write-Host "Executing command: $command"
Invoke-Expression $command

Write-Host "`nTraining complete."
