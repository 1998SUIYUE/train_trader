# AI Trading System Training Launcher (English Version)

Write-Host "=== AI Trading System ===" -ForegroundColor Green
Write-Host "Genetic Algorithm Based Trading Bot Training" -ForegroundColor Yellow

# Check Python environment
Write-Host "`nChecking Python environment..." -ForegroundColor Cyan

$python311Available = $false
$pythonCmd = ""1

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

$choice = Read-Host "`nPlease select mode (1-2)"

# Check data files
Write-Host "`nChecking data files..." -ForegroundColor Cyan

$dataFiles = @()
if (Test-Path "data") {
    $dataFiles = Get-ChildItem "data\*.csv" -ErrorAction SilentlyContinue
}

if ($dataFiles.Count -eq 0) {
    Write-Host "No data files found" -ForegroundColor Red
    Write-Host "Choose an option:" -ForegroundColor Yellow
    Write-Host "1. Generate sample data" -ForegroundColor White
    Write-Host "2. Manually add data files to data directory" -ForegroundColor White
    
    $dataChoice = Read-Host "Please select (1-2)"
    
    if ($dataChoice -eq "1") {
        Write-Host "`nGenerating sample data..." -ForegroundColor Cyan
        
        # Ensure data directory exists
        if (-not (Test-Path "data")) {
            New-Item -ItemType Directory -Path "data" -Force | Out-Null
        }
        
        # Generate sample data
        if ($python311Available) {
            if ($pythonCmd -eq "py -3.11") {
                py -3.11 examples/sample_data_generator.py --samples 5000 --pattern mixed
            } else {
                & $pythonCmd examples/sample_data_generator.py --samples 5000 --pattern mixed
            }
        } elseif ($python313Available) {
            python examples/sample_data_generator.py --samples 5000 --pattern mixed
        }
        
        # Recheck data files
        $dataFiles = Get-ChildItem "data\*.csv" -ErrorAction SilentlyContinue
        
        if ($dataFiles.Count -eq 0) {
            Write-Host "Sample data generation failed" -ForegroundColor Red
            Read-Host "Press any key to exit"
            exit 1
        }
    } else {
        Write-Host "Please add CSV data files to data directory and run again" -ForegroundColor Yellow
        Read-Host "Press any key to exit"
        exit 1
    }
}

# Select data file
if ($dataFiles.Count -eq 1) {
    $selectedFile = $dataFiles[0]
    Write-Host "Using data file: $($selectedFile.Name)" -ForegroundColor Green
} else {
    Write-Host "Found multiple data files:" -ForegroundColor Yellow
    for ($i = 0; $i -lt $dataFiles.Count; $i++) {
        Write-Host "$($i + 1). $($dataFiles[$i].Name)" -ForegroundColor White
    }
    
    $fileChoice = Read-Host "Please select file (1-$($dataFiles.Count))"
    $selectedFile = $dataFiles[$fileChoice - 1]
    Write-Host "Selected file: $($selectedFile.Name)" -ForegroundColor Green
}

# Configure training parameters
Write-Host "`nConfigure training parameters:" -ForegroundColor Cyan

if ($choice -eq "1" -and $python311Available) {
    # GPU training parameters
    Write-Host "GPU Training Mode - Recommended parameters:" -ForegroundColor Green
    $defaultPopSize = 500
    $defaultGenerations = 300
    $trainScript = "core/main_gpu.py"
    $pythonCommand = $pythonCmd
} else {
    # CPU training parameters
    Write-Host "CPU Training Mode - Recommended parameters:" -ForegroundColor Yellow
    $defaultPopSize = 200
    $defaultGenerations = 100
    $trainScript = "core/main_cpu.py"
    $pythonCommand = "python"
}

Write-Host "Population size (default: $defaultPopSize): " -NoNewline -ForegroundColor White
$popSize = Read-Host
if ([string]::IsNullOrWhiteSpace($popSize)) { $popSize = $defaultPopSize }

Write-Host "Generations (default: $defaultGenerations): " -NoNewline -ForegroundColor White
$generations = Read-Host
if ([string]::IsNullOrWhiteSpace($generations)) { $generations = $defaultGenerations }

Write-Host "Normalization method (relative/rolling/minmax, default: relative): " -NoNewline -ForegroundColor White
$normalization = Read-Host
if ([string]::IsNullOrWhiteSpace($normalization)) { $normalization = "relative" }

# Start training
Write-Host "`n=== Start Training ===" -ForegroundColor Green
$modeText = if ($choice -eq "1") { "GPU Accelerated" } else { "CPU" }
Write-Host "Training mode: $modeText" -ForegroundColor White
Write-Host "Data file: $($selectedFile.Name)" -ForegroundColor White
Write-Host "Population size: $popSize" -ForegroundColor White
Write-Host "Generations: $generations" -ForegroundColor White
Write-Host "Normalization: $normalization" -ForegroundColor White

$confirm = Read-Host "`nConfirm to start training? (y/N)"

if ($confirm -eq "y" -or $confirm -eq "Y") {
    Write-Host "`nStarting training..." -ForegroundColor Green
    
    # Build command
    $dataFilePath = $selectedFile.FullName
    $command = "$pythonCommand $trainScript --data_file `"$dataFilePath`" --population_size $popSize --generations $generations --normalization $normalization"
    
    Write-Host "Executing command: $command" -ForegroundColor Cyan
    
    # Execute training
    try {
        Invoke-Expression $command
        Write-Host "`nTraining completed!" -ForegroundColor Green
        Write-Host "Results saved in results/ directory" -ForegroundColor White
    } catch {
        Write-Host "`nTraining failed: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "Training cancelled" -ForegroundColor Yellow
}

Read-Host "`nPress any key to exit"