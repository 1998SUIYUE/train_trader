# CUDA 12.9 Environment Setup Script
# For Windows systems with NVIDIA GPU

Write-Host "=== CUDA 12.9 Environment Setup Script ===" -ForegroundColor Green
Write-Host "This script will install PyTorch and dependencies for CUDA 12.9" -ForegroundColor Yellow

# Check Python version
Write-Host "`nChecking Python version..." -ForegroundColor Cyan
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher first" -ForegroundColor Yellow
    exit 1
}
Write-Host "SUCCESS: $pythonVersion" -ForegroundColor Green

# Check pip
Write-Host "`nChecking pip..." -ForegroundColor Cyan
$pipVersion = pip --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip not installed" -ForegroundColor Red
    exit 1
}
Write-Host "SUCCESS: $pipVersion" -ForegroundColor Green

# Check NVIDIA GPU
Write-Host "`nChecking NVIDIA GPU..." -ForegroundColor Cyan
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: NVIDIA GPU detected:" -ForegroundColor Green
        $gpuInfo | ForEach-Object { Write-Host "   $_" -ForegroundColor White }
    } else {
        Write-Host "WARNING: Cannot detect NVIDIA GPU or nvidia-smi unavailable" -ForegroundColor Yellow
        Write-Host "   Please ensure NVIDIA drivers are installed" -ForegroundColor Yellow
    }
} catch {
    Write-Host "WARNING: Cannot detect GPU information" -ForegroundColor Yellow
}

# Check CUDA version
Write-Host "`nChecking CUDA version..." -ForegroundColor Cyan
try {
    $cudaVersion = nvcc --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: CUDA installed:" -ForegroundColor Green
        $cudaVersion | Select-String "release" | ForEach-Object { Write-Host "   $_" -ForegroundColor White }
    } else {
        Write-Host "WARNING: CUDA not installed or nvcc not in PATH" -ForegroundColor Yellow
        Write-Host "   Recommend installing CUDA 12.1 or higher" -ForegroundColor Yellow
    }
} catch {
    Write-Host "WARNING: Cannot detect CUDA version" -ForegroundColor Yellow
}

# Ask to continue
Write-Host "`nDo you want to continue installing PyTorch and dependencies?" -ForegroundColor Yellow
$continue = Read-Host "Enter 'y' to continue, any other key to exit"
if ($continue -ne 'y' -and $continue -ne 'Y') {
    Write-Host "Installation cancelled" -ForegroundColor Yellow
    exit 0
}

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to upgrade pip" -ForegroundColor Red
    exit 1
}
Write-Host "SUCCESS: pip upgraded" -ForegroundColor Green

# Install PyTorch (CUDA 12.1 version, compatible with CUDA 12.9)
Write-Host "`nInstalling PyTorch (CUDA version)..." -ForegroundColor Cyan
Write-Host "Installing PyTorch for CUDA 12.1 (compatible with CUDA 12.9)..." -ForegroundColor Yellow

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyTorch installation failed" -ForegroundColor Red
    Write-Host "Trying alternative installation method..." -ForegroundColor Yellow
    
    # Alternative method: use conda
    Write-Host "Trying conda installation..." -ForegroundColor Cyan
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: conda installation also failed, please install PyTorch manually" -ForegroundColor Red
        Write-Host "Visit https://pytorch.org/get-started/locally/ for installation instructions" -ForegroundColor Yellow
        exit 1
    }
}
Write-Host "SUCCESS: PyTorch installation completed" -ForegroundColor Green

# Install other dependencies
Write-Host "`nInstalling other dependencies..." -ForegroundColor Cyan
$requirements = @(
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "tqdm>=4.62.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "psutil>=5.8.0"
)

foreach ($package in $requirements) {
    Write-Host "Installing $package..." -ForegroundColor White
    pip install $package
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: $package installation failed, continuing with other packages..." -ForegroundColor Yellow
    }
}

# Optional: Install GPU monitoring tools
Write-Host "`nDo you want to install GPU monitoring tools? (nvidia-ml-py3, gpustat)" -ForegroundColor Yellow
$installGpuTools = Read-Host "Enter 'y' to install, any other key to skip"
if ($installGpuTools -eq 'y' -or $installGpuTools -eq 'Y') {
    Write-Host "Installing GPU monitoring tools..." -ForegroundColor Cyan
    pip install nvidia-ml-py3 gpustat
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: GPU monitoring tools installed" -ForegroundColor Green
    } else {
        Write-Host "WARNING: GPU monitoring tools installation failed" -ForegroundColor Yellow
    }
}

# Verify installation
Write-Host "`nVerifying PyTorch CUDA installation..." -ForegroundColor Cyan
$testScript = @"
import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    
    # Test GPU computation
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("SUCCESS: GPU computation test passed")
    except Exception as e:
        print(f"ERROR: GPU computation test failed: {e}")
else:
    print("ERROR: CUDA not available, please check installation")
"@

$testScript | python
$testResult = $LASTEXITCODE

Write-Host "`n=== Installation Complete ===" -ForegroundColor Green

if ($testResult -eq 0) {
    Write-Host "SUCCESS: Environment setup completed!" -ForegroundColor Green
    Write-Host "`nYou can now run the CUDA training program:" -ForegroundColor Cyan
    Write-Host "   python core/main_cuda.py" -ForegroundColor White
    
    Write-Host "`nOr run tests first:" -ForegroundColor Cyan
    Write-Host "   python src/cuda_gpu_utils.py" -ForegroundColor White
    Write-Host "   python src/cuda_accelerated_ga.py" -ForegroundColor White
} else {
    Write-Host "WARNING: Installation may have issues, please check output above" -ForegroundColor Yellow
}

Write-Host "`nUseful commands:" -ForegroundColor Cyan
Write-Host "   nvidia-smi                    # Check GPU status" -ForegroundColor White
Write-Host "   gpustat                       # Check GPU usage (if installed)" -ForegroundColor White
Write-Host "   python -c 'import torch; print(torch.cuda.is_available())'  # Test CUDA" -ForegroundColor White

Write-Host "`nPress any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")