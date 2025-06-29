# CUDA 12.9环境安装脚本
# 适用于Windows系统和NVIDIA GPU

Write-Host "=== CUDA 12.9 环境安装脚本 ===" -ForegroundColor Green
Write-Host "此脚本将为您安装适用于CUDA 12.9的PyTorch和相关依赖" -ForegroundColor Yellow

# 检查Python版本
Write-Host "`n检查Python版本..." -ForegroundColor Cyan
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python未安装或不在PATH中" -ForegroundColor Red
    Write-Host "请先安装Python 3.8或更高版本" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ $pythonVersion" -ForegroundColor Green

# 检查pip
Write-Host "`n检查pip..." -ForegroundColor Cyan
$pipVersion = pip --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ pip未安装" -ForegroundColor Red
    exit 1
}
Write-Host "✅ $pipVersion" -ForegroundColor Green

# 检查NVIDIA GPU
Write-Host "`n检查NVIDIA GPU..." -ForegroundColor Cyan
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ 检测到NVIDIA GPU:" -ForegroundColor Green
        $gpuInfo | ForEach-Object { Write-Host "   $_" -ForegroundColor White }
    } else {
        Write-Host "⚠️  无法检测到NVIDIA GPU或nvidia-smi不可用" -ForegroundColor Yellow
        Write-Host "   请确保已安装NVIDIA驱动程序" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  无法检测GPU信息" -ForegroundColor Yellow
}

# 检查CUDA版本
Write-Host "`n检查CUDA版本..." -ForegroundColor Cyan
try {
    $cudaVersion = nvcc --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ CUDA已安装:" -ForegroundColor Green
        $cudaVersion | Select-String "release" | ForEach-Object { Write-Host "   $_" -ForegroundColor White }
    } else {
        Write-Host "⚠️  CUDA未安装或nvcc不在PATH中" -ForegroundColor Yellow
        Write-Host "   建议安装CUDA 12.1或更高版本" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  无法检测CUDA版本" -ForegroundColor Yellow
}

# 询问是否继续
Write-Host "`n是否继续安装PyTorch和依赖包？" -ForegroundColor Yellow
$continue = Read-Host "输入 'y' 继续，其他键退出"
if ($continue -ne 'y' -and $continue -ne 'Y') {
    Write-Host "安装已取消" -ForegroundColor Yellow
    exit 0
}

# 升级pip
Write-Host "`n升级pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ pip升级失败" -ForegroundColor Red
    exit 1
}
Write-Host "✅ pip升级完成" -ForegroundColor Green

# 安装PyTorch (CUDA 12.1版本，兼容CUDA 12.9)
Write-Host "`n安装PyTorch (CUDA版本)..." -ForegroundColor Cyan
Write-Host "正在安装适用于CUDA 12.1的PyTorch (兼容CUDA 12.9)..." -ForegroundColor Yellow

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ PyTorch安装失败" -ForegroundColor Red
    Write-Host "尝试使用备用安装方法..." -ForegroundColor Yellow
    
    # 备用方法：使用conda
    Write-Host "尝试使用conda安装..." -ForegroundColor Cyan
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ conda安装也失败，请手动安装PyTorch" -ForegroundColor Red
        Write-Host "访问 https://pytorch.org/get-started/locally/ 获取安装指令" -ForegroundColor Yellow
        exit 1
    }
}
Write-Host "✅ PyTorch安装完成" -ForegroundColor Green

# 安装其他依赖
Write-Host "`n安装其他依赖包..." -ForegroundColor Cyan
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
    Write-Host "安装 $package..." -ForegroundColor White
    pip install $package
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️  $package 安装失败，继续安装其他包..." -ForegroundColor Yellow
    }
}

# 可选：安装GPU监控工具
Write-Host "`n是否安装GPU监控工具？(nvidia-ml-py3, gpustat)" -ForegroundColor Yellow
$installGpuTools = Read-Host "输入 'y' 安装，其他键跳过"
if ($installGpuTools -eq 'y' -or $installGpuTools -eq 'Y') {
    Write-Host "安装GPU监控工具..." -ForegroundColor Cyan
    pip install nvidia-ml-py3 gpustat
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ GPU监控工具安装完成" -ForegroundColor Green
    } else {
        Write-Host "⚠️  GPU监控工具安装失败" -ForegroundColor Yellow
    }
}

# 验证安装
Write-Host "`n验证PyTorch CUDA安装..." -ForegroundColor Cyan
$testScript = @"
import torch
import sys

print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    
    # 测试GPU计算
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU计算测试通过")
    except Exception as e:
        print(f"❌ GPU计算测试失败: {e}")
else:
    print("❌ CUDA不可用，请检查安装")
"@

$testScript | python
$testResult = $LASTEXITCODE

Write-Host "`n=== 安装完成 ===" -ForegroundColor Green

if ($testResult -eq 0) {
    Write-Host "✅ 环境安装成功！" -ForegroundColor Green
    Write-Host "`n现在您可以运行CUDA版本的训练程序：" -ForegroundColor Cyan
    Write-Host "   python core/main_cuda.py" -ForegroundColor White
    
    Write-Host "`n或者先运行测试：" -ForegroundColor Cyan
    Write-Host "   python src/cuda_gpu_utils.py" -ForegroundColor White
    Write-Host "   python src/cuda_accelerated_ga.py" -ForegroundColor White
} else {
    Write-Host "⚠️  安装可能存在问题，请检查上述输出" -ForegroundColor Yellow
}

Write-Host "`n有用的命令：" -ForegroundColor Cyan
Write-Host "   nvidia-smi                    # 查看GPU状态" -ForegroundColor White
Write-Host "   gpustat                       # 查看GPU使用情况（如果已安装）" -ForegroundColor White
Write-Host "   python -c 'import torch; print(torch.cuda.is_available())'  # 测试CUDA" -ForegroundColor White

Write-Host "`n按任意键退出..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")