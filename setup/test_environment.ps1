# 环境测试脚本

Write-Host "=== 环境测试 ===" -ForegroundColor Green

# 测试Python 3.11是否可用
Write-Host "`n检查Python 3.11..." -ForegroundColor Cyan

$python311Available = $false
$pythonCommand = ""

# 方法1: 使用py启动器
try {
    $pyVersion = py -3.11 --version 2>&1
    if ($pyVersion -match "Python 3\.11") {
        Write-Host "py -3.11: $pyVersion" -ForegroundColor Green
        $python311Available = $true
        $pythonCommand = "py -3.11"
    }
} catch {
    Write-Host "py -3.11: 不可用" -ForegroundColor Yellow
}

# 方法2: 直接路径
try {
    $directVersion = & "C:\Python311\python.exe" --version 2>&1
    if ($directVersion -match "Python 3\.11") {
        Write-Host "直接路径: $directVersion" -ForegroundColor Green
        $python311Available = $true
        if ($pythonCommand -eq "") {
            $pythonCommand = "C:\Python311\python.exe"
        }
    }
} catch {
    Write-Host "直接路径: 不可用" -ForegroundColor Yellow
}

if ($python311Available) {
    Write-Host "`nPython 3.11 可用!" -ForegroundColor Green
    Write-Host "推荐使用命令: $pythonCommand" -ForegroundColor White
    
    # 测试基础包
    Write-Host "`n测试基础包..." -ForegroundColor Cyan
    
    $packages = @("numpy", "pandas", "torch", "matplotlib")
    foreach ($pkg in $packages) {
        try {
            if ($pythonCommand -eq "py -3.11") {
                $result = py -3.11 -c "import $pkg; print(f'$pkg: OK')" 2>&1
            } else {
                $result = & $pythonCommand -c "import $pkg; print(f'$pkg: OK')" 2>&1
            }
            
            if ($result -match "OK") {
                Write-Host "$pkg: 已安装" -ForegroundColor Green
            } else {
                Write-Host "$pkg: 未安装" -ForegroundColor Red
            }
        } catch {
            Write-Host "$pkg: 测试失败" -ForegroundColor Red
        }
    }
    
    # 测试torch-directml
    Write-Host "`n测试torch-directml..." -ForegroundColor Cyan
    
    try {
        if ($pythonCommand -eq "py -3.11") {
            $torchTest = py -3.11 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>&1
        } else {
            $torchTest = & $pythonCommand -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>&1
        }
        
        if ($torchTest -match "PyTorch:") {
            Write-Host $torchTest -ForegroundColor Green
            
            # 测试DirectML
            if ($pythonCommand -eq "py -3.11") {
                $directmlTest = py -3.11 -c "import torch_directml; device = torch_directml.device(); print(f'DirectML: {device}')" 2>&1
            } else {
                $directmlTest = & $pythonCommand -c "import torch_directml; device = torch_directml.device(); print(f'DirectML: {device}')" 2>&1
            }
            
            if ($directmlTest -match "DirectML:") {
                Write-Host $directmlTest -ForegroundColor Green
                Write-Host "`nGPU环境配置成功!" -ForegroundColor Green
            } else {
                Write-Host "DirectML测试失败: $directmlTest" -ForegroundColor Red
                Write-Host "需要安装: $pythonCommand -m pip install torch-directml" -ForegroundColor Yellow
            }
        } else {
            Write-Host "PyTorch未安装: $torchTest" -ForegroundColor Red
            Write-Host "需要安装: $pythonCommand -m pip install torch torch-directml" -ForegroundColor Yellow
        }
        
    } catch {
        Write-Host "测试失败: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    # 显示使用说明
    Write-Host "`n使用说明:" -ForegroundColor Cyan
    Write-Host "运行训练: $pythonCommand core/main_gpu.py --data_file data/your_data.csv" -ForegroundColor White
    Write-Host "安装包: $pythonCommand -m pip install package_name" -ForegroundColor White
    
} else {
    Write-Host "`nPython 3.11 未找到!" -ForegroundColor Red
    Write-Host "请运行安装脚本: .\setup\install_python311.ps1" -ForegroundColor Yellow
}

# 检查数据目录
Write-Host "`n检查数据目录..." -ForegroundColor Cyan
if (Test-Path "data") {
    $csvFiles = Get-ChildItem "data\*.csv" -ErrorAction SilentlyContinue
    if ($csvFiles) {
        Write-Host "找到数据文件:" -ForegroundColor Green
        foreach ($file in $csvFiles) {
            Write-Host "  - $($file.Name)" -ForegroundColor White
        }
    } else {
        Write-Host "data目录为空，请添加CSV数据文件" -ForegroundColor Yellow
    }
} else {
    Write-Host "data目录不存在，已创建" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "data" -Force | Out-Null
}

# 检查结果目录
if (-not (Test-Path "results")) {
    Write-Host "results目录不存在，已创建" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "results" -Force | Out-Null
}

Write-Host "`n环境测试完成!" -ForegroundColor Green
Read-Host "`n按任意键退出"