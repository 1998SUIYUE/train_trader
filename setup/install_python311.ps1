# Python 3.11 安装脚本 (简化版)
# 自动下载并安装Python 3.11，保留现有Python 3.13

Write-Host "=== Python 3.11 安装脚本 ===" -ForegroundColor Green
Write-Host "这将并行安装Python 3.11，不会影响您的Python 3.13" -ForegroundColor Yellow

# 检查管理员权限
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "建议以管理员身份运行以获得最佳体验" -ForegroundColor Yellow
    Write-Host "继续以普通用户身份安装..." -ForegroundColor White
}

# 下载Python 3.11.9
$python311Url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
$installerPath = "$env:TEMP\python-3.11.9-amd64.exe"

Write-Host "`n下载Python 3.11.9..." -ForegroundColor Cyan

try {
    if (-not (Test-Path $installerPath)) {
        Write-Host "正在下载... (约30MB)" -ForegroundColor Yellow
        Invoke-WebRequest -Uri $python311Url -OutFile $installerPath
        Write-Host "下载完成" -ForegroundColor Green
    } else {
        Write-Host "安装文件已存在，跳过下载" -ForegroundColor Green
    }
} catch {
    Write-Host "下载失败: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "请手动下载: https://www.python.org/downloads/release/python-3119/" -ForegroundColor Yellow
    Read-Host "按任意键退出"
    exit 1
}

# 自动安装
Write-Host "`n开始自动安装..." -ForegroundColor Cyan
Write-Host "安装配置:" -ForegroundColor Yellow
Write-Host "- 安装路径: C:\Python311" -ForegroundColor White
Write-Host "- 添加到PATH: 是" -ForegroundColor White
Write-Host "- 包含pip: 是" -ForegroundColor White

try {
    # 静默安装参数
    $installArgs = @(
        "/quiet",
        "InstallAllUsers=0",
        "TargetDir=C:\Python311",
        "AssociateFiles=0",
        "Shortcuts=0",
        "Include_doc=0",
        "Include_debug=0",
        "Include_dev=1",
        "Include_exe=1",
        "Include_launcher=1",
        "Include_lib=1",
        "Include_pip=1",
        "Include_symbols=0",
        "Include_tcltk=1",
        "Include_test=0",
        "Include_tools=0",
        "PrependPath=1",
        "AppendPath=0"
    )
    
    Write-Host "正在安装Python 3.11.9..." -ForegroundColor Yellow
    $process = Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -PassThru
    
    if ($process.ExitCode -eq 0) {
        Write-Host "Python 3.11.9安装成功!" -ForegroundColor Green
        
        # 等待安装完成
        Start-Sleep -Seconds 3
        
        # 验证安装
        Write-Host "`n验证安装..." -ForegroundColor Cyan
        
        $python311Found = $false
        $pythonCmd = ""
        
        # 尝试不同的方式调用Python 3.11
        try {
            $py311Version = py -3.11 --version 2>&1
            if ($py311Version -match "3\.11") {
                Write-Host "py -3.11命令: $py311Version" -ForegroundColor Green
                $python311Found = $true
                $pythonCmd = "py -3.11"
            }
        } catch {
            Write-Host "py启动器不可用" -ForegroundColor Yellow
        }
        
        try {
            $direct311Version = & "C:\Python311\python.exe" --version 2>&1
            if ($direct311Version -match "3\.11") {
                Write-Host "直接路径: $direct311Version" -ForegroundColor Green
                $python311Found = $true
                if ($pythonCmd -eq "") {
                    $pythonCmd = "C:\Python311\python.exe"
                }
            }
        } catch {
            Write-Host "直接路径不可用" -ForegroundColor Yellow
        }
        
        if ($python311Found) {
            Write-Host "`nPython 3.11安装验证成功!" -ForegroundColor Green
            
            Write-Host "`n安装torch-directml..." -ForegroundColor Cyan
            
            try {
                # 升级pip
                Write-Host "升级pip..." -ForegroundColor Yellow
                if ($pythonCmd -eq "py -3.11") {
                    py -3.11 -m pip install --upgrade pip
                } else {
                    & $pythonCmd -m pip install --upgrade pip
                }
                
                # 安装torch-directml
                Write-Host "安装torch-directml..." -ForegroundColor Yellow
                if ($pythonCmd -eq "py -3.11") {
                    py -3.11 -m pip install torch-directml
                } else {
                    & $pythonCmd -m pip install torch-directml
                }
                
                # 安装其他依赖
                Write-Host "安装其他依赖..." -ForegroundColor Yellow
                $packages = @("numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "tqdm", "psutil")
                foreach ($pkg in $packages) {
                    if ($pythonCmd -eq "py -3.11") {
                        py -3.11 -m pip install $pkg
                    } else {
                        & $pythonCmd -m pip install $pkg
                    }
                }
                
                # 测试DirectML
                Write-Host "`n测试DirectML..." -ForegroundColor Cyan
                $testScript = @"
try:
    import torch
    import torch_directml
    device = torch_directml.device()
    print(f'DirectML设备: {device}')
    print('GPU加速环境配置成功!')
except Exception as e:
    print(f'测试失败: {e}')
"@
                
                if ($pythonCmd -eq "py -3.11") {
                    $directmlTest = py -3.11 -c $testScript 2>&1
                } else {
                    $directmlTest = & $pythonCmd -c $testScript 2>&1
                }
                
                Write-Host $directmlTest -ForegroundColor Green
                
                Write-Host "`n安装完成!" -ForegroundColor Green
                Write-Host "=" * 50 -ForegroundColor Green
                Write-Host "Python 3.11 + torch-directml 环境已就绪" -ForegroundColor Green
                Write-Host "=" * 50 -ForegroundColor Green
                
                Write-Host "`n使用说明:" -ForegroundColor Cyan
                Write-Host "1. 使用Python 3.11: $pythonCmd" -ForegroundColor White
                Write-Host "2. 运行训练: $pythonCmd core/main_gpu.py --data_file data/your_data.csv" -ForegroundColor White
                Write-Host "3. 测试环境: $pythonCmd setup/test_environment.ps1" -ForegroundColor White
                
                # 创建便捷启动脚本
                $launchScript = @"
# GPU训练启动脚本
Write-Host "启动GPU加速训练环境" -ForegroundColor Green

# 设置Python 3.11
`$pythonCmd = "$pythonCmd"

# 检查环境
Write-Host "检查环境..." -ForegroundColor Yellow
try {
    if (`$pythonCmd -eq "py -3.11") {
        `$version = py -3.11 --version
        `$directmlCheck = py -3.11 -c "import torch_directml; print('DirectML可用')" 2>&1
    } else {
        `$version = & `$pythonCmd --version
        `$directmlCheck = & `$pythonCmd -c "import torch_directml; print('DirectML可用')" 2>&1
    }
    
    Write-Host "Python版本: `$version" -ForegroundColor White
    Write-Host "DirectML状态: `$directmlCheck" -ForegroundColor White
    
    Write-Host "`n环境检查通过" -ForegroundColor Green
    
    # 检查数据文件
    if (Test-Path "data\*.csv") {
        `$dataFile = Get-ChildItem "data\*.csv" | Select-Object -First 1
        Write-Host "找到数据文件: `$(`$dataFile.Name)" -ForegroundColor Yellow
        
        Write-Host "`n开始训练..." -ForegroundColor Green
        if (`$pythonCmd -eq "py -3.11") {
            py -3.11 core/main_gpu.py --data_file "`$(`$dataFile.FullName)" --population_size 500 --generations 200
        } else {
            & `$pythonCmd core/main_gpu.py --data_file "`$(`$dataFile.FullName)" --population_size 500 --generations 200
        }
    } else {
        Write-Host "`n未找到数据文件" -ForegroundColor Yellow
        Write-Host "请将CSV数据文件放入data目录，或手动指定:" -ForegroundColor White
        Write-Host "`$pythonCmd core/main_gpu.py --data_file your_data.csv" -ForegroundColor Cyan
    }
    
} catch {
    Write-Host "环境检查失败: `$_" -ForegroundColor Red
}

Read-Host "`n按任意键退出"
"@
                
                $launchScript | Out-File -FilePath "start_gpu_training.ps1" -Encoding UTF8
                Write-Host "5. 快速启动: .\start_gpu_training.ps1" -ForegroundColor White
                
                Write-Host "`n提示:" -ForegroundColor Yellow
                Write-Host "- Python 3.13仍然保留，可以继续使用" -ForegroundColor White
                Write-Host "- 使用 '$pythonCmd' 来运行GPU加速版本" -ForegroundColor White
                
            } catch {
                Write-Host "torch-directml安装失败: $($_.Exception.Message)" -ForegroundColor Red
                Write-Host "请手动安装: $pythonCmd -m pip install torch-directml" -ForegroundColor Yellow
            }
            
        } else {
            Write-Host "Python 3.11验证失败" -ForegroundColor Red
            Write-Host "请重新打开PowerShell后再试" -ForegroundColor Yellow
        }
        
    } else {
        Write-Host "安装失败，退出代码: $($process.ExitCode)" -ForegroundColor Red
    }
    
} catch {
    Write-Host "安装过程出错: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "请尝试手动安装或以管理员身份运行" -ForegroundColor Yellow
}

Write-Host "`n完成时间: $(Get-Date)" -ForegroundColor Cyan
Read-Host "`n按任意键退出"