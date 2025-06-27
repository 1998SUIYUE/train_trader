# 交易AI训练启动脚本 (修复版)

Write-Host "=== 交易AI训练系统 ===" -ForegroundColor Green
Write-Host "基于遗传算法的智能交易员训练" -ForegroundColor Yellow

# 检查Python环境
Write-Host "`n检查Python环境..." -ForegroundColor Cyan

$python311Available = $false
$pythonCmd = ""

# 检查Python 3.11
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
        Write-Host "Python 3.11 未找到" -ForegroundColor Red
    }
}

# 检查Python 3.13
$python313Available = $false
try {
    $python313Version = python --version 2>&1
    if ($python313Version -match "Python 3\.13") {
        Write-Host "Python 3.13: $python313Version" -ForegroundColor Green
        $python313Available = $true
    }
} catch {
    Write-Host "Python 3.13 未找到" -ForegroundColor Yellow
}

# 选择训练模式
Write-Host "`n选择训练模式:" -ForegroundColor Cyan

if ($python311Available) {
    Write-Host "1. GPU加速训练 (Python 3.11 + DirectML) - 推荐" -ForegroundColor Green
}
if ($python313Available) {
    Write-Host "2. CPU训练 (Python 3.13) - 兼容性好" -ForegroundColor Yellow
}
if (-not $python311Available -and -not $python313Available) {
    Write-Host "未找到合适的Python环境" -ForegroundColor Red
    Write-Host "请运行: .\setup\install_python311.ps1" -ForegroundColor Yellow
    Read-Host "按任意键退出"
    exit 1
}

$choice = Read-Host "`n请选择模式 (1-2)"

# 检查数据文件
Write-Host "`n检查数据文件..." -ForegroundColor Cyan

$dataFiles = @()
if (Test-Path "data") {
    $dataFiles = Get-ChildItem "data\*.csv" -ErrorAction SilentlyContinue
}

if ($dataFiles.Count -eq 0) {
    Write-Host "未找到数据文件" -ForegroundColor Red
    Write-Host "选择以下选项:" -ForegroundColor Yellow
    Write-Host "1. 生成示例数据" -ForegroundColor White
    Write-Host "2. 手动添加数据文件到data目录" -ForegroundColor White
    
    $dataChoice = Read-Host "请选择 (1-2)"
    
    if ($dataChoice -eq "1") {
        Write-Host "`n生成示例数据..." -ForegroundColor Cyan
        
        # 确保data目录存在
        if (-not (Test-Path "data")) {
            New-Item -ItemType Directory -Path "data" -Force | Out-Null
        }
        
        # 生成示例数据
        if ($python311Available) {
            if ($pythonCmd -eq "py -3.11") {
                py -3.11 examples/sample_data_generator.py --samples 5000 --pattern mixed
            } else {
                & $pythonCmd examples/sample_data_generator.py --samples 5000 --pattern mixed
            }
        } elseif ($python313Available) {
            python examples/sample_data_generator.py --samples 5000 --pattern mixed
        }
        
        # 重新检查数据文件
        $dataFiles = Get-ChildItem "data\*.csv" -ErrorAction SilentlyContinue
        
        if ($dataFiles.Count -eq 0) {
            Write-Host "示例数据生成失败" -ForegroundColor Red
            Read-Host "按任意键退出"
            exit 1
        }
    } else {
        Write-Host "请将CSV数据文件放入data目录后重新运行" -ForegroundColor Yellow
        Read-Host "按任意键退出"
        exit 1
    }
}

# 选择数据文件
if ($dataFiles.Count -eq 1) {
    $selectedFile = $dataFiles[0]
    Write-Host "使用数据文件: $($selectedFile.Name)" -ForegroundColor Green
} else {
    Write-Host "找到多个数据文件:" -ForegroundColor Yellow
    for ($i = 0; $i -lt $dataFiles.Count; $i++) {
        Write-Host "$($i + 1). $($dataFiles[$i].Name)" -ForegroundColor White
    }
    
    $fileChoice = Read-Host "请选择文件 (1-$($dataFiles.Count))"
    $selectedFile = $dataFiles[$fileChoice - 1]
    Write-Host "选择文件: $($selectedFile.Name)" -ForegroundColor Green
}

# 配置训练参数
Write-Host "`n配置训练参数:" -ForegroundColor Cyan

if ($choice -eq "1" -and $python311Available) {
    # GPU训练参数
    Write-Host "GPU训练模式 - 推荐参数:" -ForegroundColor Green
    $defaultPopSize = 500
    $defaultGenerations = 300
    $trainScript = "core/main_gpu.py"
    $pythonCommand = $pythonCmd
} else {
    # CPU训练参数
    Write-Host "CPU训练模式 - 推荐参数:" -ForegroundColor Yellow
    $defaultPopSize = 200
    $defaultGenerations = 100
    $trainScript = "core/main_cpu.py"
    $pythonCommand = "python"
}

Write-Host "种群大小 (默认: $defaultPopSize): " -NoNewline -ForegroundColor White
$popSize = Read-Host
if ([string]::IsNullOrWhiteSpace($popSize)) { $popSize = $defaultPopSize }

Write-Host "进化代数 (默认: $defaultGenerations): " -NoNewline -ForegroundColor White
$generations = Read-Host
if ([string]::IsNullOrWhiteSpace($generations)) { $generations = $defaultGenerations }

Write-Host "归一化方法 (relative/rolling/minmax, 默认: relative): " -NoNewline -ForegroundColor White
$normalization = Read-Host
if ([string]::IsNullOrWhiteSpace($normalization)) { $normalization = "relative" }

# 开始训练
Write-Host "`n=== 开始训练 ===" -ForegroundColor Green
$modeText = if ($choice -eq "1") { "GPU加速" } else { "CPU" }
Write-Host "训练模式: $modeText" -ForegroundColor White
Write-Host "数据文件: $($selectedFile.Name)" -ForegroundColor White
Write-Host "种群大小: $popSize" -ForegroundColor White
Write-Host "进化代数: $generations" -ForegroundColor White
Write-Host "归一化: $normalization" -ForegroundColor White

$confirm = Read-Host "`n确认开始训练? (y/N)"

if ($confirm -eq "y" -or $confirm -eq "Y") {
    Write-Host "`n启动训练..." -ForegroundColor Green
    
    # 构建命令
    $dataFilePath = $selectedFile.FullName
    $command = "$pythonCommand $trainScript --data_file `"$dataFilePath`" --population_size $popSize --generations $generations --normalization $normalization"
    
    Write-Host "执行命令: $command" -ForegroundColor Cyan
    
    # 执行训练
    try {
        Invoke-Expression $command
        Write-Host "`n训练完成!" -ForegroundColor Green
        Write-Host "结果保存在 results/ 目录" -ForegroundColor White
    } catch {
        Write-Host "`n训练失败: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "训练已取消" -ForegroundColor Yellow
}

Read-Host "`n按任意键退出"