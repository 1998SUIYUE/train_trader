@echo off
chcp 65001 >nul
echo === AI交易员训练启动 ===

echo.
echo 检查Python环境...

REM 检查Python 3.11
py -3.11 --version >nul 2>&1
if %errorlevel% == 0 (
    echo 找到Python 3.11
    set PYTHON_CMD=py -3.11
    goto :check_data
)

REM 检查直接路径
"C:\Python311\python.exe" --version >nul 2>&1
if %errorlevel% == 0 (
    echo 找到Python 3.11 (直接路径)
    set PYTHON_CMD="C:\Python311\python.exe"
    goto :check_data
)

echo 未找到Python 3.11
echo 请运行: setup\install_python311.ps1
pause
exit /b 1

:check_data
echo.
echo 检查数据文件...

if not exist "data" mkdir data

dir /b "data\*.csv" >nul 2>&1
if %errorlevel% == 0 (
    echo 找到数据文件
    goto :start_training
)

echo 生成示例数据...
%PYTHON_CMD% examples/sample_data_generator.py --samples 3000 --pattern mixed --output XAUUSD_M1_202503142037_202506261819.csv

dir /b "data\*.csv" >nul 2>&1
if %errorlevel% neq 0 (
    echo 数据生成失败
    pause
    exit /b 1
)

:start_training
echo.
echo 开始训练...
echo 参数: 种群200, 代数100

for %%f in (data\*.csv) do (
    set DATA_FILE=%%f
    goto :run_training
)

:run_training
echo 使用数据文件: %DATA_FILE%
echo.

%PYTHON_CMD% core/main_gpu.py --data_file "%DATA_FILE%" --population_size 200 --generations 100

if %errorlevel% neq 0 (
    echo.
    echo GPU训练失败，尝试CPU版本...
    python core/main_cpu.py --data_file "%DATA_FILE%" --population_size 100 --generations 50
)

echo.
echo 训练完成！结果保存在 results/ 目录
pause