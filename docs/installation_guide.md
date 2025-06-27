# 安装指南

## 🎯 系统要求

### 硬件要求
- **操作系统**: Windows 11/10, Linux, macOS
- **内存**: 16GB+ 系统内存
- **存储**: 10GB+ 可用空间
- **GPU**: NVIDIA RTX 4060 (推荐) 或其他CUDA兼容显卡

### 软件要求
- **Python**: 3.8+ (推荐 3.11)
- **CUDA**: 11.8 或 12.1 (NVIDIA显卡)
- **网络连接**: 下载依赖包

## 🚀 安装方法

### 方法一：一键安装（推荐）

#### Windows 用户
```bash
# 运行一键安装脚本
setup/install_rtx4060.bat
```

#### Linux/Mac 用户
```bash
# 给脚本执行权限
chmod +x setup/install_rtx4060.sh
# 运行安装脚本
./setup/install_rtx4060.sh
```

#### Python 智能安装
```bash
# 自动检测环境并安装
python setup/install_dependencies.py
```

### 方法二：使用requirements文件

#### RTX 4060专用 (CUDA版本)
```bash
pip install -r requirements_rtx4060.txt
```

#### 通用版本 (CPU版本)
```bash
pip install -r requirements.txt
```

### 方法三：手动安装

#### 1. 系统安装
```powershell
# 运行自动安装脚本
.\setup\install_python311.ps1
```

这个脚本会：
- ✅ 自动下载Python 3.11.9
- ✅ 静默安装到C:\Python311
- ✅ 自动安装torch-directml
- ✅ 安装所有必要依赖
- ✅ 测试GPU环境
- ✅ 创建启动脚本

#### 2. 验证安装
```powershell
# 测试环境
.\setup\test_environment.ps1
```

### 方法二：手动安装

#### 1. 下载Python 3.11
1. 访问：https://www.python.org/downloads/release/python-3119/
2. 下载：`python-3.11.9-amd64.exe`

#### 2. 安装Python 3.11
1. **右键点击**安装文件，选择**"以管理员身份运行"**
2. **重要设置**：
   - ✅ 勾选 "Add Python 3.11 to PATH"
   - ✅ 点击 "Customize installation"
   - ✅ 安装路径设置为: `C:\Python311`

#### 3. 安装依赖包
```powershell
# 升级pip
py -3.11 -m pip install --upgrade pip

# 安装torch-directml
py -3.11 -m pip install torch-directml

# 安装其他依赖
py -3.11 -m pip install numpy pandas scikit-learn matplotlib seaborn tqdm psutil
```

#### 4. 测试安装
```powershell
# 测试torch-directml
py -3.11 -c "import torch_directml; print(torch_directml.device())"
```

## 🔄 版本管理

### 查看Python版本
```powershell
# 查看默认Python
python --version

# 查看Python 3.11
py -3.11 --version

# 查看Python 3.13
py -3.13 --version
```

### 使用特定版本
```powershell
# 运行训练程序
py -3.11 core/main_gpu.py --data_file data/your_data.csv
```

## 🛠️ 故障排除

### 问题1：py命令不识别
**解决方案**：
```powershell
# 使用直接路径
C:\Python311\python.exe --version
```

### 问题2：torch-directml安装失败
**可能原因**：
- 网络问题
- Python版本不对
- pip版本过旧

**解决方案**：
```powershell
# 确认Python版本
py -3.11 --version

# 升级pip
py -3.11 -m pip install --upgrade pip

# 使用国内镜像
py -3.11 -m pip install torch-directml -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 问题3：DirectML设备不可用
**可能原因**：
- AMD GPU驱动过旧
- DirectX 12不支持

**解决方案**：
1. 更新AMD显卡驱动
2. 确保Windows支持DirectX 12
3. 确保安装了torch-directml

### 问题4：权限不足
**解决方案**：
```powershell
# 以管理员身份运行PowerShell
# 右键点击PowerShell -> "以管理员身份运行"
```

## 📊 环境验证

### 完整测试
```powershell
# 运行完整环境测试
.\setup\test_environment.ps1
```

### 手动验证
```powershell
# 1. 检查Python版本
py -3.11 --version

# 2. 测试基础包
py -3.11 -c "import numpy, pandas, torch; print('基础包OK')"

# 3. 测试DirectML
py -3.11 -c "import torch_directml; print(f'GPU: {torch_directml.device()}')"

# 4. 简单性能测试
py -3.11 -c "
import torch
import torch_directml
import time

device = torch_directml.device()
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)

start = time.time()
z = torch.mm(x, y)
end = time.time()

print(f'GPU矩阵运算时间: {end-start:.4f}秒')
"
```

## ✅ 安装检查清单

- [ ] Python 3.11.9安装成功
- [ ] `py -3.11 --version`显示正确版本
- [ ] torch-directml安装成功
- [ ] DirectML设备可用
- [ ] 基础依赖包安装完成
- [ ] 环境测试通过
- [ ] 数据目录已创建
- [ ] 结果目录已创建

## 💡 最佳实践

### 1. 项目隔离
```powershell
# 为项目创建专用启动脚本
# gpu_training.bat
@echo off
py -3.11 core/main_gpu.py --data_file %1
pause
```

### 2. 虚拟环境（可选）
```powershell
# 创建虚拟环境
py -3.11 -m venv trading_env

# 激活虚拟环境
trading_env\Scripts\Activate.ps1

# 在虚拟环境中安装包
pip install torch-directml numpy pandas
```

### 3. 配置IDE
- **VS Code**: 设置Python解释器为`C:\Python311\python.exe`
- **PyCharm**: 添加Python 3.11解释器

## 🔧 高级配置

### 环境变量设置
```powershell
# 设置Python路径优先级
$env:PATH = "C:\Python311;C:\Python311\Scripts;" + $env:PATH
```

### 性能优化
```powershell
# 设置PyTorch线程数
$env:OMP_NUM_THREADS = "4"

# 设置DirectML内存管理
$env:DIRECTML_MEMORY_BUDGET = "4096"
```

---

**安装完成后，您就可以开始训练AI交易员了！** 🎉