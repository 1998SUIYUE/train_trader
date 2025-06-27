# AI Trading System - Setup Guide

## Quick Setup (Recommended)

Run the complete setup script that installs everything:

```powershell
.\setup_complete.ps1
```

This will:
- Check Python 3.11 installation
- Install all required packages
- Fix import issues
- Test the installation

## Alternative Setup Options

### Option 1: Install All Dependencies
```powershell
.\install_all_dependencies_en.ps1
```

### Option 2: Install Only Basic Packages
```powershell
.\install_basic_en.ps1
```

### Option 3: Install Only PyTorch
```powershell
.\install_pytorch_en.ps1
```

### Option 4: Fix Import Issues Only
```powershell
.\fix_imports_en.ps1
```

## Running the Training

After setup, you can run training with:

### English Interface (Recommended)
```powershell
.\start_training_en.ps1
```

### Simple Start
```powershell
.\simple_start.ps1
```

### Quick Test
```powershell
.\quick_start.ps1
```

## Manual Commands

### Generate Sample Data
```powershell
py -3.11 examples/sample_data_generator.py --samples 3000 --output XAUUSD_M1_202503142037_202506261819.csv
```

### GPU Training
```powershell
py -3.11 core/main_gpu.py --data_file XAUUSD_M1_202503142037_202506261819.csv --population_size 200 --generations 100
```

### CPU Training
```powershell
python core/main_cpu.py --data_file XAUUSD_M1_202503142037_202506261819.csv --population_size 200 --generations 100
```

## Required Packages

- **numpy** - Data processing
- **pandas** - Data analysis  
- **torch** - Deep learning framework
- **torch-directml** - GPU acceleration for AMD/Intel GPUs
- **scikit-learn** - Machine learning utilities
- **matplotlib** - Plotting
- **seaborn** - Advanced plotting
- **tqdm** - Progress bars
- **psutil** - System monitoring

## Troubleshooting

### Python 3.11 Not Found
Install Python 3.11 first:
```powershell
.\setup\install_python311.ps1
```

### Import Errors
Run the import fix script:
```powershell
.\fix_imports_en.ps1
```

### Package Installation Fails
Try installing packages manually:
```powershell
py -3.11 -m pip install numpy pandas torch-directml
```

### GPU Not Working
GPU training requires PyTorch with DirectML. If it fails, CPU training will still work.

## File Structure

```
├── setup_complete.ps1              # Complete setup (recommended)
├── install_all_dependencies_en.ps1 # Install all packages
├── install_basic_en.ps1            # Install basic packages only
├── install_pytorch_en.ps1          # Install PyTorch only
├── fix_imports_en.ps1              # Fix import issues
├── start_training_en.ps1           # Main training script (English)
├── simple_start.ps1                # Simple start script
└── quick_start.ps1                 # Quick test script
```

## Notes

- All PowerShell scripts use English only to avoid encoding issues
- GPU acceleration works with AMD, Intel, and NVIDIA GPUs through DirectML
- CPU training is always available as fallback
- Sample data is automatically generated if no data files exist