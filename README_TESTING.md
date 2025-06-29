# CUDA Testing Guide

This guide helps you test the CUDA environment on your remote computer.

## 🚀 Quick Start (Recommended)

### Option 1: Automated Installation and Testing
```batch
# Run this to install everything and test automatically
install_and_test.bat
```

### Option 2: Manual Step-by-Step Testing
```batch
# 1. Quick CUDA check
python quick_test.py

# 2. Simple training test
python main_cuda_simple.py

# 3. Full test suite (optional)
python test_simple_cuda.py
```

### Option 3: Use Test Runner
```batch
# Run all tests in sequence
run_tests.bat
```

## 📋 Test Files Overview

| File | Purpose | When to Use |
|------|---------|-------------|
| `quick_test.py` | Basic CUDA check | First test to run |
| `main_cuda_simple.py` | Simple training test | Verify training works |
| `test_simple_cuda.py` | Comprehensive CUDA test | Detailed diagnostics |
| `test_cuda_environment.py` | Full environment test | Complete validation |
| `demo_cuda_training.py` | Interactive demo | Learn the system |

## 🔧 Installation Options

### Automatic Installation
```batch
# Windows PowerShell (as Administrator)
powershell -ExecutionPolicy Bypass -File setup/install_cuda129.ps1
```

### Manual Installation
```bash
# Install PyTorch with CUDA 12.1 (compatible with CUDA 12.9)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install numpy pandas matplotlib seaborn tqdm scipy scikit-learn psutil
```

## 📊 Expected Test Results

### Quick Test (`quick_test.py`)
```
=== Quick CUDA Test ===
✓ PyTorch imported successfully: 2.1.0+cu121
✓ CUDA is available: 12.1
✓ GPU count: 1
✓ Basic GPU operation test: [2. 4. 6.]
✓ GPU 0: NVIDIA GeForce RTX 4080 (16.0GB)

🎉 CUDA test PASSED! Your environment is ready.
```

### Simple Training Test (`main_cuda_simple.py`)
```
Simplified CUDA Trading Agent Training
==================================================
Configuration:
  population_size: 100
  max_generations: 50
  crossover_rate: 0.8
  mutation_rate: 0.02
  window_size: 50

=== Environment Check ===
PyTorch version: 2.1.0+cu121
CUDA available: True
CUDA version: 12.1
GPU count: 1
GPU 0: NVIDIA GeForce RTX 4080 (16.0GB)

CUDA is available - using GPU acceleration

=== Creating Demo Data ===
Demo data created: data/demo_data.csv
Data shape: (2000, 4)
Price range: 1823.45 - 2234.67

=== Feature Extraction (window_size=50) ===
Loaded data shape: (2000, 4)
Creating 1951 windows...
Features shape: torch.Size([1951, 200])
Labels shape: torch.Size([1951])
Device: cuda:0

=== Simple Genetic Algorithm ===
Population size: 100
Feature dimension: 200
Max generations: 50
Device: cuda:0

Starting evolution...
Gen   0 | Best:   0.1234 | Avg:   0.0567 | Time: 0.123s
Gen  10 | Best:   0.2345 | Avg:   0.1234 | Time: 0.098s
Gen  20 | Best:   0.3456 | Avg:   0.2345 | Time: 0.087s
...

==================================================
Training Completed Successfully!
==================================================
Best fitness: 0.456789
Total time: 4.56 seconds
Average time per generation: 0.091 seconds
Results saved: results/simple_training_results_20231201_143022.json
Best individual saved: results/best_individual_20231201_143022.npy
==================================================
```

## 🐛 Troubleshooting

### Common Issues and Solutions

**Issue: "CUDA not available"**
```
Solution:
1. Check NVIDIA drivers: nvidia-smi
2. Check CUDA installation: nvcc --version
3. Reinstall PyTorch: pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Issue: "ModuleNotFoundError: No module named 'torch'"**
```
Solution:
1. Install PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
2. Check Python environment: python --version
```

**Issue: "CUDA out of memory"**
```
Solution:
1. Reduce population_size in config
2. Reduce window_size in config
3. Close other GPU applications
```

**Issue: Import errors with complex modules**
```
Solution:
1. Use simplified versions: main_cuda_simple.py instead of core/main_cuda.py
2. Check all dependencies are installed
3. Verify Python path includes src directory
```

## 📈 Performance Expectations

### GPU Performance (RTX 4080 example)
- **Simple Training**: 50 generations in ~5 seconds
- **Population 100**: ~0.1 seconds per generation
- **Population 500**: ~0.5 seconds per generation
- **Population 1000**: ~1.0 seconds per generation

### CPU Fallback Performance
- **Simple Training**: 20 generations in ~30 seconds
- **Population 50**: ~1.5 seconds per generation

## 🎯 Next Steps After Testing

### If All Tests Pass:
1. **Run Full Training**: `python core/main_cuda.py`
2. **Try Different Configurations**: Modify parameters in main_cuda.py
3. **Use Real Data**: Replace demo data with your trading data

### If Some Tests Fail:
1. **Check Error Messages**: Look for specific error details
2. **Try Simplified Version**: Use `main_cuda_simple.py`
3. **Check Dependencies**: Ensure all packages are installed

### Performance Optimization:
1. **Monitor GPU Usage**: `nvidia-smi -l 1`
2. **Adjust Batch Sizes**: Based on your GPU memory
3. **Tune Population Size**: Balance speed vs. quality

## 📞 Getting Help

If you encounter issues:

1. **Check the error messages** in the test output
2. **Run diagnostic commands**:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. **Try the simplified versions** first before the complex ones
4. **Check GPU memory usage** if you get out-of-memory errors

## 📁 File Structure After Testing

```
your_project/
├── data/
│   └── demo_data.csv          # Generated test data
├── results/
│   ├── simple_training_results_*.json
│   └── best_individual_*.npy
├── quick_test.py              # Basic CUDA test
├── main_cuda_simple.py        # Simple training
├── test_simple_cuda.py        # Comprehensive test
└── run_tests.bat             # Test runner
```

---

**Good luck with your testing!** 🚀