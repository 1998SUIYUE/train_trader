# Fix Summary: GPUDataProcessor Parameter Mismatch

## Problem
The original error was:
```
TypeError: GPUDataProcessor.__init__() got an unexpected keyword argument 'normalization'
```

## Root Cause Analysis
1. **Parameter name mismatch**: In `core/main_gpu.py`, the code was passing `normalization=args.normalization` but `GPUDataProcessor.__init__()` expected `normalization_method`.

2. **Missing method**: The `GPUDataProcessor` class didn't have a `load_and_process_data` method that was being called in `main_gpu.py`.

3. **Missing property**: The `DataNormalizer` class was missing a `feature_dim` property that was being used in `GPUDataProcessor`.

## Additional Issue Found
After the initial fixes, a new error was discovered:
```
ValueError: Must provide matching length window_shape and axis; got 2 window_shape elements and 1 axes elements.
```

This was caused by incorrect usage of `np.lib.stride_tricks.sliding_window_view` with a 2D window shape and 1D axis specification.

## Fixes Applied

### Fix 1: Parameter Name Correction
**File**: `core/main_gpu.py` (line 110)
**Change**: 
```python
# Before
processor = GPUDataProcessor(
    window_size=args.window_size,
    normalization=args.normalization,  # ❌ Wrong parameter name
    gpu_manager=gpu_manager
)

# After
processor = GPUDataProcessor(
    window_size=args.window_size,
    normalization_method=args.normalization,  # ✅ Correct parameter name
    gpu_manager=gpu_manager
)
```

### Fix 2: Added Missing Method
**File**: `src/data_processor.py` (lines 389-417)
**Added**: `load_and_process_data` method to `GPUDataProcessor` class
```python
def load_and_process_data(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    加载并处理数据，返回特征和标签
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        (特征矩阵, 标签向量)
    """
    # 加载原始数据
    self.load_data(file_path)
    
    # 提取特征
    features, prices = self.extract_features_gpu()
    
    # 计算标签（下一期收益率）
    price_array = self.gpu_manager.to_cpu(prices).numpy()
    returns = np.zeros(len(price_array))
    returns[1:] = (price_array[1:] - price_array[:-1]) / price_array[:-1]
    
    # 移除最后一个特征样本（因为没有对应的下一期收益率）
    features = features[:-1]
    labels = self.gpu_manager.to_gpu(returns[1:])  # 对应的收益率标签
    
    logger.info(f"最终特征形状: {features.shape}")
    logger.info(f"最终标签形状: {labels.shape}")
    
    return features, labels
```

### Fix 3: Added Missing Property
**File**: `src/normalization_strategies.py` (lines 22-30)
**Added**: `feature_dim` property to `DataNormalizer` class
```python
@property
def feature_dim(self) -> int:
    """Calculate feature dimension based on method and window size"""
    if self.method == 'hybrid':
        # OHLC relative prices + RSI + MA ratio + volatility
        return self.window_size * 4 + 1 + 1 + self.window_size
    else:
        # Standard OHLC features
        return self.window_size * 4
```

### Fix 4: Sliding Window Implementation
**File**: `src/data_processor.py` (lines 207-216)
**Fixed**: Replaced problematic `sliding_window_view` call with manual window creation
```python
# Before (problematic)
windows = np.lib.stride_tricks.sliding_window_view(
    ohlc_data, window_shape=(self.window_size, 4), axis=0
)[start_idx:end_idx]

# After (fixed)
batch_windows = []
for i in range(start_idx, end_idx):
    window_start = i
    window_end = i + self.window_size
    window = ohlc_data[window_start:window_end]  # shape: (window_size, 4)
    batch_windows.append(window)

windows = np.array(batch_windows)  # shape: (batch_size, window_size, 4)
```

### Fix 5: Data Type Overflow Prevention
**File**: `src/data_processor.py` (lines 154-168)
**Added**: Data range checking and appropriate dtype selection
```python
# Check data range to avoid float32 overflow
max_val = np.max(ohlc_data)
min_val = np.min(ohlc_data)
logger.info(f"价格数据范围: {min_val:.2f} - {max_val:.2f}")

# Use appropriate precision based on data range
if max_val > 1e6 or min_val < -1e6:
    logger.warning("价格数据范围较大，使用float64精度")
    ohlc_data = ohlc_data.astype(np.float64)
else:
    ohlc_data = ohlc_data.astype(np.float32)
```

## Expected Result
After these fixes, the training should proceed without the parameter mismatch error and successfully:
1. Initialize the GPUDataProcessor with correct parameters
2. Load and process the data using the new method
3. Extract features and labels properly
4. Continue with the genetic algorithm training

## Files Modified
1. `core/main_gpu.py` - Fixed parameter name
2. `src/data_processor.py` - Added missing method, fixed sliding window, added overflow prevention
3. `src/normalization_strategies.py` - Added missing property

## Testing
Created test files to verify the fixes:
- `test_fix.py` - Comprehensive test of the GPUDataProcessor
- `test_minimal.py` - Minimal training test with small parameters
- `test_fix.bat` - Batch file for easy testing on Windows

The fixes ensure compatibility between the main training script and the data processing modules.