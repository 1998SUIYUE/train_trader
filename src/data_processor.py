"""
数据处理模块
支持GPU加速的数据预处理和特征提取
"""

import pandas as pd
import numpy as np
import torch
from typing import Tuple, Optional, List, Dict
from pathlib import Path

try:
    from .cuda_gpu_utils import CudaGPUManager, get_cuda_gpu_manager
    CUDA_AVAILABLE = True
except ImportError:
    try:
        from cuda_gpu_utils import CudaGPUManager, get_cuda_gpu_manager
        CUDA_AVAILABLE = True
    except ImportError:
        CUDA_AVAILABLE = False

try:
    from .gpu_utils import WindowsGPUManager, get_windows_gpu_manager
    DIRECTML_AVAILABLE = True
except ImportError:
    try:
        from gpu_utils import WindowsGPUManager, get_windows_gpu_manager
        DIRECTML_AVAILABLE = True
    except ImportError:
        DIRECTML_AVAILABLE = False

try:
    from .normalization_strategies import DataNormalizer
except ImportError:
    from normalization_strategies import DataNormalizer

class GPUDataProcessor:
    """GPU加速数据处理器"""
    
    def __init__(self, gpu_manager=None, 
                 normalization_method: str = 'relative',
                 window_size: int = 350):
        """
        初始化数据处理器
        
        Args:
            gpu_manager: GPU管理器 (WindowsGPUManager 或 CudaGPUManager)
            normalization_method: 归一化方法
            window_size: 滑动窗口大小
        """
        # 自动选择GPU管理器
        if gpu_manager is None:
            # 优先尝试CUDA，如果不可用则使用DirectML
            import torch
            if CUDA_AVAILABLE and torch.cuda.is_available():
                self.gpu_manager = get_cuda_gpu_manager()
                print("Auto-selected CUDA GPU manager")
            elif DIRECTML_AVAILABLE:
                self.gpu_manager = get_windows_gpu_manager()
                print("Auto-selected DirectML GPU manager")
            else:
                # 创建一个基本的CPU管理器
                self.gpu_manager = self._create_cpu_manager()
                print("Using CPU manager (no GPU acceleration available)")
        else:
            self.gpu_manager = gpu_manager
        
        self.device = self.gpu_manager.device
        self.normalizer = DataNormalizer(normalization_method, window_size)
        self.window_size = window_size
        
        # 缓存数据
        self.raw_data = None
        self.processed_features = None
        self.price_series = None
        
        print("--- GPUDataProcessor配置 ---")
        print(f"归一化方法: {normalization_method}")
        print(f"滑动窗口大小: {window_size}")
        print(f"设备: {self.device}")
        print("---------------------------")
    
    def _create_cpu_manager(self):
        """创建一个基本的CPU管理器"""
        class CPUManager:
            def __init__(self):
                self.device = torch.device('cpu')
            
            def to_gpu(self, data, dtype=torch.float32):
                if isinstance(data, np.ndarray):
                    return torch.from_numpy(data).to(dtype)
                elif isinstance(data, torch.Tensor):
                    return data.to(dtype)
                else:
                    return torch.tensor(data, dtype=dtype)
            
            def to_cpu(self, tensor):
                return tensor.detach().numpy()
            
            def clear_cache(self):
                pass
        
        return CPUManager()
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        加载原始数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            原始数据DataFrame
        """
        print(f"加载数据: {file_path}")
        
        try:
            # 尝试不同的分隔符
            for sep in ['\t', ',', ';']:
                try:
                    df = pd.read_csv(file_path, sep=sep)
                    if len(df.columns) >= 4:  # 至少需要OHLC四列
                        break
                except:
                    continue
            else:
                raise ValueError("无法解析CSV文件")
            
            print(f"数据加载成功: {df.shape}")
            print(f"列名: {list(df.columns)}")
            
            # 标准化列名
            df = self._standardize_columns(df)
            
            # 基础数据清理
            df = self._clean_data(df)
            
            self.raw_data = df
            print("--- 原始数据加载完成 ---")
            print(f"数据形状: {df.shape}")
            print(f"数据列名: {df.columns.tolist()}")
            print(f"前5行数据:\n{df.head()}")
            print("-------------------------")
            return df
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        # 常见的OHLC列名映射
        column_mapping = {
            '<OPEN>': 'open',
            '<HIGH>': 'high', 
            '<LOW>': 'low',
            '<CLOSE>': 'close',
            '<DATE>': 'date',
            '<TIME>': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low', 
            'Close': 'close',
            'OPEN': 'open',
            'HIGH': 'high',
            'LOW': 'low',
            'CLOSE': 'close'
        }
        
        df = df.rename(columns=column_mapping)
        
        # 删除不需要的列
        columns_to_drop = ['<TICKVOL>', '<VOL>', '<SPREAD>', 'Volume', 'volume']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清理"""
        print(f"数据清理前: {df.shape}")
        
        # 删除缺失值
        df = df.dropna()
        print(f"删除缺失值后: {df.shape}")
        
        # 删除异常值 (价格为0或负数)
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                before_count = len(df)
                df = df[df[col] > 0]
                after_count = len(df)
                if before_count != after_count:
                    print(f"删除{col}列中<=0的值: {before_count} -> {after_count}")
        
        # 删除极端异常值 (可能是数据错误)
        for col in price_columns:
            if col in df.columns:
                before_count = len(df)
                # 使用四分位数方法检测异常值
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # 使用3倍IQR作为极端异常值阈值
                upper_bound = Q3 + 3 * IQR
                
                # 额外的安全检查：价格不应该超过合理范围
                max_reasonable_price = df[col].median() * 100  # 中位数的100倍作为上限
                upper_bound = min(upper_bound, max_reasonable_price)
                
                df = df[(df[col] >= max(lower_bound, 0)) & (df[col] <= upper_bound)]
                after_count = len(df)
                if before_count != after_count:
                    print(f"删除{col}列中的极端异常值: {before_count} -> {after_count} (范围: {lower_bound:.2f} - {upper_bound:.2f})")
        
        # 检查OHLC逻辑一致性
        if all(col in df.columns for col in price_columns):
            before_count = len(df)
            # High应该是最高价
            df = df[df['high'] >= df[['open', 'close']].max(axis=1)]
            # Low应该是最低价  
            df = df[df['low'] <= df[['open', 'close']].min(axis=1)]
            after_count = len(df)
            if before_count != after_count:
                print(f"删除OHLC逻辑不一致的数据: {before_count} -> {after_count}")
        
        # 最终检查：确保还有足够的数据
        if len(df) < self.window_size:
            raise ValueError(f"清理后数据不足，仅剩{len(df)}行，需要至少{self.window_size}行")
        
        print(f"数据清理完成: {df.shape}")
        return df
    
    def extract_features_gpu(self, batch_size: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU加速特征提取
        
        Args:
            batch_size: 批处理大小
            
        Returns:
            (特征矩阵, 价格序列)
        """
        if self.raw_data is None:
            raise ValueError("请先加载数据")
        
        print("开始GPU加速特征提取")
        
        # 提取OHLC数据
        ohlc_data = self.raw_data[['open', 'high', 'low', 'close']].values
        
        # 检查数据范围，避免float32溢出
        max_val = np.max(ohlc_data)
        min_val = np.min(ohlc_data)
        print(f"价格数据范围: {min_val:.2f} - {max_val:.2f}")
        
        # 如果数据范围过大，使用float64，否则使用float32
        if max_val > 1e6 or min_val < -1e6:
            print("价格数据范围较大，使用float64精度")
            ohlc_data = ohlc_data.astype(np.float64)
        else:
            ohlc_data = ohlc_data.astype(np.float32)
            
        prices = ohlc_data[:, 3]  # 收盘价序列
        
        # 计算可用的窗口数量
        n_samples = len(ohlc_data)
        n_windows = n_samples - self.window_size + 1
        
        if n_windows <= 0:
            raise ValueError(f"数据长度 {n_samples} 小于窗口大小 {self.window_size}")
        
        print(f"总样本数: {n_samples}, 可用窗口数: {n_windows}")
        
        # 分批处理大数据
        all_features = []
        
        for start_idx in range(0, n_windows, batch_size):
            end_idx = min(start_idx + batch_size, n_windows)
            batch_features = self._extract_batch_features_gpu(
                ohlc_data, start_idx, end_idx
            )
            all_features.append(batch_features)
        
        # 合并所有批次
        features_tensor = torch.cat(all_features, dim=0)
        prices_tensor = self.gpu_manager.to_gpu(prices[self.window_size-1:])
        
        self.processed_features = features_tensor
        self.price_series = prices_tensor
        
        print("--- 特征提取完成 ---")
        print(f"特征张量形状: {features_tensor.shape}")
        print(f"价格张量形状: {prices_tensor.shape}")
        print(f"特征张量设备: {features_tensor.device}")
        print("----------------------")
        
        return features_tensor, prices_tensor
    
    def _extract_batch_features_gpu(self, ohlc_data: np.ndarray, 
                                  start_idx: int, end_idx: int) -> torch.Tensor:
        """
        批量提取特征 (GPU加速优化版本)
        
        Args:
            ohlc_data: OHLC数据
            start_idx: 开始索引
            end_idx: 结束索引
            
        Returns:
            批次特征张量
        """
        batch_size = end_idx - start_idx
        
        # GPU优化：直接在GPU上创建滑动窗口
        # 先将整个OHLC数据转移到GPU
        ohlc_gpu = self.gpu_manager.to_gpu(ohlc_data)
        
        # 使用GPU张量操作创建滑动窗口
        windows_list = []
        for i in range(start_idx, end_idx):
            window_start = i
            window_end = i + self.window_size
            window = ohlc_gpu[window_start:window_end]  # 直接在GPU上切片
            windows_list.append(window)
        
        # 在GPU上堆叠窗口
        windows_gpu = torch.stack(windows_list, dim=0)  # shape: (batch_size, window_size, 4)
        
        # GPU向量化归一化
        if self.normalizer.method == 'relative':
            # 相对价格归一化 (完全在GPU上)
            base_prices = windows_gpu[:, 0, 3:4]  # 首个收盘价 (batch_size, 1)
            normalized_windows = windows_gpu / base_prices.unsqueeze(-1)
            
            # 只在第一批次时显示归一化信息
            if start_idx == 0:
                print("\n--- 归一化数据展示 (相对价格) ---")
                print(f"归一化后形状: {normalized_windows.shape}")
                if normalized_windows.shape[0] > 0:
                    sample_data = self.gpu_manager.to_cpu(normalized_windows[0, :5, :])
                    print(f"第一个窗口的归一化数据 (前5行):\n{sample_data}")
                print("-------------------------------------\n")

            batch_features = normalized_windows.reshape(batch_size, -1)
        
        elif self.normalizer.method == 'rolling':
            # 滚动标准化 (GPU向量化)
            mean_vals = torch.mean(windows_gpu, dim=1, keepdim=True)
            std_vals = torch.std(windows_gpu, dim=1, keepdim=True)
            
            # GPU上避免除以零
            std_vals = torch.clamp(std_vals, min=1e-8)
            normalized_windows = (windows_gpu - mean_vals) / std_vals
            
            # GPU上清理NaN和inf值
            normalized_windows = torch.nan_to_num(normalized_windows, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 只在第一批次时显示归一化信息
            if start_idx == 0:
                print("\n--- 归一化数据展示 (滚动标准化) ---")
                print(f"归一化后形状: {normalized_windows.shape}")
                if normalized_windows.shape[0] > 0:
                    sample_data = self.gpu_manager.to_cpu(normalized_windows[0, :5, :])
                    print(f"第一个窗口的归一化数据 (前5行):\n{sample_data}")
                print("-------------------------------------\n")

            batch_features = normalized_windows.reshape(batch_size, -1)
        
        elif self.normalizer.method == 'hybrid':
            # 混合归一化 (包含技术指标)
            batch_features = self._compute_hybrid_features_gpu(windows_gpu)
        
        else:
            raise ValueError(f"不支持的归一化方法: {self.normalizer.method}")
        
        return batch_features
    
    def _compute_hybrid_features_gpu(self, windows: torch.Tensor) -> torch.Tensor:
        """
        计算混合特征 (GPU加速)
        
        Args:
            windows: 窗口数据 (batch_size, window_size, 4)
            
        Returns:
            混合特征 (batch_size, feature_dim)
        """
        batch_size, window_size, _ = windows.shape
        
        # 相对价格归一化
        base_prices = windows[:, 0, 3:4]  # (batch_size, 1)
        relative_prices = windows / base_prices.unsqueeze(-1)
        
        # 提取收盘价序列
        closes = windows[:, :, 3]  # (batch_size, window_size)
        
        # 计算RSI (简化版本)
        rsi = self._compute_rsi_gpu(closes)
        
        # 计算移动平均比率
        ma_ratio = self._compute_ma_ratio_gpu(closes)
        
        # 计算波动率
        volatility = self._compute_volatility_gpu(closes)
        
        # 组合所有特征
        features = torch.cat([
            relative_prices.reshape(batch_size, -1),  # OHLC相对价格
            rsi.unsqueeze(-1),                        # RSI
            ma_ratio.unsqueeze(-1),                   # MA比率
            volatility.unsqueeze(-1).expand(-1, window_size)  # 波动率
        ], dim=-1)
        
        return features
    
    def _compute_rsi_gpu(self, closes: torch.Tensor, period: int = 14) -> torch.Tensor:
        """GPU加速RSI计算"""
        batch_size, window_size = closes.shape
        
        # 计算价格变化
        deltas = torch.diff(closes, dim=1)  # (batch_size, window_size-1)
        
        # 分离收益和损失
        gains = torch.where(deltas > 0, deltas, torch.zeros_like(deltas))
        losses = torch.where(deltas < 0, -deltas, torch.zeros_like(deltas))
        
        # 简化RSI计算 (使用滑动平均)
        if window_size > period:
            # 使用卷积计算滑动平均
            kernel = torch.ones(period, device=self.device) / period
            
            # 对每个批次计算RSI
            rsi_values = torch.zeros(batch_size, device=self.device)
            for i in range(batch_size):
                if len(gains[i]) >= period:
                    avg_gain = torch.mean(gains[i][-period:])
                    avg_loss = torch.mean(losses[i][-period:])
                    
                    if avg_loss > 1e-8:
                        rs = avg_gain / avg_loss
                        rsi_values[i] = 100 - (100 / (1 + rs))
                    else:
                        rsi_values[i] = 100.0
                else:
                    rsi_values[i] = 50.0  # 默认中性值
        else:
            rsi_values = torch.full((batch_size,), 50.0, device=self.device)
        
        return rsi_values / 100.0  # 归一化到[0,1]
    
    def _compute_ma_ratio_gpu(self, closes: torch.Tensor, 
                            short_period: int = 5, long_period: int = 20) -> torch.Tensor:
        """GPU加速移动平均比率计算"""
        batch_size, window_size = closes.shape
        
        ma_ratios = torch.ones(batch_size, device=self.device)
        
        if window_size >= long_period:
            # 计算短期和长期移动平均
            ma_short = torch.mean(closes[:, -short_period:], dim=1)
            ma_long = torch.mean(closes[:, -long_period:], dim=1)
            
            # 计算比率
            ma_ratios = torch.where(ma_long > 1e-8, ma_short / ma_long, ma_ratios)
        
        return ma_ratios
    
    def _compute_volatility_gpu(self, closes: torch.Tensor, period: int = 20) -> torch.Tensor:
        """GPU加速波动率计算"""
        batch_size, window_size = closes.shape
        
        if window_size >= period:
            # 计算收益率
            returns = torch.diff(closes[:, -period:], dim=1) / closes[:, -period:-1]
            # 计算波动率
            volatility = torch.std(returns, dim=1)
        else:
            volatility = torch.full((batch_size,), 0.01, device=self.device)
        
        return torch.clamp(volatility, 0, 1.0)  # 限制最大波动率
    
    def get_train_test_split(self, test_ratio: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        划分训练集和测试集
        
        Args:
            test_ratio: 测试集比例
            
        Returns:
            (训练特征, 测试特征, 训练价格, 测试价格)
        """
        if self.processed_features is None or self.price_series is None:
            raise ValueError("请先提取特征")
        
        n_samples = len(self.processed_features)
        split_idx = int(n_samples * (1 - test_ratio))
        
        train_features = self.processed_features[:split_idx]
        test_features = self.processed_features[split_idx:]
        train_prices = self.price_series[:split_idx]
        test_prices = self.price_series[split_idx:]
        
        print(f"数据划分完成 - 训练集: {len(train_features)}, 测试集: {len(test_features)}")
        
        return train_features, test_features, train_prices, test_prices
    
    def save_processed_data(self, filepath: str):
        """保存处理后的数据"""
        if self.processed_features is None or self.price_series is None:
            raise ValueError("没有可保存的处理数据")
        
        data_dict = {
            'features': self.gpu_manager.to_cpu(self.processed_features),
            'prices': self.gpu_manager.to_cpu(self.price_series),
            'window_size': self.window_size,
            'normalization_method': self.normalizer.method
        }
        
        torch.save(data_dict, filepath)
        print(f"处理数据已保存: {filepath}")
    
    def load_processed_data(self, filepath: str):
        """加载处理后的数据"""
        data_dict = torch.load(filepath, map_location='cpu')
        
        self.processed_features = self.gpu_manager.to_gpu(data_dict['features'])
        self.price_series = self.gpu_manager.to_gpu(data_dict['prices'])
        self.window_size = data_dict['window_size']
        
        print(f"处理数据已加载: {filepath}")
    
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
        
        # GPU优化：直接在GPU上计算标签（下一期收益率）
        returns = torch.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        
        # 移除最后一个特征样本（因为没有对应的下一期收益率）
        features = features[:-1]
        labels = returns[1:]  # 对应的收益率标签，已经在GPU上
        
        print(f"最终特征形状: {features.shape}")
        print(f"最终标签形状: {labels.shape}")
        
        return features, labels


if __name__ == "__main__":
    # 测试数据处理器
    print("=== GPU数据处理器测试 ===")
    
    # 创建模拟数据文件
    np.random.seed(42)
    n_samples = 5000
    
    # 模拟OHLC数据
    base_price = 2000
    price_changes = np.random.randn(n_samples) * 0.01
    closes = base_price + np.cumsum(price_changes)
    
    # 生成OHLC
    opens = np.roll(closes, 1)
    opens[0] = base_price
    
    highs = np.maximum(opens, closes) + np.random.exponential(0.5, n_samples)
    lows = np.minimum(opens, closes) - np.random.exponential(0.5, n_samples)
    
    # 创建DataFrame
    test_data = pd.DataFrame({
        '<OPEN>': opens,
        '<HIGH>': highs, 
        '<LOW>': lows,
        '<CLOSE>': closes,
        '<TICKVOL>': np.random.randint(100, 1000, n_samples),
        '<VOL>': np.random.randint(1000, 10000, n_samples)
    })
    
    # 保存测试数据
    test_file = 'test_data.csv'
    test_data.to_csv(test_file, sep='\t', index=False)
    
    try:
        # 测试数据处理器
        processor = GPUDataProcessor(normalization_method='relative')
        
        # 加载数据
        df = processor.load_data(test_file)
        print(f"数据加载成功: {df.shape}")
        
        # 提取特征
        features, prices = processor.extract_features_gpu(batch_size=1000)
        print(f"特征提取成功: {features.shape}, 价格序列: {prices.shape}")
        
        # 划分数据集
        train_feat, test_feat, train_prices, test_prices = processor.get_train_test_split()
        print(f"数据划分完成 - 训练: {train_feat.shape}, 测试: {test_feat.shape}")
        
        print("数据处理器测试完成！")
        
    finally:
        # 清理测试文件
        Path(test_file).unlink(missing_ok=True)