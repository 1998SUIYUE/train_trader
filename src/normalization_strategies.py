"""
Data normalization strategies for trading data
"""

import numpy as np
import torch
from typing import Union, Tuple

class DataNormalizer:
    """Data normalization strategies for financial time series"""
    
    def __init__(self, method: str = 'relative', window_size: int = 350):
        """
        Initialize normalizer
        
        Args:
            method: Normalization method ('relative', 'rolling', 'minmax', 'hybrid')
            window_size: Window size for rolling normalization
        """
        self.method = method
        self.window_size = window_size
        
    @property
    def feature_dim(self) -> int:
        """Calculate feature dimension based on method and window size"""
        if self.method == 'hybrid':
            # OHLC relative prices + RSI + MA ratio + volatility
            return self.window_size * 4 + 1 + 1 + self.window_size
        else:
            # Standard OHLC features
            return self.window_size * 4
        
    def normalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize data using specified method
        
        Args:
            data: Input data to normalize
            
        Returns:
            Normalized data
        """
        if self.method == 'relative':
            return self._relative_normalize(data)
        elif self.method == 'rolling':
            return self._rolling_normalize(data)
        elif self.method == 'minmax':
            return self._minmax_normalize(data)
        elif self.method == 'hybrid':
            return self._hybrid_normalize(data)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def _relative_normalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Relative normalization (divide by close price)"""
        if isinstance(data, torch.Tensor):
            # Assume OHLCV format: [Open, High, Low, Close, Volume]
            normalized = torch.zeros_like(data)
            close_prices = data[:, 3:4]  # Close price column
            normalized[:, :4] = data[:, :4] / close_prices  # Normalize OHLC
            normalized[:, 4:] = data[:, 4:] / torch.mean(data[:, 4:], dim=0, keepdim=True)  # Normalize volume
            return normalized
        else:
            # NumPy version
            normalized = np.zeros_like(data)
            close_prices = data[:, 3:4]
            normalized[:, :4] = data[:, :4] / close_prices
            normalized[:, 4:] = data[:, 4:] / np.mean(data[:, 4:], axis=0, keepdims=True)
            return normalized
    
    def _rolling_normalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Rolling window normalization"""
        if isinstance(data, torch.Tensor):
            normalized = torch.zeros_like(data)
            for i in range(len(data)):
                start_idx = max(0, i - self.window_size + 1)
                window_data = data[start_idx:i+1]
                mean_vals = torch.mean(window_data, dim=0)
                std_vals = torch.std(window_data, dim=0)
                std_vals = torch.where(std_vals == 0, torch.ones_like(std_vals), std_vals)
                normalized[i] = (data[i] - mean_vals) / std_vals
            return normalized
        else:
            # NumPy version
            normalized = np.zeros_like(data)
            for i in range(len(data)):
                start_idx = max(0, i - self.window_size + 1)
                window_data = data[start_idx:i+1]
                mean_vals = np.mean(window_data, axis=0)
                std_vals = np.std(window_data, axis=0)
                std_vals = np.where(std_vals == 0, 1, std_vals)
                normalized[i] = (data[i] - mean_vals) / std_vals
            return normalized
    
    def _minmax_normalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Min-max normalization"""
        if isinstance(data, torch.Tensor):
            min_vals = torch.min(data, dim=0)[0]
            max_vals = torch.max(data, dim=0)[0]
            range_vals = max_vals - min_vals
            range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
            return (data - min_vals) / range_vals
        else:
            # NumPy version
            min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
            range_vals = max_vals - min_vals
            range_vals = np.where(range_vals == 0, 1, range_vals)
            return (data - min_vals) / range_vals
    
    def _hybrid_normalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Hybrid normalization combining relative and rolling methods"""
        # First apply relative normalization
        relative_normalized = self._relative_normalize(data)
        # Then apply rolling normalization
        return self._rolling_normalize(relative_normalized)