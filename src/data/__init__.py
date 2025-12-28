"""Data loading and preprocessing utilities for disease outbreak prediction."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from ..utils import normalize_data, denormalize_data, create_time_features


logger = logging.getLogger(__name__)


class OutbreakDataset(Dataset):
    """Dataset class for disease outbreak time series data.
    
    This class handles loading, preprocessing, and batching of time series data
    for disease outbreak prediction models.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 10,
        pred_len: int = 1,
        features: Optional[np.ndarray] = None,
        normalize: bool = True,
        normalization_method: str = "minmax"
    ):
        """Initialize the dataset.
        
        Args:
            data: Time series data array.
            seq_len: Length of input sequences.
            pred_len: Length of prediction horizon.
            features: Optional additional features.
            normalize: Whether to normalize the data.
            normalization_method: Method for normalization.
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Normalize data if requested
        if self.normalize:
            self.normalized_data, self.norm_params = normalize_data(
                self.data, self.normalization_method
            )
        else:
            self.normalized_data = self.data
            self.norm_params = {}
        
        # Create sequences
        self.X, self.y = self._create_sequences()
        
        logger.info(f"Created dataset with {len(self.X)} sequences")
        logger.info(f"Sequence length: {seq_len}, Prediction length: {pred_len}")
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output sequences from time series data.
        
        Returns:
            Tuple of input sequences and target values.
        """
        X, y = [], []
        
        for i in range(len(self.normalized_data) - self.seq_len - self.pred_len + 1):
            # Input sequence
            seq = self.normalized_data[i:i + self.seq_len]
            
            # Target (single value or sequence)
            if self.pred_len == 1:
                target = self.normalized_data[i + self.seq_len]
            else:
                target = self.normalized_data[i + self.seq_len:i + self.seq_len + self.pred_len]
            
            X.append(seq)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sequence and target.
        
        Args:
            idx: Index of the sequence.
            
        Returns:
            Tuple of input sequence and target tensors.
        """
        seq = torch.tensor(self.X[idx], dtype=torch.float32)
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        
        # Add feature dimension if needed
        if len(seq.shape) == 1:
            seq = seq.unsqueeze(-1)
        
        return seq, target
    
    def denormalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Denormalize model predictions.
        
        Args:
            predictions: Normalized predictions.
            
        Returns:
            Denormalized predictions.
        """
        if not self.normalize:
            return predictions
        
        return denormalize_data(predictions, self.norm_params)


class OutbreakDataLoader:
    """Data loader utility for outbreak prediction datasets."""
    
    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True
    ):
        """Initialize the data loader.
        
        Args:
            batch_size: Batch size for data loading.
            shuffle: Whether to shuffle the data.
            num_workers: Number of worker processes.
            pin_memory: Whether to pin memory for faster GPU transfer.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def create_loaders(
        self,
        dataset: OutbreakDataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders.
        
        Args:
            dataset: The dataset to split.
            train_ratio: Ratio of data for training.
            val_ratio: Ratio of data for validation.
            test_ratio: Ratio of data for testing.
            
        Returns:
            Tuple of train, validation, and test data loaders.
        """
        # Calculate split indices
        n_samples = len(dataset)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        test_size = n_samples - train_size - val_size
        
        # Split the dataset
        train_dataset, temp_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size + test_size]
        )
        val_dataset, test_dataset = torch.utils.data.random_split(
            temp_dataset, [val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        logger.info(f"Created data loaders: train={len(train_dataset)}, "
                   f"val={len(val_dataset)}, test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader


def generate_synthetic_data(
    n_samples: int = 1000,
    seasonality: bool = True,
    trend: bool = True,
    noise_level: float = 0.1,
    outbreak_prob: float = 0.05,
    outbreak_magnitude: float = 2.0
) -> np.ndarray:
    """Generate synthetic disease outbreak data.
    
    Args:
        n_samples: Number of time points to generate.
        seasonality: Whether to include seasonal patterns.
        trend: Whether to include trend component.
        noise_level: Level of random noise.
        outbreak_prob: Probability of outbreak occurrence.
        outbreak_magnitude: Magnitude multiplier for outbreaks.
        
    Returns:
        Generated time series data.
    """
    # Base level
    base_level = 100
    
    # Trend component
    if trend:
        trend_component = np.linspace(0, 50, n_samples)
    else:
        trend_component = np.zeros(n_samples)
    
    # Seasonal component
    if seasonality:
        seasonal_component = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 52)  # Weekly seasonality
        seasonal_component += 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365)  # Yearly seasonality
    else:
        seasonal_component = np.zeros(n_samples)
    
    # Outbreak component
    outbreak_component = np.zeros(n_samples)
    outbreak_indices = np.random.choice(
        n_samples, 
        size=int(n_samples * outbreak_prob), 
        replace=False
    )
    
    for idx in outbreak_indices:
        # Create outbreak with exponential decay
        outbreak_length = np.random.randint(5, 20)
        outbreak_start = max(0, idx - outbreak_length // 2)
        outbreak_end = min(n_samples, idx + outbreak_length // 2)
        
        outbreak_values = outbreak_magnitude * np.exp(
            -0.3 * np.abs(np.arange(outbreak_start, outbreak_end) - idx)
        )
        outbreak_component[outbreak_start:outbreak_end] += outbreak_values
    
    # Noise component
    noise_component = np.random.normal(0, noise_level * base_level, n_samples)
    
    # Combine all components
    data = base_level + trend_component + seasonal_component + outbreak_component + noise_component
    
    # Ensure non-negative values
    data = np.maximum(data, 0)
    
    logger.info(f"Generated synthetic data with {n_samples} samples")
    logger.info(f"Mean: {np.mean(data):.2f}, Std: {np.std(data):.2f}")
    
    return data


def load_real_data(file_path: str, target_column: str = "cases") -> np.ndarray:
    """Load real disease outbreak data from CSV file.
    
    Args:
        file_path: Path to the CSV file.
        target_column: Name of the target column.
        
    Returns:
        Loaded time series data.
    """
    try:
        df = pd.read_csv(file_path)
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        data = df[target_column].values
        
        # Handle missing values
        if np.isnan(data).any():
            logger.warning("Found missing values in data, filling with forward fill")
            data = pd.Series(data).fillna(method='ffill').values
        
        logger.info(f"Loaded real data with {len(data)} samples")
        logger.info(f"Mean: {np.mean(data):.2f}, Std: {np.std(data):.2f}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error loading real data: {e}")
        raise


def create_time_series_features(
    data: np.ndarray,
    window_sizes: List[int] = [7, 14, 30]
) -> np.ndarray:
    """Create additional time series features.
    
    Args:
        data: Input time series data.
        window_sizes: List of window sizes for rolling statistics.
        
    Returns:
        Array of additional features.
    """
    features = []
    
    for window in window_sizes:
        if len(data) >= window:
            # Rolling mean
            rolling_mean = pd.Series(data).rolling(window=window).mean().values
            features.append(rolling_mean)
            
            # Rolling std
            rolling_std = pd.Series(data).rolling(window=window).std().values
            features.append(rolling_std)
            
            # Rolling max
            rolling_max = pd.Series(data).rolling(window=window).max().values
            features.append(rolling_max)
            
            # Rolling min
            rolling_min = pd.Series(data).rolling(window=window).min().values
            features.append(rolling_min)
    
    # Lag features
    for lag in [1, 2, 3, 7, 14]:
        if len(data) > lag:
            lag_feature = np.concatenate([np.full(lag, np.nan), data[:-lag]])
            features.append(lag_feature)
    
    # Combine all features
    if features:
        feature_matrix = np.column_stack(features)
        # Fill NaN values
        feature_matrix = pd.DataFrame(feature_matrix).fillna(method='ffill').values
    else:
        feature_matrix = np.zeros((len(data), 1))
    
    logger.info(f"Created {feature_matrix.shape[1]} additional features")
    
    return feature_matrix
