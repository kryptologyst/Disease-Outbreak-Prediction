"""Utility functions for the disease outbreak prediction project."""

import random
import logging
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        PyTorch device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level.
        log_file: Optional log file path.
        
    Returns:
        Configured logger.
    """
    logger = logging.getLogger("disease_outbreak_prediction")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object.
        config_path: Path to save configuration.
    """
    OmegaConf.save(config, config_path)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def early_stopping(
    val_losses: list[float], 
    patience: int = 10, 
    min_delta: float = 0.001
) -> bool:
    """Check if early stopping criteria is met.
    
    Args:
        val_losses: List of validation losses.
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        
    Returns:
        True if early stopping should be triggered.
    """
    if len(val_losses) < patience + 1:
        return False
    
    best_loss = min(val_losses[:-patience])
    current_loss = val_losses[-1]
    
    return current_loss - best_loss > min_delta


def normalize_data(data: np.ndarray, method: str = "minmax") -> tuple[np.ndarray, Dict[str, Any]]:
    """Normalize time series data.
    
    Args:
        data: Input time series data.
        method: Normalization method ('minmax', 'zscore', 'robust').
        
    Returns:
        Tuple of normalized data and normalization parameters.
    """
    params = {}
    
    if method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        normalized = (data - min_val) / (max_val - min_val)
        params = {"min": min_val, "max": max_val, "method": "minmax"}
        
    elif method == "zscore":
        mean_val = np.mean(data)
        std_val = np.std(data)
        normalized = (data - mean_val) / std_val
        params = {"mean": mean_val, "std": std_val, "method": "zscore"}
        
    elif method == "robust":
        median_val = np.median(data)
        mad_val = np.median(np.abs(data - median_val))
        normalized = (data - median_val) / mad_val
        params = {"median": median_val, "mad": mad_val, "method": "robust"}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_data(
    normalized_data: np.ndarray, 
    params: Dict[str, Any]
) -> np.ndarray:
    """Denormalize time series data.
    
    Args:
        normalized_data: Normalized data.
        params: Normalization parameters.
        
    Returns:
        Denormalized data.
    """
    method = params["method"]
    
    if method == "minmax":
        return normalized_data * (params["max"] - params["min"]) + params["min"]
    elif method == "zscore":
        return normalized_data * params["std"] + params["mean"]
    elif method == "robust":
        return normalized_data * params["mad"] + params["median"]
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_time_features(timestamps: np.ndarray) -> np.ndarray:
    """Create time-based features from timestamps.
    
    Args:
        timestamps: Array of timestamps.
        
    Returns:
        Array of time features.
    """
    features = []
    
    for ts in timestamps:
        # Convert to datetime if needed
        if isinstance(ts, (int, float)):
            # Assume it's a day index
            day_of_week = ts % 7
            day_of_year = ts % 365
            week_of_year = ts // 7
            month = (ts % 365) // 30
        else:
            # Handle datetime objects
            day_of_week = ts.weekday()
            day_of_year = ts.timetuple().tm_yday
            week_of_year = ts.isocalendar()[1]
            month = ts.month
        
        features.append([
            np.sin(2 * np.pi * day_of_week / 7),  # Day of week (cyclical)
            np.cos(2 * np.pi * day_of_week / 7),
            np.sin(2 * np.pi * day_of_year / 365),  # Day of year (cyclical)
            np.cos(2 * np.pi * day_of_year / 365),
            week_of_year,
            month
        ])
    
    return np.array(features)


class EarlyStopping:
    """Early stopping utility class."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered.
        
        Args:
            val_loss: Current validation loss.
            
        Returns:
            True if early stopping should be triggered.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop
