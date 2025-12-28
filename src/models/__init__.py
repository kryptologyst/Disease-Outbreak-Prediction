"""Model implementations for disease outbreak prediction."""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..utils import get_device


logger = logging.getLogger(__name__)


class LSTMPredictor(nn.Module):
    """LSTM-based disease outbreak predictor.
    
    This model uses LSTM layers to capture temporal dependencies in
    disease outbreak time series data.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        bidirectional: bool = False
    ):
        """Initialize the LSTM predictor.
        
        Args:
            input_size: Number of input features.
            hidden_size: Size of hidden state.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
            output_size: Number of output predictions.
            bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).
            
        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply output layer
        output = self.fc(last_output)
        
        return output


class GRUPredictor(nn.Module):
    """GRU-based disease outbreak predictor.
    
    This model uses GRU layers for faster training while maintaining
    good performance on time series data.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        bidirectional: bool = False
    ):
        """Initialize the GRU predictor.
        
        Args:
            input_size: Number of input features.
            hidden_size: Size of hidden state.
            num_layers: Number of GRU layers.
            dropout: Dropout rate.
            output_size: Number of output predictions.
            bidirectional: Whether to use bidirectional GRU.
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).
            
        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Use the last output
        last_output = gru_out[:, -1, :]
        
        # Apply output layer
        output = self.fc(last_output)
        
        return output


class TransformerPredictor(nn.Module):
    """Transformer-based disease outbreak predictor.
    
    This model uses self-attention mechanisms to capture long-range
    dependencies in time series data.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        output_size: int = 1,
        max_seq_len: int = 1000
    ):
        """Initialize the Transformer predictor.
        
        Args:
            input_size: Number of input features.
            d_model: Model dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer layers.
            dropout: Dropout rate.
            output_size: Number of output predictions.
            max_seq_len: Maximum sequence length for positional encoding.
        """
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(max_seq_len, d_model) * 0.1
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).
            
        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        if seq_len <= self.max_seq_len:
            x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer forward pass
        transformer_out = self.transformer(x)
        
        # Use the last output
        last_output = transformer_out[:, -1, :]
        
        # Apply output layer
        output = self.fc(last_output)
        
        return output


class CNNLSTMPredictor(nn.Module):
    """CNN-LSTM hybrid predictor for disease outbreak prediction.
    
    This model combines convolutional layers for local pattern detection
    with LSTM layers for temporal modeling.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        cnn_channels: List[int] = [32, 64],
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """Initialize the CNN-LSTM predictor.
        
        Args:
            input_size: Number of input features.
            cnn_channels: List of CNN output channels.
            lstm_hidden_size: LSTM hidden state size.
            lstm_num_layers: Number of LSTM layers.
            dropout: Dropout rate.
            output_size: Number of output predictions.
        """
        super().__init__()
        
        self.input_size = input_size
        self.cnn_channels = cnn_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout = dropout
        self.output_size = output_size
        
        # CNN layers
        cnn_layers = []
        in_channels = input_size
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size // 2, output_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).
            
        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        batch_size, seq_len, input_size = x.shape
        
        # Reshape for CNN: (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # CNN forward pass
        cnn_out = self.cnn(x)
        
        # Reshape for LSTM: (batch_size, seq_len, cnn_channels[-1])
        cnn_out = cnn_out.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(cnn_out)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply output layer
        output = self.fc(last_output)
        
        return output


class EnsemblePredictor(nn.Module):
    """Ensemble predictor combining multiple models.
    
    This model combines predictions from multiple base models
    to improve overall performance and robustness.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        method: str = "weighted_average"
    ):
        """Initialize the ensemble predictor.
        
        Args:
            models: List of base models.
            weights: Optional weights for each model.
            method: Ensemble method ('weighted_average', 'majority_vote').
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.method = method
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
        
        logger.info(f"Initialized ensemble with {len(models)} models")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Ensemble prediction.
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        if self.method == "weighted_average":
            # Weighted average of predictions
            weighted_pred = torch.zeros_like(predictions[0])
            for pred, weight in zip(predictions, self.weights):
                weighted_pred += weight * pred
            return weighted_pred
        
        elif self.method == "majority_vote":
            # Majority vote (for classification)
            stacked_preds = torch.stack(predictions, dim=0)
            return torch.mean(stacked_preds, dim=0)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")


def create_model(
    model_type: str,
    input_size: int = 1,
    output_size: int = 1,
    **kwargs
) -> nn.Module:
    """Create a model instance.
    
    Args:
        model_type: Type of model to create.
        input_size: Number of input features.
        output_size: Number of output predictions.
        **kwargs: Additional model parameters.
        
    Returns:
        Model instance.
    """
    model_classes = {
        "lstm": LSTMPredictor,
        "gru": GRUPredictor,
        "transformer": TransformerPredictor,
        "cnn_lstm": CNNLSTMPredictor
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = model_classes[model_type]
    model = model_class(
        input_size=input_size,
        output_size=output_size,
        **kwargs
    )
    
    logger.info(f"Created {model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model
