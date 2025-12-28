"""Tests for the disease outbreak prediction project."""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils import set_seed, get_device, normalize_data, denormalize_data
from src.data import OutbreakDataset, OutbreakDataLoader, generate_synthetic_data
from src.models import create_model, LSTMPredictor, GRUPredictor
from src.training import Trainer
from src.evaluation import ModelEvaluator, TimeSeriesMetrics


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        rand1 = np.random.random()
        
        set_seed(42)
        rand2 = np.random.random()
        
        assert rand1 == rand2
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_normalize_data(self):
        """Test data normalization."""
        data = np.array([1, 2, 3, 4, 5])
        
        normalized, params = normalize_data(data, "minmax")
        
        assert np.allclose(normalized, [0, 0.25, 0.5, 0.75, 1])
        assert params["method"] == "minmax"
        assert params["min"] == 1
        assert params["max"] == 5
    
    def test_denormalize_data(self):
        """Test data denormalization."""
        data = np.array([1, 2, 3, 4, 5])
        normalized, params = normalize_data(data, "minmax")
        
        denormalized = denormalize_data(normalized, params)
        
        assert np.allclose(denormalized, data)


class TestData:
    """Test data loading and preprocessing."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        data = generate_synthetic_data(n_samples=100)
        
        assert len(data) == 100
        assert np.all(data >= 0)  # Non-negative values
        assert isinstance(data, np.ndarray)
    
    def test_outbreak_dataset(self):
        """Test OutbreakDataset class."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        dataset = OutbreakDataset(data, seq_len=3, pred_len=1)
        
        assert len(dataset) == 7  # 10 - 3 - 1 + 1
        assert dataset.X.shape == (7, 3)
        assert dataset.y.shape == (7,)
        
        # Test __getitem__
        seq, target = dataset[0]
        assert seq.shape == (3, 1)
        assert target.shape == ()
    
    def test_outbreak_data_loader(self):
        """Test OutbreakDataLoader class."""
        data = generate_synthetic_data(n_samples=100)
        dataset = OutbreakDataset(data, seq_len=5, pred_len=1)
        
        data_loader = OutbreakDataLoader(batch_size=8)
        train_loader, val_loader, test_loader = data_loader.create_loaders(dataset)
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        
        # Test that we can iterate through loaders
        for batch in train_loader:
            assert batch[0].shape[0] <= 8  # Batch size
            assert batch[0].shape[1] == 5  # Sequence length
            break


class TestModels:
    """Test model implementations."""
    
    def test_create_model(self):
        """Test model creation."""
        model = create_model("lstm", input_size=1, output_size=1)
        assert isinstance(model, LSTMPredictor)
        
        model = create_model("gru", input_size=1, output_size=1)
        assert isinstance(model, GRUPredictor)
    
    def test_lstm_predictor(self):
        """Test LSTM predictor."""
        model = LSTMPredictor(input_size=1, hidden_size=32, output_size=1)
        
        # Test forward pass
        x = torch.randn(2, 10, 1)  # batch_size=2, seq_len=10, input_size=1
        output = model(x)
        
        assert output.shape == (2, 1)  # batch_size=2, output_size=1
    
    def test_gru_predictor(self):
        """Test GRU predictor."""
        model = GRUPredictor(input_size=1, hidden_size=32, output_size=1)
        
        # Test forward pass
        x = torch.randn(2, 10, 1)
        output = model(x)
        
        assert output.shape == (2, 1)
    
    def test_model_parameters(self):
        """Test that models have trainable parameters."""
        model = LSTMPredictor(input_size=1, hidden_size=32, output_size=1)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count > 0


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_time_series_metrics(self):
        """Test time series metrics."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = TimeSeriesMetrics()
        
        mae = metrics.mae(y_true, y_pred)
        mse = metrics.mse(y_true, y_pred)
        rmse = metrics.rmse(y_true, y_pred)
        
        assert mae > 0
        assert mse > 0
        assert rmse > 0
        assert rmse == np.sqrt(mse)
    
    def test_model_evaluator(self):
        """Test ModelEvaluator class."""
        # Create a simple model
        model = LSTMPredictor(input_size=1, hidden_size=16, output_size=1)
        
        # Create dummy data
        data = generate_synthetic_data(n_samples=50)
        dataset = OutbreakDataset(data, seq_len=5, pred_len=1)
        
        data_loader = OutbreakDataLoader(batch_size=8)
        _, _, test_loader = data_loader.create_loaders(dataset)
        
        # Test evaluator
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(test_loader)
        
        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert all(isinstance(v, float) for v in metrics.values())


class TestTraining:
    """Test training functionality."""
    
    def test_trainer_initialization(self):
        """Test Trainer initialization."""
        model = LSTMPredictor(input_size=1, hidden_size=16, output_size=1)
        
        # Create dummy data
        data = generate_synthetic_data(n_samples=50)
        dataset = OutbreakDataset(data, seq_len=5, pred_len=1)
        
        data_loader = OutbreakDataLoader(batch_size=8)
        train_loader, val_loader, _ = data_loader.create_loaders(dataset)
        
        # Test trainer initialization
        config = {
            "learning_rate": 0.001,
            "max_epochs": 5,
            "patience": 3
        }
        
        trainer = Trainer(model, train_loader, val_loader, torch.device("cpu"), config)
        
        assert trainer.model == model
        assert trainer.max_epochs == 5
        assert trainer.learning_rate == 0.001
    
    def test_train_epoch(self):
        """Test training for one epoch."""
        model = LSTMPredictor(input_size=1, hidden_size=16, output_size=1)
        
        # Create dummy data
        data = generate_synthetic_data(n_samples=50)
        dataset = OutbreakDataset(data, seq_len=5, pred_len=1)
        
        data_loader = OutbreakDataLoader(batch_size=8)
        train_loader, val_loader, _ = data_loader.create_loaders(dataset)
        
        config = {"learning_rate": 0.001, "max_epochs": 1, "patience": 3}
        trainer = Trainer(model, train_loader, val_loader, torch.device("cpu"), config)
        
        # Test training one epoch
        loss = trainer.train_epoch()
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_validate_epoch(self):
        """Test validation for one epoch."""
        model = LSTMPredictor(input_size=1, hidden_size=16, output_size=1)
        
        # Create dummy data
        data = generate_synthetic_data(n_samples=50)
        dataset = OutbreakDataset(data, seq_len=5, pred_len=1)
        
        data_loader = OutbreakDataLoader(batch_size=8)
        train_loader, val_loader, _ = data_loader.create_loaders(dataset)
        
        config = {"learning_rate": 0.001, "max_epochs": 1, "patience": 3}
        trainer = Trainer(model, train_loader, val_loader, torch.device("cpu"), config)
        
        # Test validation one epoch
        loss = trainer.validate_epoch()
        assert isinstance(loss, float)
        assert loss >= 0


if __name__ == "__main__":
    pytest.main([__file__])
