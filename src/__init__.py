"""Disease Outbreak Prediction - Healthcare AI Research Project.

This package provides tools for predicting disease outbreaks using time series
forecasting with various neural network architectures.

IMPORTANT DISCLAIMER:
This is a research and educational project only. It is NOT intended for clinical
use, diagnostic purposes, or medical decision-making. All predictions should be
validated by qualified healthcare professionals.
"""

__version__ = "1.0.0"
__author__ = "Healthcare AI Research Team"
__email__ = "research@example.com"

# Import main components
from .data import OutbreakDataset, OutbreakDataLoader, generate_synthetic_data
from .models import create_model, LSTMPredictor, GRUPredictor, TransformerPredictor
from .training import train_model, Trainer
from .evaluation import ModelEvaluator, TimeSeriesMetrics, UncertaintyQuantifier
from .explainability import ModelExplainer, FeatureImportanceAnalyzer
from .utils import set_seed, get_device, setup_logging

__all__ = [
    # Data
    "OutbreakDataset",
    "OutbreakDataLoader", 
    "generate_synthetic_data",
    
    # Models
    "create_model",
    "LSTMPredictor",
    "GRUPredictor", 
    "TransformerPredictor",
    
    # Training
    "train_model",
    "Trainer",
    
    # Evaluation
    "ModelEvaluator",
    "TimeSeriesMetrics",
    "UncertaintyQuantifier",
    
    # Explainability
    "ModelExplainer",
    "FeatureImportanceAnalyzer",
    
    # Utils
    "set_seed",
    "get_device",
    "setup_logging",
]
