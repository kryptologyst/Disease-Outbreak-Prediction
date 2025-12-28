"""Evaluation metrics and utilities for disease outbreak prediction."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import get_device


logger = logging.getLogger(__name__)


class TimeSeriesMetrics:
    """Time series specific evaluation metrics."""
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Directional accuracy (percentage of correct trend predictions)."""
        if len(y_true) < 2:
            return 0.0
        
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        return np.mean(true_direction == pred_direction) * 100
    
    @staticmethod
    def theil_u(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Theil's U statistic."""
        numerator = np.sqrt(np.mean((y_true - y_pred) ** 2))
        denominator = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
        return numerator / (denominator + 1e-8)


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """Initialize the evaluator.
        
        Args:
            model: Model to evaluate.
            device: Device to run evaluation on.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.model.eval()
        
        self.metrics = TimeSeriesMetrics()
        
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        denormalize_fn: Optional[callable] = None
    ) -> Dict[str, float]:
        """Evaluate the model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation.
            denormalize_fn: Optional function to denormalize predictions.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Get predictions
                pred = self.model(data)
                
                # Move to CPU and convert to numpy
                pred = pred.cpu().numpy()
                target = target.cpu().numpy()
                
                predictions.extend(pred)
                targets.extend(target)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Denormalize if function provided
        if denormalize_fn is not None:
            predictions = denormalize_fn(predictions)
            targets = denormalize_fn(targets)
        
        # Calculate metrics
        metrics = {
            "mae": self.metrics.mae(targets, predictions),
            "mse": self.metrics.mse(targets, predictions),
            "rmse": self.metrics.rmse(targets, predictions),
            "mape": self.metrics.mape(targets, predictions),
            "smape": self.metrics.smape(targets, predictions),
            "r2": r2_score(targets, predictions),
            "directional_accuracy": self.metrics.directional_accuracy(targets, predictions),
            "theil_u": self.metrics.theil_u(targets, predictions)
        }
        
        return metrics
    
    def predict(
        self,
        data_loader: torch.utils.data.DataLoader,
        denormalize_fn: Optional[callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from the model.
        
        Args:
            data_loader: Data loader for prediction.
            denormalize_fn: Optional function to denormalize predictions.
            
        Returns:
            Tuple of predictions and targets.
        """
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                pred = self.model(data)
                
                pred = pred.cpu().numpy()
                target = target.cpu().numpy()
                
                predictions.extend(pred)
                targets.extend(target)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        if denormalize_fn is not None:
            predictions = denormalize_fn(predictions)
            targets = denormalize_fn(targets)
        
        return predictions, targets


class UncertaintyQuantifier:
    """Uncertainty quantification for model predictions."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """Initialize the uncertainty quantifier.
        
        Args:
            model: Model to evaluate.
            device: Device to run evaluation on.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        
    def monte_carlo_dropout(
        self,
        data_loader: torch.utils.data.DataLoader,
        n_samples: int = 100,
        dropout_rate: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get uncertainty estimates using Monte Carlo dropout.
        
        Args:
            data_loader: Data loader for prediction.
            n_samples: Number of Monte Carlo samples.
            dropout_rate: Dropout rate for uncertainty estimation.
            
        Returns:
            Tuple of mean predictions and uncertainty estimates.
        """
        self.model.train()  # Enable dropout
        
        all_predictions = []
        
        for _ in range(n_samples):
            predictions = []
            
            with torch.no_grad():
                for data, _ in data_loader:
                    data = data.to(self.device)
                    pred = self.model(data)
                    predictions.extend(pred.cpu().numpy())
            
            all_predictions.append(predictions)
        
        all_predictions = np.array(all_predictions)
        
        # Calculate mean and uncertainty
        mean_pred = np.mean(all_predictions, axis=0)
        uncertainty = np.std(all_predictions, axis=0)
        
        self.model.eval()  # Disable dropout
        
        return mean_pred, uncertainty
    
    def prediction_intervals(
        self,
        data_loader: torch.utils.data.DataLoader,
        confidence: float = 0.95,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate prediction intervals.
        
        Args:
            data_loader: Data loader for prediction.
            confidence: Confidence level for intervals.
            n_samples: Number of Monte Carlo samples.
            
        Returns:
            Tuple of mean predictions, lower bounds, and upper bounds.
        """
        mean_pred, uncertainty = self.monte_carlo_dropout(data_loader, n_samples)
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        z_score = 1.96  # For 95% confidence
        
        lower_bound = mean_pred - z_score * uncertainty
        upper_bound = mean_pred + z_score * uncertainty
        
        return mean_pred, lower_bound, upper_bound


class ModelComparison:
    """Compare multiple models and create leaderboard."""
    
    def __init__(self):
        """Initialize the model comparison."""
        self.results = {}
        
    def add_model(
        self,
        model_name: str,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        denormalize_fn: Optional[callable] = None
    ) -> Dict[str, float]:
        """Add a model to the comparison.
        
        Args:
            model_name: Name of the model.
            model: Model to evaluate.
            test_loader: Test data loader.
            denormalize_fn: Optional denormalization function.
            
        Returns:
            Evaluation metrics for the model.
        """
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(test_loader, denormalize_fn)
        
        self.results[model_name] = metrics
        
        logger.info(f"Added {model_name} to comparison")
        logger.info(f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
        
        return metrics
    
    def create_leaderboard(self, metric: str = "rmse") -> pd.DataFrame:
        """Create a leaderboard sorted by the specified metric.
        
        Args:
            metric: Metric to sort by.
            
        Returns:
            DataFrame with model rankings.
        """
        if not self.results:
            logger.warning("No models added to comparison")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(self.results).T
        df = df.sort_values(metric)
        df['rank'] = range(1, len(df) + 1)
        
        # Reorder columns
        cols = ['rank'] + [col for col in df.columns if col != 'rank']
        df = df[cols]
        
        return df
    
    def plot_comparison(self, metric: str = "rmse", save_path: Optional[str] = None):
        """Plot model comparison.
        
        Args:
            metric: Metric to plot.
            save_path: Optional path to save the plot.
        """
        if not self.results:
            logger.warning("No models added to comparison")
            return
        
        df = self.create_leaderboard(metric)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x=df.index, y=metric)
        plt.title(f"Model Comparison - {metric.upper()}")
        plt.xlabel("Model")
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def evaluate_model_performance(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    denormalize_fn: Optional[callable] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Comprehensive model evaluation.
    
    Args:
        model: Model to evaluate.
        test_loader: Test data loader.
        denormalize_fn: Optional denormalization function.
        device: Device to run evaluation on.
        
    Returns:
        Dictionary with evaluation results.
    """
    evaluator = ModelEvaluator(model, device)
    
    # Basic metrics
    metrics = evaluator.evaluate(test_loader, denormalize_fn)
    
    # Uncertainty quantification
    uncertainty_quantifier = UncertaintyQuantifier(model, device)
    mean_pred, uncertainty = uncertainty_quantifier.monte_carlo_dropout(test_loader)
    
    # Get predictions for plotting
    predictions, targets = evaluator.predict(test_loader, denormalize_fn)
    
    results = {
        "metrics": metrics,
        "predictions": predictions,
        "targets": targets,
        "uncertainty": uncertainty,
        "mean_predictions": mean_pred
    }
    
    return results


def create_evaluation_report(
    results: Dict[str, Any],
    model_name: str,
    save_dir: str = "assets"
) -> None:
    """Create a comprehensive evaluation report.
    
    Args:
        results: Evaluation results.
        model_name: Name of the model.
        save_dir: Directory to save plots.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = results["metrics"]
    predictions = results["predictions"]
    targets = results["targets"]
    uncertainty = results["uncertainty"]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Prediction vs Actual
    axes[0, 0].scatter(targets, predictions, alpha=0.6)
    axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    axes[0, 0].set_xlabel("Actual")
    axes[0, 0].set_ylabel("Predicted")
    axes[0, 0].set_title("Predicted vs Actual")
    
    # Residuals
    residuals = predictions - targets
    axes[0, 1].scatter(predictions, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Residual Plot")
    
    # Time series plot
    axes[1, 0].plot(targets[:100], label="Actual", alpha=0.7)
    axes[1, 0].plot(predictions[:100], label="Predicted", alpha=0.7)
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_title("Time Series Comparison")
    axes[1, 0].legend()
    
    # Uncertainty plot
    axes[1, 1].fill_between(
        range(len(uncertainty[:100])),
        predictions[:100] - uncertainty[:100],
        predictions[:100] + uncertainty[:100],
        alpha=0.3,
        label="Uncertainty"
    )
    axes[1, 1].plot(targets[:100], label="Actual", alpha=0.7)
    axes[1, 1].plot(predictions[:100], label="Predicted", alpha=0.7)
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].set_title("Predictions with Uncertainty")
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_evaluation.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print metrics
    print(f"\n{model_name} Evaluation Results:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Save metrics to file
    with open(f"{save_dir}/{model_name}_metrics.txt", "w") as f:
        f.write(f"{model_name} Evaluation Results\n")
        f.write("=" * 50 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric.upper()}: {value:.4f}\n")
