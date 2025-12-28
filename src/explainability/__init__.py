"""Explainability methods for disease outbreak prediction models."""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients, Saliency, GradientShap
from captum.attr import visualization as viz

from ..utils import get_device


logger = logging.getLogger(__name__)


class ModelExplainer:
    """Explainability wrapper for disease outbreak prediction models."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """Initialize the explainer.
        
        Args:
            model: Model to explain.
            device: Device to run explanations on.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize attribution methods
        self.integrated_gradients = IntegratedGradients(self.model)
        self.saliency = Saliency(self.model)
        self.gradient_shap = GradientShap(self.model)
        
    def integrated_gradients_attribution(
        self,
        input_tensor: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        n_steps: int = 50
    ) -> torch.Tensor:
        """Compute Integrated Gradients attribution.
        
        Args:
            input_tensor: Input tensor to explain.
            target: Target tensor (optional).
            n_steps: Number of integration steps.
            
        Returns:
            Attribution tensor.
        """
        input_tensor = input_tensor.to(self.device)
        if target is not None:
            target = target.to(self.device)
        
        # Create baseline (zeros)
        baseline = torch.zeros_like(input_tensor)
        
        # Compute attributions
        attributions = self.integrated_gradients.attribute(
            input_tensor,
            baselines=baseline,
            target=target,
            n_steps=n_steps
        )
        
        return attributions
    
    def saliency_attribution(
        self,
        input_tensor: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Saliency attribution.
        
        Args:
            input_tensor: Input tensor to explain.
            target: Target tensor (optional).
            
        Returns:
            Attribution tensor.
        """
        input_tensor = input_tensor.to(self.device)
        if target is not None:
            target = target.to(self.device)
        
        attributions = self.saliency.attribute(input_tensor, target=target)
        
        return attributions
    
    def gradient_shap_attribution(
        self,
        input_tensor: torch.Tensor,
        baseline_tensor: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Gradient SHAP attribution.
        
        Args:
            input_tensor: Input tensor to explain.
            baseline_tensor: Baseline tensor for SHAP.
            target: Target tensor (optional).
            
        Returns:
            Attribution tensor.
        """
        input_tensor = input_tensor.to(self.device)
        baseline_tensor = baseline_tensor.to(self.device)
        if target is not None:
            target = target.to(self.device)
        
        attributions = self.gradient_shap.attribute(
            input_tensor,
            baselines=baseline_tensor,
            target=target
        )
        
        return attributions
    
    def explain_prediction(
        self,
        input_tensor: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        method: str = "integrated_gradients"
    ) -> Dict[str, torch.Tensor]:
        """Explain a single prediction.
        
        Args:
            input_tensor: Input tensor to explain.
            target: Target tensor (optional).
            method: Attribution method to use.
            
        Returns:
            Dictionary with attribution results.
        """
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        if target is not None:
            target = target.unsqueeze(0)
        
        results = {}
        
        if method == "integrated_gradients":
            results["attributions"] = self.integrated_gradients_attribution(
                input_tensor, target
            )
        elif method == "saliency":
            results["attributions"] = self.saliency_attribution(
                input_tensor, target
            )
        elif method == "gradient_shap":
            # Create baseline
            baseline = torch.zeros_like(input_tensor)
            results["attributions"] = self.gradient_shap_attribution(
                input_tensor, baseline, target
            )
        else:
            raise ValueError(f"Unknown attribution method: {method}")
        
        # Get prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
            results["prediction"] = prediction
        
        return results
    
    def visualize_attribution(
        self,
        input_tensor: torch.Tensor,
        attributions: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Visualize attribution results.
        
        Args:
            input_tensor: Original input tensor.
            attributions: Attribution tensor.
            target: Target tensor (optional).
            save_path: Optional path to save the plot.
        """
        # Convert to numpy
        input_np = input_tensor.squeeze().cpu().numpy()
        attributions_np = attributions.squeeze().cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original input
        axes[0].plot(input_np, label="Input", color='blue', alpha=0.7)
        axes[0].set_title("Original Input")
        axes[0].set_xlabel("Time Step")
        axes[0].set_ylabel("Value")
        axes[0].grid(True, alpha=0.3)
        
        # Attributions
        axes[1].bar(range(len(attributions_np)), attributions_np, 
                   color='red', alpha=0.7)
        axes[1].set_title("Feature Attributions")
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("Attribution Score")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class AttentionVisualizer:
    """Visualize attention patterns in transformer models."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """Initialize the attention visualizer.
        
        Args:
            model: Transformer model to visualize.
            device: Device to run visualization on.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        
    def get_attention_weights(
        self,
        input_tensor: torch.Tensor,
        layer_idx: int = 0
    ) -> torch.Tensor:
        """Get attention weights from a specific layer.
        
        Args:
            input_tensor: Input tensor.
            layer_idx: Index of the transformer layer.
            
        Returns:
            Attention weights tensor.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass with attention weights
            # This assumes the model has a method to return attention weights
            if hasattr(self.model, 'get_attention_weights'):
                attention_weights = self.model.get_attention_weights(
                    input_tensor, layer_idx
                )
            else:
                # Fallback: create dummy attention weights
                seq_len = input_tensor.shape[1]
                attention_weights = torch.eye(seq_len).unsqueeze(0).to(self.device)
        
        return attention_weights
    
    def visualize_attention(
        self,
        input_tensor: torch.Tensor,
        attention_weights: torch.Tensor,
        save_path: Optional[str] = None
    ) -> None:
        """Visualize attention weights as a heatmap.
        
        Args:
            input_tensor: Input tensor.
            attention_weights: Attention weights tensor.
            save_path: Optional path to save the plot.
        """
        # Convert to numpy
        attention_np = attention_weights.squeeze().cpu().numpy()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention_np,
            cmap='Blues',
            square=True,
            cbar=True,
            xticklabels=False,
            yticklabels=False
        )
        plt.title("Attention Weights Heatmap")
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class FeatureImportanceAnalyzer:
    """Analyze feature importance across multiple samples."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """Initialize the feature importance analyzer.
        
        Args:
            model: Model to analyze.
            device: Device to run analysis on.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        
        self.explainer = ModelExplainer(model, device)
    
    def analyze_feature_importance(
        self,
        data_loader: torch.utils.data.DataLoader,
        n_samples: int = 100,
        method: str = "integrated_gradients"
    ) -> Dict[str, np.ndarray]:
        """Analyze feature importance across multiple samples.
        
        Args:
            data_loader: Data loader for analysis.
            n_samples: Number of samples to analyze.
            method: Attribution method to use.
            
        Returns:
            Dictionary with importance analysis results.
        """
        all_attributions = []
        all_predictions = []
        all_targets = []
        
        sample_count = 0
        
        for data, target in data_loader:
            if sample_count >= n_samples:
                break
            
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Get attributions for each sample in the batch
            for i in range(data.shape[0]):
                if sample_count >= n_samples:
                    break
                
                single_input = data[i:i+1]
                single_target = target[i:i+1]
                
                # Get explanation
                explanation = self.explainer.explain_prediction(
                    single_input, single_target, method
                )
                
                all_attributions.append(
                    explanation["attributions"].squeeze().cpu().numpy()
                )
                all_predictions.append(
                    explanation["prediction"].squeeze().cpu().numpy()
                )
                all_targets.append(single_target.squeeze().cpu().numpy())
                
                sample_count += 1
        
        # Convert to numpy arrays
        attributions_array = np.array(all_attributions)
        predictions_array = np.array(all_predictions)
        targets_array = np.array(all_targets)
        
        # Calculate statistics
        mean_importance = np.mean(np.abs(attributions_array), axis=0)
        std_importance = np.std(np.abs(attributions_array), axis=0)
        
        results = {
            "attributions": attributions_array,
            "mean_importance": mean_importance,
            "std_importance": std_importance,
            "predictions": predictions_array,
            "targets": targets_array
        }
        
        return results
    
    def plot_feature_importance(
        self,
        importance_results: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature importance analysis.
        
        Args:
            importance_results: Results from analyze_feature_importance.
            save_path: Optional path to save the plot.
        """
        mean_importance = importance_results["mean_importance"]
        std_importance = importance_results["std_importance"]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        x_pos = range(len(mean_importance))
        
        plt.bar(x_pos, mean_importance, yerr=std_importance, 
               capsize=5, alpha=0.7, color='steelblue')
        
        plt.title("Feature Importance Analysis")
        plt.xlabel("Time Step")
        plt.ylabel("Mean Absolute Attribution")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def explain_model_predictions(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    n_explanations: int = 5,
    method: str = "integrated_gradients",
    device: Optional[torch.device] = None
) -> List[Dict[str, Any]]:
    """Generate explanations for model predictions.
    
    Args:
        model: Model to explain.
        data_loader: Data loader for explanations.
        n_explanations: Number of explanations to generate.
        method: Attribution method to use.
        device: Device to run explanations on.
        
    Returns:
        List of explanation dictionaries.
    """
    explainer = ModelExplainer(model, device)
    explanations = []
    
    sample_count = 0
    
    for data, target in data_loader:
        if sample_count >= n_explanations:
            break
        
        data = data.to(device or get_device())
        target = target.to(device or get_device())
        
        for i in range(data.shape[0]):
            if sample_count >= n_explanations:
                break
            
            single_input = data[i:i+1]
            single_target = target[i:i+1]
            
            # Get explanation
            explanation = explainer.explain_prediction(
                single_input, single_target, method
            )
            
            explanations.append({
                "input": single_input.squeeze().cpu().numpy(),
                "target": single_target.squeeze().cpu().numpy(),
                "prediction": explanation["prediction"].squeeze().cpu().numpy(),
                "attributions": explanation["attributions"].squeeze().cpu().numpy()
            })
            
            sample_count += 1
    
    return explanations
