"""Streamlit demo application for disease outbreak prediction."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import set_seed, get_device
from src.data import OutbreakDataset, generate_synthetic_data
from src.models import create_model
from src.training import train_model
from src.evaluation import ModelEvaluator, UncertaintyQuantifier
from src.explainability import ModelExplainer


# Page configuration
st.set_page_config(
    page_title="Disease Outbreak Prediction",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🦠 Disease Outbreak Prediction</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <ul>
        <li>❌ <strong>NOT FOR CLINICAL USE</strong> - This software is not intended for diagnostic or clinical decision-making</li>
        <li>❌ <strong>NOT MEDICAL ADVICE</strong> - Predictions should not be used as medical advice</li>
        <li>✅ <strong>RESEARCH ONLY</strong> - Designed for research, education, and demonstration purposes</li>
        <li>👨‍⚕️ <strong>CLINICIAN SUPERVISION REQUIRED</strong> - Any healthcare use requires appropriate clinical supervision</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["lstm", "gru", "transformer", "cnn_lstm"],
    help="Choose the neural network architecture"
)

# Data parameters
st.sidebar.subheader("Data Parameters")
seq_len = st.sidebar.slider("Sequence Length", 5, 30, 10, help="Number of time steps to use for prediction")
pred_len = st.sidebar.slider("Prediction Length", 1, 5, 1, help="Number of future time steps to predict")

# Model parameters
st.sidebar.subheader("Model Parameters")
hidden_size = st.sidebar.slider("Hidden Size", 16, 128, 64, help="Size of hidden layers")
num_layers = st.sidebar.slider("Number of Layers", 1, 4, 2, help="Number of recurrent layers")
dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, help="Dropout probability")

# Training parameters
st.sidebar.subheader("Training Parameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
epochs = st.sidebar.slider("Epochs", 10, 200, 50, help="Number of training epochs")
batch_size = st.sidebar.slider("Batch Size", 8, 64, 32)

# Data generation parameters
st.sidebar.subheader("Synthetic Data Parameters")
n_samples = st.sidebar.slider("Number of Samples", 500, 2000, 1000)
seasonality = st.sidebar.checkbox("Include Seasonality", True)
trend = st.sidebar.checkbox("Include Trend", True)
outbreak_prob = st.sidebar.slider("Outbreak Probability", 0.01, 0.2, 0.05)
outbreak_magnitude = st.sidebar.slider("Outbreak Magnitude", 1.0, 5.0, 2.0)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Model Training", "📈 Predictions", "🔍 Explainability"])

with tab1:
    st.header("Data Overview")
    
    # Generate synthetic data
    if st.button("Generate New Data", key="generate_data"):
        with st.spinner("Generating synthetic data..."):
            data = generate_synthetic_data(
                n_samples=n_samples,
                seasonality=seasonality,
                trend=trend,
                noise_level=0.1,
                outbreak_prob=outbreak_prob,
                outbreak_magnitude=outbreak_magnitude
            )
            st.session_state.data = data
    
    if 'data' not in st.session_state:
        st.session_state.data = generate_synthetic_data(
            n_samples=n_samples,
            seasonality=seasonality,
            trend=trend,
            noise_level=0.1,
            outbreak_prob=outbreak_prob,
            outbreak_magnitude=outbreak_magnitude
        )
    
    data = st.session_state.data
    
    # Data statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{np.mean(data):.2f}")
    with col2:
        st.metric("Std", f"{np.std(data):.2f}")
    with col3:
        st.metric("Min", f"{np.min(data):.2f}")
    with col4:
        st.metric("Max", f"{np.max(data):.2f}")
    
    # Time series plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=data,
        mode='lines',
        name='Disease Cases',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Disease Outbreak Time Series",
        xaxis_title="Time (days)",
        yaxis_title="Number of Cases",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data distribution
    fig_hist = px.histogram(
        x=data,
        title="Distribution of Case Counts",
        labels={'x': 'Number of Cases', 'y': 'Frequency'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.header("Model Training")
    
    if st.button("Train Model", key="train_model"):
        with st.spinner("Training model..."):
            # Set random seed
            set_seed(42)
            
            # Create dataset
            dataset = OutbreakDataset(
                data=data,
                seq_len=seq_len,
                pred_len=pred_len,
                normalize=True,
                normalization_method="minmax"
            )
            
            # Create data loaders
            from src.data import OutbreakDataLoader
            data_loader = OutbreakDataLoader(
                batch_size=batch_size,
                shuffle=True
            )
            
            train_loader, val_loader, test_loader = data_loader.create_loaders(
                dataset,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15
            )
            
            # Create model
            device = get_device()
            model = create_model(
                model_type=model_type,
                input_size=1,
                output_size=pred_len,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
            
            # Training configuration
            training_config = {
                "learning_rate": learning_rate,
                "max_epochs": epochs,
                "patience": 10,
                "min_delta": 0.001
            }
            
            # Train model
            trainer = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=training_config,
                device=device
            )
            
            # Store in session state
            st.session_state.model = model
            st.session_state.dataset = dataset
            st.session_state.test_loader = test_loader
            st.session_state.trainer = trainer
            
            st.success("Model training completed!")
    
    if 'model' in st.session_state:
        st.success("✅ Model is ready for predictions!")
        
        # Training history
        if hasattr(st.session_state.trainer, 'train_losses'):
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=st.session_state.trainer.train_losses,
                mode='lines',
                name='Training Loss',
                line=dict(color='blue')
            ))
            fig_loss.add_trace(go.Scatter(
                y=st.session_state.trainer.val_losses,
                mode='lines',
                name='Validation Loss',
                line=dict(color='red')
            ))
            
            fig_loss.update_layout(
                title="Training History",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=400
            )
            
            st.plotly_chart(fig_loss, use_container_width=True)

with tab3:
    st.header("Predictions")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first!")
    else:
        # Get predictions
        evaluator = ModelEvaluator(st.session_state.model)
        metrics = evaluator.evaluate(
            st.session_state.test_loader,
            denormalize_fn=st.session_state.dataset.denormalize_predictions
        )
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"{metrics['mae']:.4f}")
        with col2:
            st.metric("RMSE", f"{metrics['rmse']:.4f}")
        with col3:
            st.metric("MAPE", f"{metrics['mape']:.2f}%")
        with col4:
            st.metric("R²", f"{metrics['r2']:.4f}")
        
        # Get predictions for plotting
        predictions, targets = evaluator.predict(
            st.session_state.test_loader,
            denormalize_fn=st.session_state.dataset.denormalize_predictions
        )
        
        # Prediction vs Actual plot
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            y=targets[:100],
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        fig_pred.add_trace(go.Scatter(
            y=predictions[:100],
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=2)
        ))
        
        fig_pred.update_layout(
            title="Predictions vs Actual Values",
            xaxis_title="Time",
            yaxis_title="Number of Cases",
            height=500
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Scatter plot
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=targets,
            y=predictions,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', opacity=0.6)
        ))
        
        # Add perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig_scatter.update_layout(
            title="Predicted vs Actual Scatter Plot",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            height=500
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Uncertainty quantification
        st.subheader("Uncertainty Quantification")
        
        uncertainty_quantifier = UncertaintyQuantifier(st.session_state.model)
        mean_pred, uncertainty = uncertainty_quantifier.monte_carlo_dropout(
            st.session_state.test_loader,
            n_samples=50
        )
        
        # Uncertainty plot
        fig_uncertainty = go.Figure()
        
        # Add uncertainty band
        fig_uncertainty.add_trace(go.Scatter(
            y=mean_pred[:50] + uncertainty[:50],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_uncertainty.add_trace(go.Scatter(
            y=mean_pred[:50] - uncertainty[:50],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            name='Uncertainty',
            hoverinfo='skip'
        ))
        
        fig_uncertainty.add_trace(go.Scatter(
            y=targets[:50],
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        fig_uncertainty.add_trace(go.Scatter(
            y=mean_pred[:50],
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=2)
        ))
        
        fig_uncertainty.update_layout(
            title="Predictions with Uncertainty",
            xaxis_title="Time",
            yaxis_title="Number of Cases",
            height=500
        )
        
        st.plotly_chart(fig_uncertainty, use_container_width=True)

with tab4:
    st.header("Model Explainability")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first!")
    else:
        st.subheader("Feature Attribution Analysis")
        
        # Get a sample for explanation
        sample_data, sample_target = next(iter(st.session_state.test_loader))
        sample_input = sample_data[0:1]  # Take first sample
        sample_target_val = sample_target[0:1]
        
        # Create explainer
        explainer = ModelExplainer(st.session_state.model)
        
        # Get attributions
        explanation = explainer.explain_prediction(
            sample_input,
            sample_target_val,
            method="integrated_gradients"
        )
        
        attributions = explanation["attributions"].squeeze().cpu().numpy()
        input_values = sample_input.squeeze().cpu().numpy()
        
        # Attribution plot
        fig_attr = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Input Time Series", "Feature Attributions"),
            vertical_spacing=0.1
        )
        
        fig_attr.add_trace(
            go.Scatter(
                y=input_values,
                mode='lines',
                name='Input',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig_attr.add_trace(
            go.Bar(
                x=list(range(len(attributions))),
                y=attributions,
                name='Attributions',
                marker_color=['red' if x > 0 else 'blue' for x in attributions]
            ),
            row=2, col=1
        )
        
        fig_attr.update_layout(
            height=600,
            showlegend=False
        )
        
        fig_attr.update_xaxes(title_text="Time Step", row=2, col=1)
        fig_attr.update_yaxes(title_text="Value", row=1, col=1)
        fig_attr.update_yaxes(title_text="Attribution Score", row=2, col=1)
        
        st.plotly_chart(fig_attr, use_container_width=True)
        
        # Attribution statistics
        st.subheader("Attribution Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Attribution", f"{np.max(attributions):.4f}")
        with col2:
            st.metric("Min Attribution", f"{np.min(attributions):.4f}")
        with col3:
            st.metric("Mean |Attribution|", f"{np.mean(np.abs(attributions)):.4f}")
        
        # Most important time steps
        st.subheader("Most Important Time Steps")
        
        # Get indices sorted by absolute attribution
        sorted_indices = np.argsort(np.abs(attributions))[::-1]
        
        importance_df = pd.DataFrame({
            'Time Step': sorted_indices[:10],
            'Attribution Score': attributions[sorted_indices[:10]],
            'Input Value': input_values[sorted_indices[:10]]
        })
        
        st.dataframe(importance_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><strong>Disease Outbreak Prediction Demo</strong> - Research and Educational Use Only</p>
    <p>⚠️ This is not a clinical tool and should not be used for medical decision-making</p>
</div>
""", unsafe_allow_html=True)
