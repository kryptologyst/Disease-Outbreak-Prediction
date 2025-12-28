"""FastAPI service for disease outbreak prediction."""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from src.utils import set_seed, get_device
from src.data import OutbreakDataset, generate_synthetic_data
from src.models import create_model
from src.training import train_model
from src.evaluation import ModelEvaluator


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Disease Outbreak Prediction API",
    description="Research and educational API for disease outbreak prediction using time series forecasting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model storage
models = {}


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    data: List[float] = Field(..., description="Time series data for prediction")
    model_type: str = Field(default="lstm", description="Model type to use")
    seq_len: int = Field(default=10, description="Sequence length")
    pred_len: int = Field(default=1, description="Prediction length")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: List[float] = Field(..., description="Predicted values")
    confidence: Optional[float] = Field(None, description="Confidence score")
    model_type: str = Field(..., description="Model type used")


class TrainingRequest(BaseModel):
    """Request model for training."""
    data: List[float] = Field(..., description="Training data")
    model_type: str = Field(default="lstm", description="Model type")
    seq_len: int = Field(default=10, description="Sequence length")
    pred_len: int = Field(default=1, description="Prediction length")
    epochs: int = Field(default=50, description="Training epochs")
    learning_rate: float = Field(default=0.001, description="Learning rate")


class TrainingResponse(BaseModel):
    """Response model for training."""
    model_id: str = Field(..., description="Trained model ID")
    training_loss: float = Field(..., description="Final training loss")
    validation_loss: float = Field(..., description="Final validation loss")
    epochs_trained: int = Field(..., description="Number of epochs trained")


class EvaluationRequest(BaseModel):
    """Request model for evaluation."""
    model_id: str = Field(..., description="Model ID to evaluate")
    test_data: List[float] = Field(..., description="Test data")


class EvaluationResponse(BaseModel):
    """Response model for evaluation."""
    metrics: Dict[str, float] = Field(..., description="Evaluation metrics")
    predictions: List[float] = Field(..., description="Predictions on test data")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Disease Outbreak Prediction API",
        "version": "1.0.0",
        "status": "active",
        "disclaimer": "This is a research and educational API only. NOT FOR CLINICAL USE."
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "device": str(get_device())}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using a trained model.
    
    Args:
        request: Prediction request containing data and model parameters.
        
    Returns:
        Prediction response with predicted values.
    """
    try:
        # Convert data to numpy array
        data = np.array(request.data)
        
        # Create dataset
        dataset = OutbreakDataset(
            data=data,
            seq_len=request.seq_len,
            pred_len=request.pred_len,
            normalize=True
        )
        
        # Create model if not exists
        model_key = f"{request.model_type}_{request.seq_len}_{request.pred_len}"
        
        if model_key not in models:
            model = create_model(
                model_type=request.model_type,
                input_size=1,
                output_size=request.pred_len
            )
            models[model_key] = model
        else:
            model = models[model_key]
        
        # Get the last sequence for prediction
        if len(dataset.X) == 0:
            raise HTTPException(status_code=400, detail="Insufficient data for prediction")
        
        last_sequence = dataset.X[-1:].unsqueeze(-1)  # Add feature dimension
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            prediction = model(torch.tensor(last_sequence, dtype=torch.float32))
            prediction = prediction.squeeze().numpy()
        
        # Denormalize prediction
        if request.pred_len == 1:
            prediction = dataset.denormalize_predictions(np.array([prediction]))[0]
        else:
            prediction = dataset.denormalize_predictions(prediction)
        
        return PredictionResponse(
            prediction=prediction.tolist(),
            model_type=request.model_type
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=TrainingResponse)
async def train_model_endpoint(request: TrainingRequest):
    """Train a new model.
    
    Args:
        request: Training request containing data and parameters.
        
    Returns:
        Training response with model ID and training results.
    """
    try:
        # Convert data to numpy array
        data = np.array(request.data)
        
        # Create dataset
        dataset = OutbreakDataset(
            data=data,
            seq_len=request.seq_len,
            pred_len=request.pred_len,
            normalize=True
        )
        
        # Create data loaders
        from src.data import OutbreakDataLoader
        data_loader = OutbreakDataLoader(batch_size=32)
        train_loader, val_loader, _ = data_loader.create_loaders(dataset)
        
        # Create model
        model = create_model(
            model_type=request.model_type,
            input_size=1,
            output_size=request.pred_len
        )
        
        # Training configuration
        training_config = {
            "learning_rate": request.learning_rate,
            "max_epochs": request.epochs,
            "patience": 10,
            "min_delta": 0.001
        }
        
        # Train model
        trainer = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            device=get_device()
        )
        
        # Store model
        model_id = f"{request.model_type}_{request.seq_len}_{request.pred_len}_{len(models)}"
        models[model_id] = model
        
        return TrainingResponse(
            model_id=model_id,
            training_loss=trainer.train_losses[-1] if trainer.train_losses else 0.0,
            validation_loss=trainer.val_losses[-1] if trainer.val_losses else 0.0,
            epochs_trained=len(trainer.train_losses)
        )
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_model(request: EvaluationRequest):
    """Evaluate a trained model.
    
    Args:
        request: Evaluation request containing model ID and test data.
        
    Returns:
        Evaluation response with metrics and predictions.
    """
    try:
        if request.model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model = models[request.model_id]
        
        # Create test dataset
        test_data = np.array(request.test_data)
        dataset = OutbreakDataset(
            data=test_data,
            seq_len=10,  # Default sequence length
            pred_len=1,   # Default prediction length
            normalize=True
        )
        
        # Create test data loader
        from src.data import OutbreakDataLoader
        data_loader = OutbreakDataLoader(batch_size=32, shuffle=False)
        _, _, test_loader = data_loader.create_loaders(dataset)
        
        # Evaluate model
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(test_loader, dataset.denormalize_predictions)
        
        # Get predictions
        predictions, _ = evaluator.predict(test_loader, dataset.denormalize_predictions)
        
        return EvaluationResponse(
            metrics=metrics,
            predictions=predictions.tolist()
        )
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List all trained models."""
    return {
        "models": list(models.keys()),
        "count": len(models)
    }


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a trained model.
    
    Args:
        model_id: ID of the model to delete.
    """
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    del models[model_id]
    return {"message": f"Model {model_id} deleted successfully"}


@app.get("/generate-synthetic-data")
async def generate_synthetic_data_endpoint(
    n_samples: int = 1000,
    seasonality: bool = True,
    trend: bool = True,
    outbreak_prob: float = 0.05,
    outbreak_magnitude: float = 2.0
):
    """Generate synthetic disease outbreak data.
    
    Args:
        n_samples: Number of samples to generate.
        seasonality: Whether to include seasonal patterns.
        trend: Whether to include trend component.
        outbreak_prob: Probability of outbreak occurrence.
        outbreak_magnitude: Magnitude multiplier for outbreaks.
        
    Returns:
        Generated synthetic data.
    """
    try:
        data = generate_synthetic_data(
            n_samples=n_samples,
            seasonality=seasonality,
            trend=trend,
            noise_level=0.1,
            outbreak_prob=outbreak_prob,
            outbreak_magnitude=outbreak_magnitude
        )
        
        return {
            "data": data.tolist(),
            "n_samples": len(data),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data))
        }
        
    except Exception as e:
        logger.error(f"Synthetic data generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Set random seed
    set_seed(42)
    
    # Run the API server
    uvicorn.run(
        "service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
