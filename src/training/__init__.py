"""Training utilities for disease outbreak prediction models."""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from ..utils import get_device, EarlyStopping
from ..models import create_model


logger = logging.getLogger(__name__)


class Trainer:
    """Training class for disease outbreak prediction models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """Initialize the trainer.
        
        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            device: Device to train on.
            config: Training configuration.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Training parameters
        self.learning_rate = config.get("learning_rate", 0.001)
        self.weight_decay = config.get("weight_decay", 1e-5)
        self.max_epochs = config.get("max_epochs", 100)
        self.patience = config.get("patience", 10)
        self.min_delta = config.get("min_delta", 0.001)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.loss_fn = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.patience,
            min_delta=self.min_delta
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        logger.info(f"Initialized trainer with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / num_batches
    
    def validate_epoch(self) -> float:
        """Validate for one epoch.
        
        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.loss_fn(output, target)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / num_batches
    
    def train(self) -> Dict[str, List[float]]:
        """Train the model.
        
        Returns:
            Training history.
        """
        logger.info("Starting training...")
        
        for epoch in range(self.max_epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.max_epochs}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            
            # Early stopping check
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model state")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss
        }
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint.
        
        Args:
            filepath: Path to load checkpoint from.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        logger.info(f"Loaded checkpoint from {filepath}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: Optional[torch.device] = None
) -> Trainer:
    """Train a model with the given configuration.
    
    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.
        device: Device to train on.
        
    Returns:
        Trained trainer object.
    """
    if device is None:
        device = get_device()
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # Train the model
    history = trainer.train()
    
    return trainer


def train_multiple_models(
    model_configs: List[Dict[str, Any]],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: Optional[torch.device] = None
) -> List[Trainer]:
    """Train multiple models with different configurations.
    
    Args:
        model_configs: List of model configurations.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Device to train on.
        
    Returns:
        List of trained trainers.
    """
    if device is None:
        device = get_device()
    
    trainers = []
    
    for i, config in enumerate(model_configs):
        logger.info(f"Training model {i+1}/{len(model_configs)}: {config['model_type']}")
        
        # Create model
        model = create_model(**config["model_params"])
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config["training_params"]
        )
        
        # Train model
        trainer.train()
        trainers.append(trainer)
        
        logger.info(f"Completed training model {i+1}")
    
    return trainers


def hyperparameter_search(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    search_space: Dict[str, Any],
    n_trials: int = 50,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Perform hyperparameter search using Optuna.
    
    Args:
        model_type: Type of model to optimize.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        search_space: Hyperparameter search space.
        n_trials: Number of trials to run.
        device: Device to train on.
        
    Returns:
        Best hyperparameters found.
    """
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna not available, skipping hyperparameter search")
        return {}
    
    if device is None:
        device = get_device()
    
    def objective(trial):
        # Sample hyperparameters
        params = {}
        for param_name, param_config in search_space.items():
            if param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
            elif param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name, param_config["low"], param_config["high"]
                )
        
        # Create model with sampled parameters
        model = create_model(model_type=model_type, **params)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config={"max_epochs": 20, "patience": 5}  # Reduced for search
        )
        
        # Train model
        trainer.train()
        
        return trainer.best_val_loss
    
    # Create study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Best validation loss: {study.best_value}")
    
    return study.best_params
