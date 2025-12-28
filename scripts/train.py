"""Main training script for disease outbreak prediction."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.utils import set_seed, get_device, setup_logging
from src.data import OutbreakDataset, OutbreakDataLoader, generate_synthetic_data
from src.models import create_model
from src.training import train_model
from src.evaluation import evaluate_model_performance, create_evaluation_report


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train disease outbreak prediction model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model-specific configuration file"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to real data CSV file (if not provided, synthetic data will be used)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> DictConfig:
    """Load and merge configuration files."""
    # Load base config
    config = OmegaConf.load(args.config)
    
    # Load model-specific config if provided
    if args.model_config:
        model_config = OmegaConf.load(args.model_config)
        config = OmegaConf.merge(config, model_config)
    
    # Override with command line arguments
    if args.seed is not None:
        config.seed = args.seed
    
    if args.device != "auto":
        config.device.auto_detect = False
        config.device[f"force_{args.device}"] = True
    
    if args.wandb:
        config.logging.use_wandb = True
    
    return config


def setup_device(config: DictConfig) -> torch.device:
    """Setup device based on configuration."""
    if not config.device.auto_detect:
        if config.device.force_cpu:
            return torch.device("cpu")
        elif config.device.force_cuda:
            return torch.device("cuda")
        elif config.device.force_mps:
            return torch.device("mps")
    
    return get_device()


def prepare_data(config: DictConfig, data_path: str = None) -> tuple:
    """Prepare training data."""
    if data_path and os.path.exists(data_path):
        # Load real data
        from src.data import load_real_data
        data = load_real_data(data_path)
    else:
        # Generate synthetic data
        data = generate_synthetic_data(
            n_samples=1000,
            seasonality=True,
            trend=True,
            noise_level=0.1,
            outbreak_prob=0.05,
            outbreak_magnitude=2.0
        )
    
    # Create dataset
    dataset = OutbreakDataset(
        data=data,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        normalize=config.data.normalize,
        normalization_method=config.data.normalization_method
    )
    
    # Create data loader
    data_loader = OutbreakDataLoader(
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    # Create train/val/test loaders
    train_loader, val_loader, test_loader = data_loader.create_loaders(
        dataset,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio
    )
    
    return train_loader, val_loader, test_loader, dataset


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args)
    
    # Set random seed
    set_seed(config.seed)
    
    # Setup logging
    logger = setup_logging(
        level=config.logging.level,
        log_file=config.logging.log_file
    )
    
    # Setup device
    device = setup_device(config)
    logger.info(f"Using device: {device}")
    
    # Setup Weights & Biases if requested
    if config.logging.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.logging.wandb_project,
                config=OmegaConf.to_container(config, resolve=True)
            )
            logger.info("Initialized Weights & Biases logging")
        except ImportError:
            logger.warning("Weights & Biases not available, skipping logging")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    logger.info("Preparing data...")
    train_loader, val_loader, test_loader, dataset = prepare_data(config, args.data_path)
    
    # Create model
    logger.info(f"Creating {config.model.type} model...")
    model = create_model(
        model_type=config.model.type,
        input_size=config.model.input_size,
        output_size=config.model.output_size,
        **{k: v for k, v in config.model.items() if k not in ["type", "input_size", "output_size"]}
    )
    
    # Train model
    logger.info("Starting training...")
    trainer = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        device=device
    )
    
    # Save model checkpoint
    checkpoint_path = output_dir / "best_model.pth"
    trainer.save_checkpoint(str(checkpoint_path))
    
    # Evaluate model
    logger.info("Evaluating model...")
    results = evaluate_model_performance(
        model=model,
        test_loader=test_loader,
        denormalize_fn=dataset.denormalize_predictions,
        device=device
    )
    
    # Create evaluation report
    create_evaluation_report(
        results=results,
        model_name=config.model.type,
        save_dir=str(output_dir)
    )
    
    # Save configuration
    config_path = output_dir / "config.yaml"
    OmegaConf.save(config, str(config_path))
    
    logger.info(f"Training completed. Results saved to {output_dir}")
    
    # Close wandb if used
    if config.logging.use_wandb:
        try:
            wandb.finish()
        except:
            pass


if __name__ == "__main__":
    main()
