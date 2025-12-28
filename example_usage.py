#!/usr/bin/env python3
"""Example usage of the Disease Outbreak Prediction system.

This script demonstrates how to use the modernized disease outbreak prediction
system for research and educational purposes.

IMPORTANT: This is for research and educational use only. NOT FOR CLINICAL USE.
"""

import logging
from pathlib import Path

from src.utils import set_seed, get_device, setup_logging
from src.data import OutbreakDataset, OutbreakDataLoader, generate_synthetic_data
from src.models import create_model
from src.training import train_model
from src.evaluation import evaluate_model_performance, create_evaluation_report


def main():
    """Main demonstration function."""
    # Setup
    set_seed(42)
    logger = setup_logging(level="INFO")
    device = get_device()
    
    logger.info("🦠 Disease Outbreak Prediction Demo")
    logger.info("=" * 50)
    logger.info("⚠️  RESEARCH AND EDUCATIONAL USE ONLY")
    logger.info("❌ NOT FOR CLINICAL USE")
    logger.info("=" * 50)
    
    # Generate synthetic data
    logger.info("Generating synthetic disease outbreak data...")
    data = generate_synthetic_data(
        n_samples=1000,
        seasonality=True,
        trend=True,
        noise_level=0.1,
        outbreak_prob=0.05,
        outbreak_magnitude=2.0
    )
    
    logger.info(f"Generated {len(data)} data points")
    logger.info(f"Mean: {data.mean():.2f}, Std: {data.std():.2f}")
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = OutbreakDataset(
        data=data,
        seq_len=10,
        pred_len=1,
        normalize=True,
        normalization_method="minmax"
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    data_loader = OutbreakDataLoader(
        batch_size=32,
        shuffle=True
    )
    
    train_loader, val_loader, test_loader = data_loader.create_loaders(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Train different models
    models_to_train = [
        ("lstm", {"hidden_size": 64, "num_layers": 2, "dropout": 0.2}),
        ("gru", {"hidden_size": 64, "num_layers": 2, "dropout": 0.2}),
        ("transformer", {"d_model": 64, "nhead": 8, "num_layers": 3, "dropout": 0.1})
    ]
    
    training_config = {
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "max_epochs": 50,
        "patience": 10,
        "min_delta": 0.001
    }
    
    trained_models = {}
    
    for model_type, model_params in models_to_train:
        logger.info(f"\nTraining {model_type.upper()} model...")
        
        # Create model
        model = create_model(
            model_type=model_type,
            input_size=1,
            output_size=1,
            **model_params
        )
        
        # Train model
        trainer = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            device=device
        )
        
        trained_models[model_type] = {
            "model": model,
            "trainer": trainer
        }
        
        logger.info(f"✅ {model_type.upper()} training completed")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    # Evaluate models
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    
    results = {}
    
    for model_type, model_data in trained_models.items():
        logger.info(f"\nEvaluating {model_type.upper()} model...")
        
        # Evaluate model
        eval_results = evaluate_model_performance(
            model=model_data["model"],
            test_loader=test_loader,
            denormalize_fn=dataset.denormalize_predictions,
            device=device
        )
        
        results[model_type] = eval_results
        
        # Print metrics
        metrics = eval_results["metrics"]
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"MAPE: {metrics['mape']:.2f}%")
        logger.info(f"R²: {metrics['r2']:.4f}")
        logger.info(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
    
    # Create evaluation reports
    logger.info("\nCreating evaluation reports...")
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    
    for model_type, eval_results in results.items():
        create_evaluation_report(
            results=eval_results,
            model_name=model_type,
            save_dir=str(output_dir)
        )
    
    # Find best model
    best_model = min(results.keys(), key=lambda k: results[k]["metrics"]["rmse"])
    logger.info(f"\n🏆 Best model: {best_model.upper()}")
    logger.info(f"RMSE: {results[best_model]['metrics']['rmse']:.4f}")
    
    logger.info("\n" + "=" * 50)
    logger.info("DEMO COMPLETED SUCCESSFULLY")
    logger.info("=" * 50)
    logger.info("📊 Check the 'assets/' directory for evaluation plots")
    logger.info("🚀 Run 'streamlit run demo/app.py' for interactive demo")
    logger.info("🌐 Run 'python service.py' for FastAPI service")
    logger.info("\n⚠️  REMEMBER: This is for research and educational use only!")
    logger.info("❌ NOT FOR CLINICAL USE OR MEDICAL DECISION-MAKING")


if __name__ == "__main__":
    main()
