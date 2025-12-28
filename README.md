# Disease Outbreak Prediction - Healthcare AI Research Project

## DISCLAIMER

**IMPORTANT: This is a research and educational demonstration project only.**

- **NOT FOR CLINICAL USE**: This software is not intended for diagnostic, therapeutic, or clinical decision-making purposes
- **NOT MEDICAL ADVICE**: The predictions and outputs should not be used as medical advice
- **RESEARCH ONLY**: This project is designed for research, education, and demonstration purposes
- **NO WARRANTY**: No warranty is provided regarding accuracy, reliability, or fitness for any purpose
- **CLINICIAN SUPERVISION**: Any use in healthcare settings requires appropriate clinical supervision and validation

## Overview

This project implements time series forecasting models for disease outbreak prediction using historical epidemiological data. The system supports multiple model architectures including LSTM, GRU, Transformer, and traditional time series models.

## Features

- **Multiple Model Architectures**: LSTM, GRU, Transformer, ARIMA, Prophet
- **Comprehensive Evaluation**: Time series specific metrics, calibration, uncertainty quantification
- **Explainability**: SHAP values, attention maps, feature importance
- **Interactive Demo**: Streamlit application for model exploration
- **Production Ready**: FastAPI service, structured logging, configuration management
- **Research Focus**: Synthetic data generation, ablation studies, bias analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Disease-Outbreak-Prediction.git
cd Disease-Outbreak-Prediction

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models import LSTMPredictor
from src.data import OutbreakDataset
from src.train import train_model

# Load data
dataset = OutbreakDataset.load_synthetic_data()

# Train model
model = LSTMPredictor(input_size=1, hidden_size=64)
trainer = train_model(model, dataset, epochs=50)

# Make predictions
predictions = model.predict(dataset.test_data)
```

### Run Demo

```bash
streamlit run demo/app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data loading and preprocessing
│   ├── training/          # Training loops and utilities
│   ├── evaluation/        # Metrics and evaluation
│   ├── explainability/    # Explainability methods
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── data/                  # Data storage
├── scripts/               # Training and evaluation scripts
├── tests/                 # Unit tests
├── demo/                  # Streamlit demo application
├── assets/                # Generated plots and results
└── docs/                  # Documentation
```

## Models

### Deep Learning Models
- **LSTM**: Long Short-Term Memory networks for sequential prediction
- **GRU**: Gated Recurrent Units with reduced complexity
- **Transformer**: Attention-based models for long-range dependencies
- **CNN-LSTM**: Hybrid convolutional and recurrent architecture

### Traditional Models
- **ARIMA**: AutoRegressive Integrated Moving Average
- **Prophet**: Facebook's time series forecasting tool
- **Exponential Smoothing**: Holt-Winters methods
- **Gradient Boosting**: XGBoost and LightGBM for tabular features

## Evaluation Metrics

- **Forecasting Accuracy**: MAE, RMSE, MAPE, SMAPE
- **Directional Accuracy**: Percentage of correct trend predictions
- **Calibration**: Reliability diagrams, Brier score
- **Uncertainty**: Prediction intervals, confidence bounds
- **Bias Analysis**: Performance across different demographic groups

## Configuration

The project uses OmegaConf for configuration management. Key configuration files:

- `configs/default.yaml`: Default model and training parameters
- `configs/models/`: Model-specific configurations
- `configs/data/`: Data preprocessing configurations

## Data

### Synthetic Data
The project includes synthetic disease outbreak data generation for demonstration purposes.

### Real Data Support
The system supports various epidemiological data formats:
- CSV time series data
- WHO/CDC data formats
- Custom epidemiological datasets

## Development

### Code Quality
- **Formatting**: Black for code formatting
- **Linting**: Ruff for code quality
- **Type Checking**: MyPy for static type analysis
- **Testing**: Pytest for unit tests

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## API Reference

### Models
- `LSTMPredictor`: LSTM-based outbreak predictor
- `GRUPredictor`: GRU-based outbreak predictor
- `TransformerPredictor`: Transformer-based outbreak predictor

### Data
- `OutbreakDataset`: Dataset class for outbreak data
- `DataLoader`: Data loading utilities

### Training
- `train_model()`: Model training function
- `evaluate_model()`: Model evaluation function

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{disease_outbreak_prediction,
  title={Disease Outbreak Prediction: A Healthcare AI Research Project},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Disease-Outbreak-Prediction}
}
```

## Contact

For questions or support, please open an issue on GitHub or contact the research team.
# Disease-Outbreak-Prediction
