# Integrated Stock Prediction System

A sophisticated stock prediction system that combines deep learning with Natural Language Processing (NLP) using FinBERT for sentiment analysis and market prediction.

## 🚀 Features

### Core Capabilities
- **Hybrid Deep Learning Architecture**: BiLSTM + Multi-Head Self-Attention + Cross-Modal Attention
- **Advanced NLP Integration**: FinBERT-based sentiment analysis for financial news
- **Comprehensive Technical Analysis**: 80+ technical indicators with learnable temporal weights
- **Multi-Source Data Collection**: Stock prices, financial news, market context
- **Walk-Forward Cross-Validation**: Realistic time-series evaluation methodology
- **Enhanced Backtesting**: Transaction costs, slippage, and market impact modeling
- **Explainable AI**: Attention visualization and feature importance analysis
- **Publication-Quality Reporting**: LaTeX tables and professional visualizations

### Technical Specifications
- **Python Version**: 3.8+
- **Deep Learning Framework**: PyTorch with CUDA support
- **NLP Model**: FinBERT (ProsusAI/finbert) for financial sentiment analysis
- **Data Sources**: Yahoo Finance (yfinance), Yahoo News RSS feeds
- **Architecture**: Hybrid model with separate branches for numerical and text features
- **Optimization**: Optuna hyperparameter optimization with AdamW optimizer

## 📁 Project Structure

```
StockGPT/
├── configs/
│   └── config.yaml                 # Central configuration file
├── src/
│   ├── data_collection/
│   │   └── collectors.py           # Stock and news data collection
│   ├── preprocessing/
│   │   └── nlp_processor.py        # FinBERT NLP processing pipeline
│   ├── features/
│   │   └── feature_engineering.py  # Technical indicators and features
│   ├── models/
│   │   └── hybrid_model.py         # Hybrid deep learning architecture
│   ├── training/
│   │   └── trainer.py              # Training and optimization pipeline
│   ├── evaluation/
│   │   └── evaluators.py           # Evaluation and backtesting suite
│   ├── explainability/
│   │   └── explainer.py            # XAI and visualization tools
│   ├── reporting/
│   │   └── report_generator.py     # LaTeX reporting and figures
│   └── utils/
│       └── helpers.py              # Utility functions and helpers
├── data/
│   ├── raw/                        # Raw data storage
│   └── processed/                  # Processed data storage
├── models/                         # Trained model storage
├── results/                        # Results and reports
├── logs/                           # Training and system logs
├── main.py                         # Main entry point
├── requirements.txt                # Python dependencies
├── setup_and_run.bat              # Complete setup and execution script
├── train_model.bat                 # Quick training script
├── run_predictions.bat             # Model prediction script
└── README.md                       # This file
```

## 🛠️ Installation & Setup

### Automated Setup (Recommended)
Run the automated setup script that handles everything:
```bash
setup_and_run.bat
```

This script will:
1. Create and activate a Python virtual environment
2. Upgrade pip and install all dependencies
3. Install all dependencies (spaCy removed for compatibility)
4. Create necessary directories
5. Present a menu for different execution modes

### Manual Setup
If you prefer manual installation:

1. **Create Virtual Environment**:
```bash
python -m venv venv
venv\Scripts\activate
```

2. **Install Dependencies**:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

   **If you encounter issues with some packages (like pyfolio compatibility), use the minimal requirements:**
```bash
pip install -r requirements_minimal.txt
```

3. **Note**: spaCy dependency removed for better compatibility - the system now uses regex-based entity extraction

4. **Create Directories**:
```bash
mkdir data\raw data\processed models results logs
```

### Troubleshooting Installation

If you encounter package compatibility issues:

1. **Use Python 3.8-3.11** (avoid Python 3.12+ for now due to some package compatibility)
2. **For PyTorch installation issues**, visit [PyTorch installation guide](https://pytorch.org/get-started/locally/)
3. **For package conflicts**, try installing with minimal requirements first:
   ```bash
   pip install -r requirements_minimal.txt
   ```

## 🎯 Usage

### Quick Start Options

1. **Full Pipeline Execution**:
```bash
python main.py --mode full --symbols AAPL,GOOGL,MSFT
```

2. **Data Collection Only**:
```bash
python main.py --mode data_collection --symbols AAPL,TSLA
```

3. **Training Only**:
```bash
python main.py --mode train --config configs/config.yaml
```

4. **Evaluation Only**:
```bash
python main.py --mode evaluate --model-path models/trained_model.pth
```

5. **Hyperparameter Optimization**:
```bash
python main.py --mode optimize --trials 100
```

### Batch File Shortcuts

- **setup_and_run.bat**: Complete setup and interactive menu
- **train_model.bat**: Quick training with default settings  
- **run_predictions.bat**: Load trained model and make predictions

### Interactive Menu (via setup_and_run.bat)
```
===============================================
Stock Prediction System
===============================================
1. Run Full Pipeline (Data + Training + Evaluation)
2. Data Collection Only
3. Training Only
4. Evaluation Only
5. Hyperparameter Optimization
6. Exit
```

## 📊 Model Architecture

### Hybrid Deep Learning Model
The system uses a sophisticated hybrid architecture:

1. **Numerical Feature Branch**:
   - Input: 80+ technical indicators
   - BiLSTM layers for temporal sequence modeling
   - Multi-head self-attention for feature importance
   - Learnable temporal weights

2. **Text Feature Branch**:
   - Input: FinBERT embeddings (768-dim)
   - BiLSTM layers for sequence processing
   - Multi-head self-attention

3. **Cross-Modal Fusion**:
   - Cross-modal attention between numerical and text features
   - Feature fusion and prediction head
   - Output: Stock price predictions

### Technical Indicators
- **Trend Indicators**: SMA, EMA, MACD, ADX
- **Momentum Indicators**: RSI, Stochastic, Williams %R
- **Volatility Indicators**: Bollinger Bands, ATR, Standard Deviation
- **Volume Indicators**: OBV, Volume SMA, VWAP
- **Market Context**: VIX, sector correlations, market regime

## 📈 Evaluation Methodology

### Walk-Forward Cross-Validation
- **Training Window**: 252 trading days (1 year)
- **Validation Window**: 63 trading days (3 months) 
- **Test Window**: 21 trading days (1 month)
- **Step Size**: 21 days (monthly updates)

### Performance Metrics
- **Regression**: RMSE, MAE, MAPE, R²
- **Classification**: Accuracy, Precision, Recall, F1-score
- **Financial**: Sharpe Ratio, Maximum Drawdown, Total Return
- **Statistical**: Diebold-Mariano test, Model Confidence Set

### Backtesting Features
- Transaction costs (0.1% per trade)
- Market impact and slippage modeling
- Position sizing and risk management
- Benchmark comparison (Buy & Hold, Market Index)

## 🔍 Explainable AI Features

### Attention Visualization
- Multi-head attention weight heatmaps
- Temporal attention patterns
- Cross-modal attention analysis

### Feature Importance
- SHAP value analysis
- Permutation importance
- Ablation study results

### Event Analysis
- News sentiment impact tracking
- Market event correlation
- Counterfactual analysis

## 📋 Configuration

### Main Configuration (configs/config.yaml)
```yaml
data:
  symbols: [AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA, NFLX, AMD, CRM, ADBE, PYPL, INTC, ORCL, IBM]
  
model:
  hidden_dim: 128
  num_layers: 2
  num_heads: 8
  dropout: 0.2
  
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10
```

### Supported Stocks
The system includes 15 major stocks across different sectors:
- **Technology**: AAPL, GOOGL, MSFT, AMZN, META, NVDA, AMD, INTC, ORCL, IBM
- **Services**: NFLX, CRM, ADBE, PYPL
- **Automotive**: TSLA

## 📊 Output & Results

### Generated Reports
- **Performance Summary**: Comprehensive metrics and statistics
- **Visualization Suite**: Interactive plots and attention heatmaps
- **LaTeX Tables**: Publication-ready performance tables
- **Backtesting Results**: Detailed trading simulation results

### File Outputs
- **Models**: `models/hybrid_model_YYYYMMDD_HHMMSS.pth`
- **Results**: `results/evaluation_report_YYYYMMDD_HHMMSS.json`
- **Figures**: `results/figures/` (PNG, HTML formats)
- **Reports**: `results/reports/` (LaTeX, PDF formats)

## 🔧 System Requirements

### Hardware Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space
- **GPU**: CUDA-compatible GPU recommended (optional)

### Software Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10/11 (batch files), Linux/macOS (manual setup)
- **Internet**: Required for data collection and model downloads

## 📞 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**:
   - Install PyTorch with CPU-only if no GPU available
   - Check CUDA compatibility with your GPU

2. **Memory Issues**:
   - Reduce batch size in config.yaml
   - Use gradient checkpointing for large models

3. **Data Collection Failures**:
   - Check internet connection
   - Verify Yahoo Finance API availability
   - Increase retry delays in collectors.py

4. **Package Compatibility**: spaCy has been removed to avoid numpy conflicts - entity extraction now uses regex patterns

### Performance Optimization
- Use GPU acceleration when available
- Optimize batch size based on available RAM
- Enable mixed precision training for faster execution
- Use data caching to avoid repeated downloads

## 📝 License & Citation

This is an academic research project. If you use this code in your research, please cite appropriately.

## 🤝 Contributing

This is a research implementation. For improvements or bug fixes, please ensure all changes maintain the academic rigor and evaluation standards of the original system.