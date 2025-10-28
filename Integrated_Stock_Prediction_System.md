# Integrated Stock Prediction System

## 1. Introduction

### 1.1 Goal
To develop, train, and rigorously evaluate an advanced deep learning system for predicting short-term stock price directional movements (Up/Down/Neutral). The system will achieve high accuracy (target: >80%) by synergistically integrating financial news analysis (NLP using FinBERT) with comprehensive historical market data, featuring a novel hybrid architecture with learnable temporal parameters.

### 1.2 Description
This project creates an end-to-end pipeline encompassing enhanced data collection (including market context), sophisticated feature engineering, advanced NLP processing using FinBERT, a hybrid deep learning model (BiLSTM, Attention elements) with learnable temporal weights, automated hyperparameter optimization, and a comprehensive, publication-ready evaluation suite (Walk-Forward CV, enhanced backtesting, multi-sector analysis, statistical significance testing, ablation studies, explainability). The system prioritizes research reproducibility and insightful predictions.

### 1.3 Target Users
- Quantitative Finance Researchers  
- Machine Learning Practitioners in Finance  
- Academics publishing in ML/Finance domains  

---

## 2. System Pipeline & Features

The system follows a modular pipeline:

### 2.1 Enhanced Data Collection
- Collects daily OHLCV data for specified tickers using yfinance.  
- Collects financial news primarily via Yahoo Finance, using RSS feeds and potentially historical web scraping techniques. The system should be configurable to potentially include other sources later if needed.  
- Collects contextual data: market indices (S&P 500, VIX, Nasdaq), relevant sector ETFs, macro indicators (Treasury yields, DXY), and competitor stock data/news.  
- Targets extensive history (1000+ days recommended).  

### 2.2 NLP Processing
- Cleans and preprocesses news text (headlines/summaries primarily).  
- Uses FinBERT (ProsusAI/finbert) to generate 768-dim contextual embeddings and 3-class sentiment probabilities (Pos/Neg/Neu) for news items.  
- Performs NER (spaCy) and Event Classification (regex based on predefined financial event types).  

### 2.3 Feature Engineering
- Calculates a comprehensive set (target 80+) of numerical features: technical indicators, volatility, volume, market/sector/competitor correlations, macro indicators.  
- **Learnable Temporal Relevance:** Implements layers where event importance multipliers and temporal decay half-life are learned during training. These weights are applied primarily to news embeddings.  
- Calculates a daily aggregated, temporally weighted sentiment score (using learned weights) and includes it as an explicit numerical feature.  
- Ensures strict temporal alignment and prevention of lookahead bias during feature creation and sequence generation.  

### 2.4 Model Architecture (Hybrid)
- Employs an advanced Hybrid Model architecture.  
- Features separate BiLSTM branches for weighted text embeddings and numerical features.  
- Incorporates Multi-Head Self-Attention within branches and Cross-Modal Attention between branches.  
- Uses Late Fusion to combine learned representations.  
- Integrates the learnable temporal parameters into the model's forward pass.  

### 2.5 Training & Optimization
- Uses an enhanced training loop (AdamW, LR scheduling, gradient clipping, early stopping).  
- Leverages Optuna for systematic Bayesian hyperparameter optimization (50-100 trials recommended).  

### 2.6 Rigorous Evaluation
- Primary method: Walk-Forward Cross-Validation (expanding window).  
- Includes Enhanced Backtesting simulator with costs, slippage, etc..  
- Performs Statistical Significance Testing (bootstrap CIs, p-values, effect sizes).  
- Conducts Multi-Sector Evaluation for generalization assessment.  
- Includes Ablation Studies to validate component contributions.  

### 2.7 Explainability (XAI)
- Implements Attention Visualization, Temporal Heatmaps, Event Contribution Analysis, and Counterfactual Analysis.  

### 2.8 Automated Reporting
- Generates summary CSVs, an executive text summary, comprehensive JSON results, publication-quality figures, and LaTeX tables (IEEE format) automatically from evaluation outputs.  

---

## 3. Research Plan & Novelty

### 3.1 Research Question
Can a deep learning model, integrating rich contextual information from financial news (via FinBERT embeddings obtained primarily from Yahoo Finance sources) and comprehensive market data, significantly improve stock price directional prediction accuracy by utilizing a hybrid attention architecture with automatically learned temporal relevance parameters?

### 3.2 Key Novel Contributions
- End-to-end joint learning architecture combining FinBERT, BiLSTM, multi-head self/cross-attention.  
- Learnable temporal parameters (event weights, decay rate) optimized within the model.  
- Integration of a very broad feature set including market, macro, and competitor context.  
- Application of a comprehensive, publication-standard evaluation suite (Walk-Forward CV, enhanced backtesting, multi-sector, stats tests, ablation) within a unified framework.  
- Advanced, tailored explainability methods (temporal heatmaps, event contribution).  

---

## 4. Technology Stack

**Programming Language:** Python 3.8+  

**Core Libraries:**  
PyTorch (primary DL), Transformers (Hugging Face), Pandas, NumPy, Scikit-learn, spaCy, NLTK, yfinance, ta, Optuna, SciPy, Matplotlib, Seaborn, Plotly, PyYAML, Requests, BeautifulSoup4.  
Potentially TensorFlow/Keras for baseline comparisons.  

**External Data Sources:**  
Yahoo Finance (via yfinance library, RSS feeds, and potentially scraping).  

---

## 5. Evaluation Plan

**Primary Metric:** Directional Accuracy on the test set(s) derived from Walk-Forward CV.  

**Secondary Metrics:** F1-Score, Precision, Recall, ROC-AUC, MCC.  

**Trading Metrics:** Sharpe Ratio, CAGR, Max Drawdown, Win Rate (from enhanced backtesting).  

**Validation Strategy:** Walk-Forward CV (Expanding Window).  

**Baselines:** Compare against Random Forest, Logistic Regression, Simple LSTM, BiLSTM+Attention, Transformer, and potentially a FinBERT+SimpleLSTM pipeline.  

**Significance:** Use paired t-tests/Wilcoxon tests (p < 0.05) and bootstrap CIs to validate improvements over baselines.  

**Generalization:** Evaluate on 15+ stocks across 5 sectors; measure mean performance and cross-sector variance (CV).  

**Robustness:** Analyze performance across different market volatility regimes (using VIX).  

**Component Validation:** Conduct ablation studies removing key architectural/feature components (e.g., learnable params, attention, enhanced features).  

---

## 6. Output Requirements (for Research Paper)

### Figures
- Model Accuracy Comparison (Bar Chart)  
- Training/Validation Curves (Line Plot)  
- Confusion Matrices (Heatmap Grid)  
- Ablation Study Contributions (Waterfall/Bar Chart)  
- Feature Importance (Bar Chart)  
- Multi-Stock/Sector Performance (Box/Violin Plot)  
- Hyperparameter Optimization History (Scatter/Line Plot)  
- Sequence Length Analysis (Line Plot)  
- Manual: System Architecture Diagram  
- Manual: Detailed Model Architecture Diagram  

### Tables (LaTeX format)
- Comprehensive Model Performance Comparison  
- Ablation Study Results  
- Multi-Stock/Sector Results  
- Optimal Hyperparameters Found by Optuna  
- Feature Category Performance/Importance  
- Statistical Significance Test Results  
- Computational Efficiency Comparison  
- Detailed Confusion Matrix Metrics  
- Training Time Breakdown  
- Error Analysis by Market Condition  
- Dataset Statistics  

---

## 7. Proposed File Structure

```
stock_predictor/
├── configs/             # config.yaml
├── data/                # Raw, Processed, Cache (gitignore)
├── examples/            # Runnable component demos
├── models/              # Saved model checkpoints (gitignore)
├── results/             # Evaluation outputs, reports (gitignore)
│   ├── walk_forward/
│   ├── ablation_study/
│   ├── multi_sector/
│   ├── significance_tests/
│   ├── optimization/
│   ├── explanations/
│   └── reports/         # Final CSV, TXT, JSON, LaTeX tables
│       └── paper/         # Final Figures & Tables for paper
│           ├── figures/
│           └── tables/
├── research_paper/      # LaTeX source, references.bib
├── scripts/             # High-level execution scripts
├── src/                 # Core source code modules
│   ├── data_collection/ # Includes Yahoo scraping/RSS logic
│   ├── preprocessing/
│   ├── features/        # Includes learnable temporal
│   ├── models/          # Includes hybrid architecture
│   ├── training/        # Includes Optuna optimizer
│   ├── evaluation/      # Includes WF-CV, Stats tests, etc.
│   ├── explainability/
│   ├── reporting/       # Includes LaTeX generation
│   └── utils/
├── tests/               # Unit/integration tests
├── venv/                # Virtual environment (gitignore)
├── .env                 # API Keys (if any needed later, gitignore)
├── .gitignore
├── main.py              # Main CLI entry point
├── requirements.txt
├── README.md
├── health_check.py
└── RUN_EXPERIMENTS.sh   # Script to run all paper experiments
```
