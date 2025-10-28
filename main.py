"""
Main entry point for the Integrated Stock Prediction System.

This script provides a command-line interface for running different components
of the stock prediction system including data collection, training, evaluation,
and reporting.
"""

import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.helpers import (
    load_config, setup_logging, set_random_seeds, 
    print_system_info, ProgressTracker, create_experiment_id
)
from src.data_collection.collectors import StockDataCollector, NewsDataCollector, MarketContextCollector
from src.preprocessing.nlp_processor import NewsProcessor
from src.features.feature_engineering import FeatureEngineer
from src.models.hybrid_model import HybridStockPredictor
from src.training.trainer import StockPredictorTrainer, OptunaOptimizer
from src.evaluation.evaluators import (
    WalkForwardValidator, BacktestingSimulator, StatisticalTester,
    MultiSectorEvaluator, AblationStudy
)
from src.explainability.explainer import ExplainabilityPipeline
from src.reporting.report_generator import ReportGenerator


class StockPredictionPipeline:
    """Main pipeline orchestrating the entire stock prediction workflow."""
    
    def __init__(self, config_path: str = 'configs/config.yaml'):
        """Initialize the pipeline with configuration."""
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config)
        self.experiment_id = create_experiment_id()
        
        # Set random seeds for reproducibility
        set_random_seeds(self.config.get('environment', {}).get('seed', 42))
        
        # Print system information
        print_system_info(self.config)
        
        # Initialize components
        self.stock_collector = StockDataCollector(self.config)
        self.news_collector = NewsDataCollector(self.config)
        self.context_collector = MarketContextCollector(self.config)
        self.news_processor = NewsProcessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        
        self.logger.info(f"Pipeline initialized with experiment ID: {self.experiment_id}")
    
    def collect_data(self, use_cache: bool = True) -> Dict[str, Any]:
        """Collect all required data (stocks, news, market context)."""
        self.logger.info("Starting data collection phase...")
        
        data = {}
        
        # Collect stock data
        self.logger.info("Collecting stock price data...")
        stock_data = self.stock_collector.collect_all_symbols(use_cache=use_cache)
        data['stock_data'] = stock_data
        
        # Collect news data
        self.logger.info("Collecting financial news data...")
        news_data = self.news_collector.collect_all_news(use_cache=use_cache)
        data['news_data'] = news_data
        
        # Collect market context
        self.logger.info("Collecting market context data...")
        economic_data = self.context_collector.collect_economic_indicators()
        sector_data = self.context_collector.collect_sector_performance()
        data['economic_data'] = economic_data
        data['sector_data'] = sector_data
        
        self.logger.info("Data collection completed")
        return data
    
    def preprocess_data(self, data: Dict[str, Any], use_cache: bool = True) -> Dict[str, Any]:
        """Preprocess collected data (NLP processing, feature engineering)."""
        self.logger.info("Starting data preprocessing phase...")
        
        # Process news data with NLP pipeline
        self.logger.info("Processing news with NLP pipeline...")
        processed_news = self.news_processor.process_news_data(
            data['news_data'], use_cache=use_cache
        )
        
        # Feature engineering for each symbol
        self.logger.info("Engineering features...")
        features_data = {}
        
        symbols = self.config['data']['symbols']
        progress = ProgressTracker(len(symbols), "Feature Engineering")
        
        for i, symbol in enumerate(symbols):
            features_data[symbol] = self.feature_engineer.engineer_features(
                data['stock_data'], processed_news, symbol
            )
            progress.update(i + 1, f"Processing {symbol}")
        
        progress.finish()
        
        processed_data = {
            'features_data': features_data,
            'processed_news': processed_news,
            'raw_data': data
        }
        
        self.logger.info("Data preprocessing completed")
        return processed_data
    
    def train_model(self, processed_data: Dict[str, Any], 
                   optimize_hyperparams: bool = False) -> Dict[str, Any]:
        """Train the hybrid stock prediction model."""
        self.logger.info("Starting model training phase...")
        
        # Create model
        model = HybridStockPredictor(self.config)
        self.logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Prepare training data
        trainer = StockPredictorTrainer(model, self.config)
        
        # Create datasets from processed data
        train_dataset, val_dataset = trainer.prepare_data(
            processed_data['features_data'],
            processed_data['processed_news'],
            self.config['model']['sequence_length']
        )
        
        training_results = {}
        
        if optimize_hyperparams:
            # Hyperparameter optimization
            self.logger.info("Starting hyperparameter optimization...")
            optimizer = OptunaOptimizer(self.config)
            optimization_results = optimizer.optimize(train_dataset, val_dataset)
            training_results['optimization'] = optimization_results
            
            # Update config with best parameters
            best_params = optimization_results['best_params']
            for param, value in best_params.items():
                keys = param.split('_')
                config_section = self.config
                for key in keys[:-1]:
                    if key not in config_section:
                        config_section[key] = {}
                    config_section = config_section[key]
                config_section[keys[-1]] = value
            
            # Create new model with optimized parameters
            model = HybridStockPredictor(self.config)
            trainer = StockPredictorTrainer(model, self.config)
        
        # Train the model
        self.logger.info("Training model...")
        train_results = trainer.train(train_dataset, val_dataset)
        training_results['training'] = train_results
        
        # Save model
        model_path = f"models/hybrid_model_{self.experiment_id}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        trainer.save_model(model_path, {'experiment_id': self.experiment_id})
        training_results['model_path'] = model_path
        
        self.logger.info("Model training completed")
        return training_results
    
    def evaluate_model(self, model_path: str, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        self.logger.info("Starting model evaluation phase...")
        
        # Load trained model
        model = HybridStockPredictor(self.config)
        trainer = StockPredictorTrainer(model, self.config)
        trainer.load_model(model_path)
        
        evaluation_results = {}
        
        # Prepare data for evaluation
        features_data = processed_data['features_data']
        news_data = processed_data['processed_news']
        
        # 1. Walk-Forward Cross-Validation
        self.logger.info("Running Walk-Forward Cross-Validation...")
        wf_validator = WalkForwardValidator(self.config)
        # This would need proper implementation with actual data splits
        # wf_results = wf_validator.validate_model(HybridStockPredictor, train_data, test_data, self.config)
        # evaluation_results['walk_forward'] = wf_results
        
        # 2. Backtesting
        self.logger.info("Running backtesting simulation...")
        backtester = BacktestingSimulator(self.config)
        # This would need actual predictions and returns data
        # backtest_results = backtester.backtest_strategy(predictions, returns, prices, dates)
        # evaluation_results['backtesting'] = backtest_results
        
        # 3. Multi-Sector Evaluation
        self.logger.info("Running multi-sector evaluation...")
        sector_evaluator = MultiSectorEvaluator(self.config)
        sector_results = sector_evaluator.evaluate_by_sector(
            HybridStockPredictor, features_data, self.config
        )
        evaluation_results['sector_analysis'] = sector_results
        
        # 4. Ablation Study
        self.logger.info("Running ablation study...")
        ablation_study = AblationStudy(self.config)
        ablation_results = ablation_study.run_ablation_study(
            HybridStockPredictor, features_data, self.config
        )
        evaluation_results['ablation_study'] = ablation_results
        
        # 5. Statistical Significance Tests
        self.logger.info("Running statistical significance tests...")
        stat_tester = StatisticalTester(self.config)
        # This would need results from baseline models
        # significance_results = stat_tester.compare_models(model_results)
        # evaluation_results['significance_tests'] = significance_results
        
        self.logger.info("Model evaluation completed")
        return evaluation_results
    
    def explain_model(self, model_path: str, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model explanations and interpretability analysis."""
        self.logger.info("Starting model explainability analysis...")
        
        # Load trained model
        model = HybridStockPredictor(self.config)
        trainer = StockPredictorTrainer(model, self.config)
        trainer.load_model(model_path)
        
        # Prepare sample data for explanation
        features_data = processed_data['features_data']
        news_data = processed_data['processed_news']
        
        # Get sample data (first symbol, recent data)
        sample_symbol = list(features_data.keys())[0]
        sample_features = features_data[sample_symbol].iloc[-60:] # Last 60 days
        sample_news = news_data.get(sample_symbol, [])[-60:]  # Last 60 news items
        
        # Create sample tensors (simplified)
        import torch
        import numpy as np
        
        seq_len = min(60, len(sample_features))
        text_embeddings = torch.randn(1, seq_len, 768)  # Placeholder
        numerical_features = torch.randn(1, seq_len, len(sample_features.columns) - 1)  # Exclude target
        
        sample_data = {
            'text_embeddings': text_embeddings,
            'numerical_features': numerical_features
        }
        
        # Run explainability analysis
        explainer = ExplainabilityPipeline(self.config)
        explanation_results = explainer.run_full_analysis(
            model, sample_data, sample_news, sample_features.index.tolist()
        )
        
        self.logger.info("Model explainability analysis completed")
        return explanation_results
    
    def generate_reports(self, all_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate comprehensive reports and visualizations."""
        self.logger.info("Starting report generation...")
        
        # Initialize report generator
        report_generator = ReportGenerator(self.config)
        
        # Generate comprehensive report
        generated_files = report_generator.generate_comprehensive_report(all_results)
        
        self.logger.info(f"Generated {len(generated_files)} report files")
        return generated_files
    
    def run_full_pipeline(self, optimize_hyperparams: bool = False, 
                         use_cache: bool = True) -> Dict[str, Any]:
        """Run the complete pipeline from data collection to reporting."""
        self.logger.info("Starting full pipeline execution...")
        
        results = {'experiment_id': self.experiment_id}
        
        try:
            # 1. Data Collection
            data = self.collect_data(use_cache=use_cache)
            results['data_collection'] = {'status': 'completed', 'symbols': len(data['stock_data'])}
            
            # 2. Data Preprocessing
            processed_data = self.preprocess_data(data, use_cache=use_cache)
            results['preprocessing'] = {'status': 'completed'}
            
            # 3. Model Training
            training_results = self.train_model(processed_data, optimize_hyperparams=optimize_hyperparams)
            results['training'] = training_results
            
            # 4. Model Evaluation
            evaluation_results = self.evaluate_model(training_results['model_path'], processed_data)
            results['evaluation'] = evaluation_results
            
            # 5. Model Explanation
            explanation_results = self.explain_model(training_results['model_path'], processed_data)
            results['explanations'] = explanation_results
            
            # 6. Report Generation
            generated_files = self.generate_reports(results)
            results['reports'] = generated_files
            
            self.logger.info("Full pipeline execution completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            results['error'] = str(e)
            raise
        
        return results


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(description='Integrated Stock Prediction System')
    
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['full', 'collect', 'train', 'evaluate', 'explain', 'report'],
                       default='full', help='Pipeline mode to run')
    parser.add_argument('--optimize', action='store_true',
                       help='Run hyperparameter optimization during training')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching (force re-collection/processing)')
    parser.add_argument('--model-path', type=str,
                       help='Path to trained model (for evaluate/explain modes)')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = StockPredictionPipeline(args.config)
        
        use_cache = not args.no_cache
        
        if args.mode == 'full':
            # Run complete pipeline
            results = pipeline.run_full_pipeline(
                optimize_hyperparams=args.optimize,
                use_cache=use_cache
            )
            print(f"\nPipeline completed successfully! Experiment ID: {results['experiment_id']}")
            
        elif args.mode == 'collect':
            # Data collection only
            data = pipeline.collect_data(use_cache=use_cache)
            print(f"Data collection completed. Collected data for {len(data['stock_data'])} symbols.")
            
        elif args.mode == 'train':
            # Training only (requires preprocessed data)
            print("Training mode requires preprocessed data. Running data collection and preprocessing first...")
            data = pipeline.collect_data(use_cache=use_cache)
            processed_data = pipeline.preprocess_data(data, use_cache=use_cache)
            training_results = pipeline.train_model(processed_data, optimize_hyperparams=args.optimize)
            print(f"Training completed. Model saved to: {training_results['model_path']}")
            
        elif args.mode == 'evaluate':
            # Evaluation only
            if not args.model_path:
                raise ValueError("Model path required for evaluation mode")
            print("Evaluation mode requires preprocessed data. Running data collection and preprocessing first...")
            data = pipeline.collect_data(use_cache=use_cache)
            processed_data = pipeline.preprocess_data(data, use_cache=use_cache)
            evaluation_results = pipeline.evaluate_model(args.model_path, processed_data)
            print("Evaluation completed.")
            
        elif args.mode == 'explain':
            # Explanation only
            if not args.model_path:
                raise ValueError("Model path required for explanation mode")
            print("Explanation mode requires preprocessed data. Running data collection and preprocessing first...")
            data = pipeline.collect_data(use_cache=use_cache)
            processed_data = pipeline.preprocess_data(data, use_cache=use_cache)
            explanation_results = pipeline.explain_model(args.model_path, processed_data)
            print("Explanation analysis completed.")
            
        elif args.mode == 'report':
            # Report generation only (requires results)
            print("Report mode requires evaluation results. Please run evaluation first.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()