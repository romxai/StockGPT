"""
Comprehensive evaluation suite with Walk-Forward CV, backtesting, statistical tests, and multi-sector evaluation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class WalkForwardValidator:
    """Walk-Forward Cross-Validation with expanding window."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.wf_config = config['evaluation']['walk_forward']
        self.initial_train_size = self.wf_config['initial_train_size']
        self.step_size = self.wf_config['step_size']
        self.expanding_window = self.wf_config['expanding_window']
        
    def create_splits(self, data_length: int) -> List[Tuple[List[int], List[int]]]:
        """
        Create walk-forward splits.
        
        Args:
            data_length: Total length of the dataset
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []
        
        current_train_end = self.initial_train_size
        
        while current_train_end + self.step_size <= data_length:
            if self.expanding_window:
                # Expanding window: train from beginning to current_train_end
                train_indices = list(range(0, current_train_end))
            else:
                # Rolling window: train from (current_train_end - initial_train_size) to current_train_end
                train_start = max(0, current_train_end - self.initial_train_size)
                train_indices = list(range(train_start, current_train_end))
            
            # Test indices
            test_start = current_train_end
            test_end = min(current_train_end + self.step_size, data_length)
            test_indices = list(range(test_start, test_end))
            
            if len(test_indices) > 0:
                splits.append((train_indices, test_indices))
            
            current_train_end += self.step_size
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def validate_model(self, model_class, train_data: Dict, test_data: Dict, 
                      config: Dict) -> Dict[str, Any]:
        """
        Perform walk-forward validation for a single model.
        
        Args:
            model_class: Model class to instantiate
            train_data: Training data dictionary
            test_data: Test data dictionary
            config: Model configuration
            
        Returns:
            Validation results dictionary
        """
        from ..training.trainer import StockPredictorTrainer, StockDataset
        
        # Prepare full dataset
        X_text = np.concatenate([train_data['text_embeddings'], test_data['text_embeddings']])
        X_num = np.concatenate([train_data['numerical_features'], test_data['numerical_features']])
        y = np.concatenate([train_data['targets'], test_data['targets']])
        
        # Create splits
        splits = self.create_splits(len(y))
        
        fold_results = []
        
        for fold_idx, (train_indices, test_indices) in enumerate(splits):
            logger.info(f"Processing fold {fold_idx + 1}/{len(splits)}")
            
            # Create datasets for this fold
            train_dataset = StockDataset(
                X_text[train_indices],
                X_num[train_indices],
                y[train_indices]
            )
            
            test_dataset = StockDataset(
                X_text[test_indices],
                X_num[test_indices],
                y[test_indices]
            )
            
            # Initialize and train model
            model = model_class(config)
            trainer = StockPredictorTrainer(model, config)
            
            # Reduce epochs for faster validation
            trainer.epochs = min(10, trainer.epochs)
            
            # Train model
            train_results = trainer.train(train_dataset, test_dataset)
            
            # Evaluate on test set
            test_metrics = self._evaluate_predictions(
                trainer.model, test_dataset, trainer.device
            )
            
            fold_results.append({
                'fold': fold_idx,
                'train_size': len(train_indices),
                'test_size': len(test_indices),
                'train_results': train_results,
                'test_metrics': test_metrics
            })
        
        # Aggregate results
        aggregated_results = self._aggregate_fold_results(fold_results)
        
        return {
            'fold_results': fold_results,
            'aggregated_results': aggregated_results,
            'n_folds': len(splits)
        }
    
    def _evaluate_predictions(self, model: nn.Module, dataset: Any, device: torch.device) -> Dict[str, float]:
        """Evaluate model predictions on a dataset."""
        from torch.utils.data import DataLoader
        from ..training.trainer import collate_fn
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        with torch.no_grad():
            for batch in dataloader:
                text_embeddings = batch['text_embeddings'].to(device)
                numerical_features = batch['numerical_features'].to(device)
                targets = batch['targets'].to(device)
                
                temporal_info = batch.get('temporal_info')
                if temporal_info:
                    temporal_info = {k: v.to(device) for k, v in temporal_info.items()}
                
                outputs = model(text_embeddings, numerical_features, temporal_info)
                probabilities = torch.softmax(outputs['logits'], dim=1)
                _, predicted = torch.max(outputs['logits'], 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(all_targets, all_predictions)
        
        # ROC AUC for multiclass
        try:
            roc_auc = roc_auc_score(all_targets, all_probabilities, multi_class='ovr', average='weighted')
        except ValueError:
            roc_auc = 0.0
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'mcc': mcc
        }
    
    def _aggregate_fold_results(self, fold_results: List[Dict]) -> Dict[str, float]:
        """Aggregate results across folds."""
        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc', 'mcc']
        
        aggregated = {}
        for metric in metrics:
            values = [fold['test_metrics'][metric] for fold in fold_results]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
        
        return aggregated


class BacktestingSimulator:
    """Enhanced backtesting simulator with transaction costs and slippage."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.trading_config = config['evaluation']['trading']
        self.transaction_cost = self.trading_config['transaction_cost']
        self.slippage = self.trading_config['slippage']
        self.initial_capital = self.trading_config['initial_capital']
        self.position_size = self.trading_config['position_size']
        
    def backtest_strategy(self, predictions: np.ndarray, actual_returns: np.ndarray,
                         prices: np.ndarray, dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Backtest trading strategy based on model predictions.
        
        Args:
            predictions: Model predictions (0=Down, 1=Neutral, 2=Up)
            actual_returns: Actual returns for the period
            prices: Stock prices
            dates: Corresponding dates
            
        Returns:
            Backtesting results dictionary
        """
        if len(predictions) != len(actual_returns) or len(predictions) != len(prices):
            raise ValueError("All input arrays must have the same length")
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # Current position (-1, 0, 1)
        portfolio_values = [capital]
        positions = [0]
        trades = []
        
        for i in range(len(predictions)):
            current_price = prices[i]
            prediction = predictions[i]
            actual_return = actual_returns[i]
            
            # Determine target position based on prediction
            if prediction == 2:  # Up prediction
                target_position = 1
            elif prediction == 0:  # Down prediction
                target_position = -1
            else:  # Neutral prediction
                target_position = 0
            
            # Execute trade if position change is needed
            if target_position != position:
                trade_size = target_position - position
                trade_value = abs(trade_size) * current_price * self.position_size * capital
                
                # Apply transaction costs and slippage
                total_cost = trade_value * (self.transaction_cost + self.slippage)
                capital -= total_cost
                
                # Record trade
                trades.append({
                    'date': dates[i],
                    'action': 'BUY' if trade_size > 0 else 'SELL',
                    'size': abs(trade_size),
                    'price': current_price,
                    'cost': total_cost,
                    'position_before': position,
                    'position_after': target_position
                })
                
                position = target_position
            
            # Calculate portfolio value change
            if position != 0:
                position_return = position * actual_return
                capital *= (1 + position_return * self.position_size)
            
            portfolio_values.append(capital)
            positions.append(position)
        
        # Calculate performance metrics
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        metrics = self._calculate_trading_metrics(
            portfolio_returns, portfolio_values, dates, trades
        )
        
        results = {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'positions': positions,
            'trades': trades,
            'metrics': metrics,
            'final_capital': capital,
            'total_return': (capital - self.initial_capital) / self.initial_capital
        }
        
        return results
    
    def _calculate_trading_metrics(self, returns: np.ndarray, values: List[float],
                                  dates: pd.DatetimeIndex, trades: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive trading performance metrics."""
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (values[-1] - values[0]) / values[0]
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        winning_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades
        }


class StatisticalTester:
    """Statistical significance testing for model comparisons."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.significance_config = config['evaluation']['significance']
        self.alpha = self.significance_config['alpha']
        self.bootstrap_samples = self.significance_config['bootstrap_samples']
        
    def paired_t_test(self, model1_scores: np.ndarray, model2_scores: np.ndarray) -> Dict[str, float]:
        """Perform paired t-test between two models."""
        if len(model1_scores) != len(model2_scores):
            raise ValueError("Score arrays must have the same length")
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
        
        # Effect size (Cohen's d)
        differences = model1_scores - model2_scores
        effect_size = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < self.alpha,
            'mean_difference': np.mean(differences)
        }
    
    def wilcoxon_test(self, model1_scores: np.ndarray, model2_scores: np.ndarray) -> Dict[str, float]:
        """Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test)."""
        if len(model1_scores) != len(model2_scores):
            raise ValueError("Score arrays must have the same length")
        
        # Perform Wilcoxon signed-rank test
        stat, p_value = stats.wilcoxon(model1_scores, model2_scores, alternative='two-sided')
        
        return {
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
    
    def bootstrap_confidence_interval(self, scores: np.ndarray, 
                                    confidence_level: float = 0.95) -> Tuple[float, float, float]:
        """Calculate bootstrap confidence interval for mean score."""
        n_samples = len(scores)
        bootstrap_means = []
        
        for _ in range(self.bootstrap_samples):
            bootstrap_sample = np.random.choice(scores, size=n_samples, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        mean_score = np.mean(scores)
        
        return mean_score, ci_lower, ci_upper
    
    def compare_models(self, model_results: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compare multiple models with statistical tests."""
        model_names = list(model_results.keys())
        n_models = len(model_names)
        
        # Pairwise comparisons
        pairwise_results = {}
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                model1_name = model_names[i]
                model2_name = model_names[j]
                
                comparison_key = f"{model1_name}_vs_{model2_name}"
                
                # Paired t-test
                t_test_result = self.paired_t_test(
                    model_results[model1_name],
                    model_results[model2_name]
                )
                
                # Wilcoxon test
                wilcoxon_result = self.wilcoxon_test(
                    model_results[model1_name],
                    model_results[model2_name]
                )
                
                pairwise_results[comparison_key] = {
                    'paired_t_test': t_test_result,
                    'wilcoxon_test': wilcoxon_result
                }
        
        # Bootstrap confidence intervals for each model
        confidence_intervals = {}
        for model_name, scores in model_results.items():
            mean_score, ci_lower, ci_upper = self.bootstrap_confidence_interval(scores)
            confidence_intervals[model_name] = {
                'mean': mean_score,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
        
        return {
            'pairwise_comparisons': pairwise_results,
            'confidence_intervals': confidence_intervals
        }


class MultiSectorEvaluator:
    """Multi-sector evaluation for generalization assessment."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sectors = config['evaluation']['sectors']
        
    def evaluate_by_sector(self, model_class, data_by_symbol: Dict[str, Any], 
                          config: Dict) -> Dict[str, Any]:
        """
        Evaluate model performance across different sectors.
        
        Args:
            model_class: Model class to evaluate
            data_by_symbol: Data dictionary organized by symbol
            config: Model configuration
            
        Returns:
            Sector evaluation results
        """
        sector_results = {}
        
        for sector_name, symbols in self.sectors.items():
            logger.info(f"Evaluating sector: {sector_name}")
            
            # Collect data for this sector
            sector_data = {}
            for symbol in symbols:
                if symbol in data_by_symbol:
                    sector_data[symbol] = data_by_symbol[symbol]
            
            if not sector_data:
                logger.warning(f"No data found for sector {sector_name}")
                continue
            
            # Evaluate sector
            sector_performance = self._evaluate_sector(model_class, sector_data, config)
            sector_results[sector_name] = sector_performance
        
        # Calculate cross-sector statistics
        cross_sector_stats = self._calculate_cross_sector_stats(sector_results)
        
        return {
            'sector_results': sector_results,
            'cross_sector_stats': cross_sector_stats
        }
    
    def _evaluate_sector(self, model_class, sector_data: Dict[str, Any], 
                        config: Dict) -> Dict[str, Any]:
        """Evaluate model on a single sector."""
        # This would implement sector-specific evaluation
        # For brevity, returning placeholder structure
        return {
            'accuracy': np.random.uniform(0.5, 0.9),  # Placeholder
            'f1_score': np.random.uniform(0.4, 0.8),  # Placeholder
            'n_symbols': len(sector_data),
            'symbol_performance': {symbol: np.random.uniform(0.4, 0.9) 
                                 for symbol in sector_data.keys()}
        }
    
    def _calculate_cross_sector_stats(self, sector_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate statistics across sectors."""
        if not sector_results:
            return {}
        
        # Collect accuracy scores across sectors
        accuracies = [result['accuracy'] for result in sector_results.values()]
        f1_scores = [result['f1_score'] for result in sector_results.values()]
        
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'cv_accuracy': np.std(accuracies) / np.mean(accuracies) if np.mean(accuracies) > 0 else 0,
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'cv_f1': np.std(f1_scores) / np.mean(f1_scores) if np.mean(f1_scores) > 0 else 0,
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies)
        }


class AblationStudy:
    """Ablation study to validate component contributions."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.components = config['evaluation']['ablation']['components']
        
    def run_ablation_study(self, base_model_class, data: Dict[str, Any], 
                          config: Dict) -> Dict[str, Any]:
        """
        Run ablation study by removing components one by one.
        
        Args:
            base_model_class: Base model class
            data: Training/validation data
            config: Model configuration
            
        Returns:
            Ablation study results
        """
        results = {}
        
        # Baseline (full model)
        logger.info("Evaluating baseline (full model)")
        baseline_performance = self._evaluate_model_variant(
            base_model_class, data, config, removed_components=[]
        )
        results['baseline'] = baseline_performance
        
        # Ablated variants
        for component in self.components:
            logger.info(f"Evaluating model without {component}")
            ablated_performance = self._evaluate_model_variant(
                base_model_class, data, config, removed_components=[component]
            )
            results[f'without_{component}'] = ablated_performance
        
        # Calculate component contributions
        contributions = self._calculate_contributions(results)
        
        return {
            'results': results,
            'contributions': contributions
        }
    
    def _evaluate_model_variant(self, model_class, data: Dict[str, Any], 
                               config: Dict, removed_components: List[str]) -> Dict[str, float]:
        """Evaluate a specific model variant."""
        # Create modified config
        modified_config = config.copy()
        
        for component in removed_components:
            if component == 'finbert_embeddings':
                # Disable FinBERT embeddings (use zeros)
                modified_config['use_finbert'] = False
            elif component == 'temporal_weights':
                # Disable temporal weighting
                modified_config['features']['temporal']['learn_temporal_weights'] = False
            elif component == 'attention_mechanism':
                # Disable attention
                modified_config['model']['attention']['use_self_attention'] = False
                modified_config['model']['attention']['use_cross_attention'] = False
            elif component == 'market_context':
                # Disable market context features
                modified_config['use_market_context'] = False
            elif component == 'technical_indicators':
                # Disable technical indicators
                modified_config['use_technical_indicators'] = False
        
        # Placeholder evaluation (in practice, would train and evaluate model)
        return {
            'accuracy': np.random.uniform(0.4, 0.8),  # Placeholder
            'f1_score': np.random.uniform(0.3, 0.7)   # Placeholder
        }
    
    def _calculate_contributions(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate the contribution of each component."""
        baseline_accuracy = results['baseline']['accuracy']
        contributions = {}
        
        for key, performance in results.items():
            if key != 'baseline':
                component_name = key.replace('without_', '')
                contribution = baseline_accuracy - performance['accuracy']
                contributions[component_name] = contribution
        
        return contributions


def main():
    """Test the evaluation functionality."""
    import yaml
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test walk-forward validation
    wf_validator = WalkForwardValidator(config)
    splits = wf_validator.create_splits(1000)
    print(f"Created {len(splits)} walk-forward splits")
    
    # Test backtesting
    backtester = BacktestingSimulator(config)
    
    # Generate dummy data
    n_days = 252
    predictions = np.random.randint(0, 3, n_days)
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = 100 * np.cumprod(1 + returns)
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    backtest_results = backtester.backtest_strategy(predictions, returns, prices, dates)
    print(f"Backtest completed. Final return: {backtest_results['total_return']:.2%}")
    
    # Test statistical testing
    stat_tester = StatisticalTester(config)
    model1_scores = np.random.normal(0.75, 0.05, 20)
    model2_scores = np.random.normal(0.72, 0.05, 20)
    
    t_test_result = stat_tester.paired_t_test(model1_scores, model2_scores)
    print(f"T-test p-value: {t_test_result['p_value']:.4f}")


if __name__ == "__main__":
    main()