"""
Training and optimization module with AdamW, scheduling, and Optuna hyperparameter optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import pickle
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping mechanism to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            score: Current validation score (higher is better)
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        
        return False


class LearningRateScheduler:
    """Custom learning rate scheduler with warmup and cosine annealing."""
    
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, 
                 max_steps: int, base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class StockDataset:
    """Dataset class for stock prediction data."""
    
    def __init__(self, text_embeddings: np.ndarray, numerical_features: np.ndarray,
                 targets: np.ndarray, temporal_info: Optional[Dict[str, np.ndarray]] = None):
        self.text_embeddings = torch.FloatTensor(text_embeddings)
        self.numerical_features = torch.FloatTensor(numerical_features)
        self.targets = torch.LongTensor(targets)
        
        if temporal_info:
            self.temporal_info = {
                key: torch.FloatTensor(value) for key, value in temporal_info.items()
            }
        else:
            self.temporal_info = None
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        item = {
            'text_embeddings': self.text_embeddings[idx],
            'numerical_features': self.numerical_features[idx],
            'targets': self.targets[idx]
        }
        
        if self.temporal_info:
            item['temporal_info'] = {
                key: value[idx] for key, value in self.temporal_info.items()
            }
        
        return item


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    text_embeddings = torch.stack([item['text_embeddings'] for item in batch])
    numerical_features = torch.stack([item['numerical_features'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    
    result = {
        'text_embeddings': text_embeddings,
        'numerical_features': numerical_features,
        'targets': targets
    }
    
    # Handle temporal info if present
    if 'temporal_info' in batch[0]:
        temporal_info = {}
        for key in batch[0]['temporal_info'].keys():
            temporal_info[key] = torch.stack([item['temporal_info'][key] for item in batch])
        result['temporal_info'] = temporal_info
    
    return result


class StockPredictorTrainer:
    """Main training class for stock prediction model."""
    
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training parameters
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']
        self.gradient_clip_norm = config['training']['gradient_clip_norm']
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            warmup_steps=config['training']['warmup_steps'],
            max_steps=config['training']['max_steps'],
            base_lr=self.learning_rate
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping_patience']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision training
        self.use_mixed_precision = config['training']['mixed_precision']
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
    
    def prepare_data(self, features_data: Dict[str, pd.DataFrame], 
                    news_data: Dict[str, List[Dict]], 
                    sequence_length: int) -> Tuple[StockDataset, StockDataset]:
        """
        Prepare training and validation datasets.
        
        Args:
            features_data: Dictionary of feature DataFrames per symbol
            news_data: Dictionary of processed news data per symbol
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        all_sequences = []
        all_targets = []
        all_text_embeddings = []
        all_temporal_info = []
        
        for symbol in features_data.keys():
            if symbol not in news_data:
                continue
            
            df = features_data[symbol]
            news = news_data[symbol]
            
            # Create sequences
            sequences, targets, text_emb, temporal = self._create_sequences(
                df, news, sequence_length
            )
            
            if len(sequences) > 0:
                all_sequences.extend(sequences)
                all_targets.extend(targets)
                all_text_embeddings.extend(text_emb)
                all_temporal_info.extend(temporal)
        
        if not all_sequences:
            raise ValueError("No valid sequences created from data")
        
        # Convert to arrays
        X_num = np.array(all_sequences)
        X_text = np.array(all_text_embeddings)
        y = np.array(all_targets)
        
        # Temporal info
        temporal_dict = None
        if all_temporal_info and all_temporal_info[0]:
            temporal_dict = {
                'days_ago': np.array([t['days_ago'] for t in all_temporal_info]),
                'event_categories': np.array([t['event_categories'] for t in all_temporal_info])
            }
        
        # Train/validation split
        val_split = self.config['training']['val_split']
        split_idx = int(len(X_num) * (1 - val_split))
        
        # Training data
        train_dataset = StockDataset(
            X_text[:split_idx],
            X_num[:split_idx],
            y[:split_idx],
            {k: v[:split_idx] for k, v in temporal_dict.items()} if temporal_dict else None
        )
        
        # Validation data
        val_dataset = StockDataset(
            X_text[split_idx:],
            X_num[split_idx:],
            y[split_idx:],
            {k: v[split_idx:] for k, v in temporal_dict.items()} if temporal_dict else None
        )
        
        logger.info(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def _create_sequences(self, df: pd.DataFrame, news: List[Dict], 
                         sequence_length: int) -> Tuple[List, List, List, List]:
        """Create sequences from DataFrame and news data."""
        sequences = []
        targets = []
        text_embeddings = []
        temporal_info = []
        
        # Get feature columns (exclude target and metadata)
        feature_cols = [col for col in df.columns if col not in ['target', 'Symbol', 'future_return']]
        
        # Create news embedding lookup by date
        news_lookup = {}
        for news_item in news:
            date = news_item['date'].date() if hasattr(news_item['date'], 'date') else news_item['date']
            if date not in news_lookup:
                news_lookup[date] = []
            news_lookup[date].append(news_item)
        
        for i in range(sequence_length, len(df)):
            # Numerical sequence
            seq = df[feature_cols].iloc[i-sequence_length:i].values
            target = df['target'].iloc[i]
            
            if np.isnan(target) or np.any(np.isnan(seq)):
                continue
            
            # Text embeddings sequence
            text_seq = []
            temporal_seq = {'days_ago': [], 'event_categories': []}
            
            for j in range(i-sequence_length, i):
                date = df.index[j].date()
                if date in news_lookup:
                    # Use first news item for the day (could be improved)
                    news_item = news_lookup[date][0]
                    text_seq.append(news_item.get('finbert_embedding', np.zeros(768)))
                    
                    # Temporal information
                    days_ago = (df.index[i].date() - date).days
                    temporal_seq['days_ago'].append(days_ago)
                    temporal_seq['event_categories'].append(0)  # Simplified
                else:
                    # No news for this day
                    text_seq.append(np.zeros(768))
                    temporal_seq['days_ago'].append(0)
                    temporal_seq['event_categories'].append(0)
            
            sequences.append(seq)
            targets.append(target)
            text_embeddings.append(np.array(text_seq))
            temporal_info.append({
                'days_ago': np.array(temporal_seq['days_ago']),
                'event_categories': np.array(temporal_seq['event_categories'])
            })
        
        return sequences, targets, text_embeddings, temporal_info
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            text_embeddings = batch['text_embeddings'].to(self.device)
            numerical_features = batch['numerical_features'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            temporal_info = None
            if 'temporal_info' in batch:
                temporal_info = {
                    key: value.to(self.device) for key, value in batch['temporal_info'].items()
                }
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(text_embeddings, numerical_features, temporal_info)
                    loss = self.criterion(outputs['logits'], targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(text_embeddings, numerical_features, temporal_info)
                loss = self.criterion(outputs['logits'], targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                # Optimizer step
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs['logits'], 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                text_embeddings = batch['text_embeddings'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                temporal_info = None
                if 'temporal_info' in batch:
                    temporal_info = {
                        key: value.to(self.device) for key, value in batch['temporal_info'].items()
                    }
                
                # Forward pass
                outputs = self.model(text_embeddings, numerical_features, temporal_info)
                loss = self.criterion(outputs['logits'], targets)
                
                total_loss += loss.item()
                
                # Collect predictions
                _, predicted = torch.max(outputs['logits'], 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }
        
        return avg_loss, accuracy, metrics
    
    def train(self, train_dataset: StockDataset, val_dataset: StockDataset) -> Dict[str, Any]:
        """Main training loop."""
        logger.info("Starting training...")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.config['training']['shuffle'],
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        best_val_accuracy = 0.0
        
        for epoch in range(self.epochs):
            # Training
            train_loss, train_accuracy = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_accuracy, val_metrics = self.validate(val_loader)
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['learning_rates'].append(self.scheduler.get_lr())
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} - "
                f"LR: {self.scheduler.get_lr():.6f}"
            )
            
            # Track best validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
            
            # Early stopping
            if self.early_stopping(val_accuracy, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        final_results = {
            'best_val_accuracy': best_val_accuracy,
            'final_val_metrics': val_metrics,
            'training_history': self.training_history,
            'total_epochs': epoch + 1
        }
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f}")
        
        return final_results
    
    def save_model(self, filepath: str, metadata: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config,
            'metadata': metadata or {}
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        
        logger.info(f"Model loaded from {filepath}")
        
        return checkpoint.get('metadata', {})


class OptunaOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.optimization_config = config['optimization']
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        
        # Study settings
        self.n_trials = self.optimization_config['n_trials']
        self.timeout = self.optimization_config['timeout']
        self.study_name = self.optimization_config['study_name']
        
    def objective(self, trial: optuna.Trial, train_dataset: StockDataset, 
                 val_dataset: StockDataset) -> float:
        """Objective function for Optuna optimization."""
        
        # Sample hyperparameters
        search_spaces = self.optimization_config['search_spaces']
        
        learning_rate = trial.suggest_float('learning_rate', *search_spaces['learning_rate'], log=True)
        batch_size = trial.suggest_categorical('batch_size', 
                                              list(range(search_spaces['batch_size'][0], 
                                                        search_spaces['batch_size'][1] + 1, 8)))
        hidden_dim = trial.suggest_categorical('hidden_dim',
                                              list(range(search_spaces['hidden_dim'][0],
                                                        search_spaces['hidden_dim'][1] + 1, 32)))
        num_layers = trial.suggest_int('num_layers', *search_spaces['num_layers'])
        dropout = trial.suggest_float('dropout', *search_spaces['dropout'])
        attention_heads = trial.suggest_categorical('attention_heads', [4, 8, 12, 16])
        sequence_length = trial.suggest_categorical('sequence_length',
                                                   list(range(search_spaces['sequence_length'][0],
                                                            search_spaces['sequence_length'][1] + 1, 10)))
        
        # Update config with sampled parameters
        trial_config = self.config.copy()
        trial_config['training']['learning_rate'] = learning_rate
        trial_config['training']['batch_size'] = batch_size
        trial_config['model']['hidden_dim'] = hidden_dim
        trial_config['model']['num_layers'] = num_layers
        trial_config['model']['dropout'] = dropout
        trial_config['model']['attention']['num_heads'] = attention_heads
        trial_config['model']['sequence_length'] = sequence_length
        
        try:
            # Create model with trial parameters
            from ..models.hybrid_model import HybridStockPredictor
            model = HybridStockPredictor(trial_config)
            
            # Create trainer
            trainer = StockPredictorTrainer(model, trial_config)
            
            # Reduce epochs for optimization
            trainer.epochs = min(20, trainer.epochs)  # Limit epochs for faster optimization
            
            # Train model
            results = trainer.train(train_dataset, val_dataset)
            
            # Return best validation accuracy
            return results['best_val_accuracy']
            
        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return 0.0  # Return worst possible score
    
    def optimize(self, train_dataset: StockDataset, val_dataset: StockDataset) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        logger.info("Starting hyperparameter optimization with Optuna...")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            study_name=self.study_name
        )
        
        # Run optimization
        study.optimize(
            lambda trial: self.objective(trial, train_dataset, val_dataset),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Get results
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Optimization completed. Best accuracy: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Save results
        results_path = os.path.join(self.config['reporting']['output_path'], 'optimization')
        os.makedirs(results_path, exist_ok=True)
        
        results = {
            'best_params': best_params,
            'best_value': best_value,
            'study': study,
            'trials_df': study.trials_dataframe()
        }
        
        # Save results
        with open(os.path.join(results_path, 'optimization_results.json'), 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_value': best_value,
                'n_trials': len(study.trials)
            }, f, indent=2)
        
        return results


def main():
    """Test the training functionality."""
    import yaml
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data for testing
    n_samples = 1000
    seq_len = 60
    text_dim = 768
    num_dim = 50
    n_classes = 3
    
    text_data = np.random.randn(n_samples, seq_len, text_dim)
    num_data = np.random.randn(n_samples, seq_len, num_dim)
    targets = np.random.randint(0, n_classes, n_samples)
    
    # Create datasets
    train_dataset = StockDataset(text_data[:800], num_data[:800], targets[:800])
    val_dataset = StockDataset(text_data[800:], num_data[800:], targets[800:])
    
    # Create model
    from ..models.hybrid_model import HybridStockPredictor
    model = HybridStockPredictor(config)
    
    # Create trainer
    trainer = StockPredictorTrainer(model, config)
    trainer.epochs = 2  # Reduce for testing
    
    # Test training
    results = trainer.train(train_dataset, val_dataset)
    print(f"Training completed. Best validation accuracy: {results['best_val_accuracy']:.4f}")


if __name__ == "__main__":
    main()