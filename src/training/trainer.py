"""
Training and optimization module with AdamW, scheduling, and Optuna hyperparameter optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F # Added for Focal Loss
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, roc_auc_score # Added roc_auc
from torch.utils.data import DataLoader, TensorDataset
import warnings
import sys # Added for path manipulation in main
import traceback # Added for debug logging

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# --- NEW: Focal Loss Implementation (from diagram) ---
class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss for binary classification.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (B, 1) - raw logits
        # targets: (B, 1) - 0.0 or 1.0
        
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # p_t
        
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        F_loss = alpha_t * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
# --- END NEW ---

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
        # --- DEBUG: Flag to indicate if best weights were restored ---
        self.restored_weights = False 
        # -----------------------------------------------------------
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            score: Current validation score (higher is better)
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        stop_training = False
        # --- DEBUG: Check if score is NaN ---
        if score is None or not np.isfinite(score):
            logger.warning("EarlyStopping received non-finite score. Treating as no improvement.")
            score = -float('inf') # Treat NaN/Inf as worst possible score
        # -------------------------------------

        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                # --- DEBUG: Log weight saving ---
                logger.debug(f"EarlyStopping: Saving initial best weights (score: {score:.4f}).")
                # -------------------------------
                self.best_weights = {k: v.cpu() for k, v in model.state_dict().items()} # Store on CPU
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.debug(f"EarlyStopping counter: {self.counter}/{self.patience} (Score {score:.4f} did not improve over {self.best_score:.4f} by {self.min_delta})")
            if self.counter >= self.patience:
                logger.info(f"Early stopping patience ({self.patience}) reached.")
                if self.restore_best_weights and self.best_weights is not None:
                    # --- DEBUG: Log weight restoration ---
                    logger.info("Restoring best model weights from early stopping.")
                    try:
                         # Ensure weights are moved to the correct device
                         model.load_state_dict({k: v.to(model.parameters().__next__().device) for k, v in self.best_weights.items()})
                         self.restored_weights = True
                    except Exception as e:
                         logger.error(f"Failed to restore best weights: {e}")
                    # ------------------------------------
                stop_training = True
        else:
            logger.debug(f"EarlyStopping: Validation score improved to {score:.4f} from {self.best_score:.4f}.")
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                 # --- DEBUG: Log weight saving ---
                logger.debug(f"EarlyStopping: Saving new best weights (score: {score:.4f}).")
                # -------------------------------
                self.best_weights = {k: v.cpu() for k, v in model.state_dict().items()} # Store on CPU
        
        return stop_training


class LearningRateScheduler:
    """Custom learning rate scheduler with warmup and cosine annealing."""
    
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, 
                 max_steps: int, base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = max(0, warmup_steps) # Ensure non-negative
        self.max_steps = max(1, max_steps) # Ensure at least 1 step
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        
        if self.warmup_steps > 0 and self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        # --- Check if annealing phase is valid ---
        elif self.max_steps > self.warmup_steps: 
            # Cosine annealing
            # Ensure denominator is positive
            denominator = max(1, self.max_steps - self.warmup_steps) 
            progress = (self.current_step - self.warmup_steps) / denominator
            progress = min(progress, 1.0) # Cap progress at 1
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        # ------------------------------------------
        else:
            # No warmup or annealing needed/possible (e.g., max_steps <= warmup_steps)
            lr = self.base_lr

        # Ensure learning rate doesn't go below min_lr (except during potential initial phase)
        lr = max(lr, self.min_lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        # --- Handle case where optimizer has no param_groups ---
        if not self.optimizer.param_groups:
             return 0.0 # Or raise an error
        # ----------------------------------------------------
        return self.optimizer.param_groups[0]['lr']


class StockDataset:
    """Dataset class for stock prediction data."""
    
    def __init__(self, text_embeddings: np.ndarray, numerical_features: np.ndarray,
                 targets: np.ndarray, num_classes: int, # --- MODIFIED: Added num_classes ---
                 temporal_info: Optional[Dict[str, np.ndarray]] = None):
        
        # --- DEBUG: Check input shapes and types ---
        logger.debug(f"StockDataset init: text_embeddings shape={text_embeddings.shape}, dtype={text_embeddings.dtype}")
        logger.debug(f"StockDataset init: numerical_features shape={numerical_features.shape}, dtype={numerical_features.dtype}")
        logger.debug(f"StockDataset init: targets shape={targets.shape}, dtype={targets.dtype}")
        if temporal_info:
            for k, v in temporal_info.items():
                logger.debug(f"StockDataset init: temporal_info['{k}'] shape={v.shape}, dtype={v.dtype}")
        # -------------------------------------------

        # --- Convert to Tensors with explicit types ---
        self.text_embeddings = torch.as_tensor(text_embeddings, dtype=torch.float32)
        self.numerical_features = torch.as_tensor(numerical_features, dtype=torch.float32)
        
        # --- MODIFIED: Handle target type based on num_classes ---
        if num_classes == 1:
            # Binary classification (BCE/Focal Loss) expects Float targets, shape (B, 1)
            self.targets = torch.as_tensor(targets, dtype=torch.float32).unsqueeze(-1)
            logger.debug(f"StockDataset: Targets set to FLOAT, shape={self.targets.shape}")
        else:
            # Multiclass (CrossEntropy) expects Long targets, shape (B)
            self.targets = torch.as_tensor(targets, dtype=torch.long) 
            logger.debug(f"StockDataset: Targets set to LONG, shape={self.targets.shape}")
        # --- END MODIFICATION ---
        
        # Handle data types correctly for temporal info
        self.temporal_info = None
        if temporal_info:
            self.temporal_info = {}
            for key, value in temporal_info.items():
                if key == 'event_categories':
                    # Ensure categories are LongTensor for indexing
                    self.temporal_info[key] = torch.as_tensor(value, dtype=torch.long)
                else:
                    # Other temporal info (like days_ago) can be FloatTensor
                    self.temporal_info[key] = torch.as_tensor(value, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        item = {
            'text_embeddings': self.text_embeddings[idx],
            'numerical_features': self.numerical_features[idx],
            'targets': self.targets[idx]
        }
        
        if self.temporal_info:
            # Make sure to handle potential missing keys gracefully if needed
            item['temporal_info'] = {}
            for key, value_tensor in self.temporal_info.items():
                 # --- DEBUG: Add check for index out of bounds ---
                 if idx < len(value_tensor):
                     item['temporal_info'][key] = value_tensor[idx]
                 else:
                     logger.error(f"Index {idx} out of bounds for temporal key '{key}' with length {len(value_tensor)}")
                     # Handle error: maybe return default value or raise error
                     # For now, let it potentially raise IndexError later or handle in collate_fn
                 # ---------------------------------------------
        return item


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    # --- DEBUG: Check if batch is empty ---
    if not batch:
        logger.warning("collate_fn received an empty batch.")
        # Return empty tensors or dictionary of empty tensors
        # Adjust shapes based on your model's expected input
        return { 
            'text_embeddings': torch.empty((0, 0, 0), dtype=torch.float), 
            'numerical_features': torch.empty((0, 0, 0), dtype=torch.float),
            'targets': torch.empty((0,), dtype=torch.long)
        }
    # -------------------------------------

    try:
        text_embeddings = torch.stack([item['text_embeddings'] for item in batch])
        numerical_features = torch.stack([item['numerical_features'] for item in batch])
        targets = torch.stack([item['targets'] for item in batch])
        
        result = {
            'text_embeddings': text_embeddings,
            'numerical_features': numerical_features,
            'targets': targets
        }
        
        # Handle temporal info if present and non-empty
        if batch and 'temporal_info' in batch[0] and batch[0]['temporal_info']:
            temporal_info = {}
            # Iterate through keys present in the first item's temporal_info
            for key in batch[0]['temporal_info'].keys():
                # Ensure all items in the batch have this key before stacking
                if all(key in item.get('temporal_info', {}) for item in batch):
                    try:
                        temporal_info[key] = torch.stack([item['temporal_info'][key] for item in batch])
                    except Exception as e:
                         logger.error(f"Error stacking temporal key '{key}': {e}. Skipping this key for the batch.")
                         # --- DEBUG: Log shapes on error ---
                         shapes = [item['temporal_info'][key].shape for item in batch if key in item.get('temporal_info', {})]
                         logger.debug(f"Shapes for key '{key}': {shapes}")
                         # ---------------------------------
                else:
                     logger.warning(f"Temporal key '{key}' missing in some items of the batch. Skipping.")
            if temporal_info: # Only add if we successfully stacked something
                 result['temporal_info'] = temporal_info
        
        return result

    except Exception as collate_err:
        logger.error(f"Error during collate_fn: {collate_err}")
        # --- DEBUG: Log item details on error ---
        for i, item in enumerate(batch):
            logger.debug(f"Batch item {i}:")
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    logger.debug(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                elif isinstance(v, dict):
                     logger.debug(f"  {k}:")
                     for sub_k, sub_v in v.items():
                          if isinstance(sub_v, torch.Tensor):
                               logger.debug(f"    {sub_k}: shape={sub_v.shape}, dtype={sub_v.dtype}")
                          else:
                               logger.debug(f"    {sub_k}: {sub_v}")
                else:
                    logger.debug(f"  {k}: {v}")
        # --------------------------------------
        # Propagate error or return dummy batch
        raise collate_err # Re-raise error to stop training


class ModelTrainer:
    """Individual model trainer for consistent training across different model types."""
    
    def __init__(self, model: nn.Module, model_name: str, config: Dict, device: torch.device):
        self.model = model
        self.model_name = model_name
        self.config = config
        self.device = device
        self.model.to(self.device)
        
        # Training parameters
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']
        self.gradient_clip_norm = config['training']['gradient_clip_norm']
        
        # --- MODIFIED: Added num_classes ---
        self.num_classes = config['model']['output']['num_classes']
        # --- END MODIFICATION ---
        
        # Initialize optimizer
        try:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        except ValueError as e:
            logger.error(f"Error initializing optimizer for {model_name}: {e}")
            raise
        
        # Calculate scheduler parameters
        estimated_train_size = self.config.get('training', {}).get('estimated_train_samples', 8000)
        if self.batch_size > 0:
            estimated_steps_per_epoch = max(1, estimated_train_size // self.batch_size)
        else:
            estimated_steps_per_epoch = 100
        calculated_max_steps = self.epochs * estimated_steps_per_epoch
        
        config_max_steps = config['training'].get('max_steps', 0)
        if config_max_steps > 0 and config_max_steps < calculated_max_steps:
            final_max_steps_to_use = config_max_steps
        else:
            final_max_steps_to_use = calculated_max_steps
        
        # Initialize scheduler
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            warmup_steps=config['training'].get('warmup_steps', 0),
            max_steps=final_max_steps_to_use,
            base_lr=self.learning_rate
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping_patience']
        )
        
        # --- MODIFIED: Loss function based on config ---
        loss_type = config['training'].get('loss_function', 'cross_entropy')
        if self.num_classes == 1:
            if loss_type == 'focal_loss':
                logger.info(f"{model_name}: Using FocalLoss for binary classification.")
                self.criterion = FocalLoss()
            else: # Default to BCE for binary
                logger.info(f"{model_name}: Using BCEWithLogitsLoss for binary classification.")
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            logger.info(f"{model_name}: Using CrossEntropyLoss for {self.num_classes}-class classification.")
            self.criterion = nn.CrossEntropyLoss()
        # --- END MODIFICATION ---
        
        # Mixed precision training
        self.use_mixed_precision = config['training'].get('mixed_precision', False)
        self.scaler = None
        if self.use_mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        elif self.use_mixed_precision and self.device.type != 'cuda':
            logger.warning(f"Mixed precision requested for {model_name} but CUDA not available. Disabling mixed precision.")
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        nan_loss_batches = 0
        nan_logit_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move batch to device
                text_embeddings = batch['text_embeddings'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Handle temporal info if present
                temporal_info = None
                if 'temporal_info' in batch and batch['temporal_info']:
                    temporal_info = {
                        'days_ago': batch['temporal_info']['days_ago'].to(self.device),
                        'event_categories': batch['temporal_info']['event_categories'].to(self.device)
                    }
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision if enabled
                if self.use_mixed_precision and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = self.model(text_embeddings, numerical_features, temporal_info)
                        logits = output['logits']
                        loss = self.criterion(logits, targets)
                else:
                    output = self.model(text_embeddings, numerical_features, temporal_info)
                    logits = output['logits']
                    loss = self.criterion(logits, targets)
                
                # Check for NaN/Inf in logits and loss
                if not torch.isfinite(logits).all():
                    nan_logit_batches += 1
                    logger.warning(f"{self.model_name}: NaN/Inf detected in logits at batch {batch_idx}")
                    continue
                
                if not torch.isfinite(loss):
                    nan_loss_batches += 1
                    logger.warning(f"{self.model_name}: NaN/Inf detected in loss at batch {batch_idx}")
                    continue
                
                # Backward pass
                if self.use_mixed_precision and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    if self.gradient_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.optimizer.step()
                
                # Step scheduler
                self.scheduler.step()
                
                # Update statistics
                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # --- MODIFIED: Calculate accuracy (binary vs multiclass) ---
                if self.num_classes == 1:
                    predicted = (torch.sigmoid(logits) > 0.5).int()
                    total_correct += (predicted == targets.int()).sum().item()
                else:
                    _, predicted = torch.max(logits, 1)
                    total_correct += (predicted == targets).sum().item()
                # --- END MODIFICATION ---
                
            except Exception as batch_err:
                logger.error(f"{self.model_name}: Error in training batch {batch_idx}: {batch_err}")
                continue
        
        if nan_logit_batches > 0:
            logger.warning(f"{self.model_name}: Had {nan_logit_batches} batches with NaN/Inf logits")
        if nan_loss_batches > 0:
            logger.warning(f"{self.model_name}: Had {nan_loss_batches} batches with NaN/Inf loss")
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        if not np.isfinite(avg_loss):
            logger.warning(f"{self.model_name}: Average training loss is not finite: {avg_loss}")
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = [] # For ROC AUC
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    # Move batch to device
                    text_embeddings = batch['text_embeddings'].to(self.device)
                    numerical_features = batch['numerical_features'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    
                    # Handle temporal info if present
                    temporal_info = None
                    if 'temporal_info' in batch and batch['temporal_info']:
                        temporal_info = {
                            'days_ago': batch['temporal_info']['days_ago'].to(self.device),
                            'event_categories': batch['temporal_info']['event_categories'].to(self.device)
                        }
                    
                    # Forward pass
                    output = self.model(text_embeddings, numerical_features, temporal_info)
                    logits = output['logits']
                    loss = self.criterion(logits, targets)
                    
                    # Skip if NaN/Inf
                    if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                        logger.warning(f"{self.model_name}: Skipping validation batch {batch_idx} due to NaN/Inf")
                        continue
                    
                    # Update statistics
                    batch_size = targets.size(0)
                    total_loss += loss.item() * batch_size
                    
                    # --- MODIFIED: Store predictions (binary vs multiclass) ---
                    if self.num_classes == 1:
                        probs = torch.sigmoid(logits)
                        predicted = (probs > 0.5).int()
                        all_probabilities.extend(probs.cpu().numpy())
                        all_predictions.extend(predicted.cpu().numpy())
                        all_targets.extend(targets.int().cpu().numpy())
                    else:
                        probs = torch.softmax(logits, dim=1)
                        _, predicted = torch.max(logits, 1)
                        all_probabilities.extend(probs.cpu().numpy())
                        all_predictions.extend(predicted.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    # --- END MODIFICATION ---
                    
                except Exception as batch_err:
                    logger.error(f"{self.model_name}: Error in validation batch {batch_idx}: {batch_err}")
                    continue
        
        total_samples = len(all_targets)
        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        
        if not all_targets or not all_predictions or total_samples == 0:
            logger.error(f"{self.model_name}: No valid validation samples")
            return float('nan'), 0.0, {}
        
        if not np.isfinite(avg_loss):
            logger.warning(f"{self.model_name}: Average validation loss is not finite: {avg_loss}")
        
        # --- MODIFIED: Metrics calculation (binary vs multiclass) ---
        accuracy = accuracy_score(all_targets, all_predictions)
        
        if self.num_classes == 1:
            avg_mode = 'binary'
            roc_auc_avg = 'roc_auc' # Use default for binary
        else:
            avg_mode = 'weighted'
            roc_auc_avg = 'weighted'
            
        try:
            f1 = f1_score(all_targets, all_predictions, average=avg_mode, zero_division=0)
            precision = precision_score(all_targets, all_predictions, average=avg_mode, zero_division=0)
            recall = recall_score(all_targets, all_predictions, average=avg_mode, zero_division=0)
            
            # ROC AUC
            if self.num_classes == 1:
                roc_auc = roc_auc_score(all_targets, all_probabilities)
            else:
                # Ensure all_probabilities is correct shape for multiclass
                roc_auc = roc_auc_score(all_targets, all_probabilities, multi_class='ovr', average=roc_auc_avg)

        except ValueError as metric_err:
            logger.warning(f"{self.model_name}: Error calculating metrics: {metric_err}")
            f1 = precision = recall = roc_auc = 0.0
        # --- END MODIFICATION ---

        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc # Added ROC AUC
        }
        
        return avg_loss, accuracy, metrics
    
    def train(self, train_dataset: StockDataset, val_dataset: StockDataset) -> Dict[str, Any]:
        """Main training loop for individual model."""
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            logger.error(f"{self.model_name}: Empty dataset provided")
            return {'error': 'Empty dataset'}
        
        logger.info(f"Starting training for {self.model_name}...")
        
        # Create data loaders
        try:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0
            )
        except Exception as loader_err:
            logger.error(f"{self.model_name}: Error creating data loaders: {loader_err}")
            return {'error': f'Data loader creation failed: {loader_err}'}
        
        # Handle Random Forest special case
        if hasattr(self.model, 'model_type') and self.model.model_type == 'random_forest':
            return self._train_random_forest(train_dataset, val_dataset)
        
        best_val_accuracy = -1.0
        epochs_run = 0
        final_val_metrics = {}
        
        try:
            for epoch in range(self.epochs):
                epochs_run = epoch + 1
                
                # Training
                train_loss, train_acc = self.train_epoch(train_loader)
                
                # Validation
                val_loss, val_acc, val_metrics = self.validate(val_loader)
                
                # Update history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['train_accuracy'].append(train_acc)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_acc)
                self.training_history['learning_rates'].append(self.scheduler.get_lr())
                
                # Track best accuracy
                if np.isfinite(val_acc) and val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    final_val_metrics = val_metrics.copy()
                
                # Log progress
                if epoch % 10 == 0 or epoch == self.epochs - 1:
                    logger.info(f"{self.model_name} Epoch {epoch+1}/{self.epochs}: "
                              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Early stopping check
                if self.early_stopping(val_acc, self.model):
                    logger.info(f"{self.model_name}: Early stopping triggered at epoch {epoch+1}")
                    break
                    
        except Exception as train_err:
            logger.error(f"{self.model_name}: Training failed: {train_err}")
            return {'error': f'Training failed: {train_err}'}
        
        final_best_val_acc = self.early_stopping.best_score if self.early_stopping.restored_weights else best_val_accuracy
        
        final_results = {
            'best_val_accuracy': final_best_val_acc,
            'final_val_metrics': final_val_metrics,
            'training_history': self.training_history,
            'total_epochs': epochs_run
        }
        
        logger.info(f"{self.model_name} training completed. Best validation accuracy: {final_best_val_acc:.4f} over {epochs_run} epochs.")
        
        return final_results
    
    def _train_random_forest(self, train_dataset: StockDataset, val_dataset: StockDataset) -> Dict[str, Any]:
        """Special training procedure for Random Forest."""
        try:
            # Prepare training data
            X_train = []
            y_train = []
            
            for i in range(len(train_dataset)):
                item = train_dataset[i]
                text_emb = item['text_embeddings'].numpy()
                num_feat = item['numerical_features'].numpy()
                target = item['targets'].item()
                
                # Concatenate and flatten features
                combined = np.concatenate([text_emb, num_feat], axis=-1)
                X_train.append(combined.flatten())
                y_train.append(target)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Fit the model
            self.model.fit(X_train, y_train)
            
            # Validate
            X_val = []
            y_val = []
            
            for i in range(len(val_dataset)):
                item = val_dataset[i]
                text_emb = item['text_embeddings'].numpy()
                num_feat = item['numerical_features'].numpy()
                target = item['targets'].item()
                
                combined = np.concatenate([text_emb, num_feat], axis=-1)
                X_val.append(combined.flatten())
                y_val.append(target)
            
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            
            # Get predictions using the model's forward method
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
            all_predictions = []
            all_targets = []
            
            self.model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    text_embeddings = batch['text_embeddings']
                    numerical_features = batch['numerical_features']
                    targets = batch['targets']
                    
                    output = self.model(text_embeddings, numerical_features)
                    logits = output['logits']
                    
                    _, predicted = torch.max(logits, 1)
                    all_predictions.extend(predicted.numpy())
                    all_targets.extend(targets.numpy())
            
            # Calculate metrics
            accuracy = accuracy_score(all_targets, all_predictions)
            f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
            precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall
            }
            
            return {
                'best_val_accuracy': accuracy,
                'final_val_metrics': metrics,
                'training_history': {'val_accuracy': [accuracy]},
                'total_epochs': 1
            }
            
        except Exception as e:
            logger.error(f"{self.model_name}: Random Forest training failed: {e}")
            return {'error': f'Random Forest training failed: {e}'}
    
    def save_model(self, filepath: str, metadata: Optional[Dict] = None):
        """Save model checkpoint."""
        # Ensure directory exists just before saving
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        except OSError as e:
            logger.error(f"{self.model_name}: Error creating directory for {filepath}: {e}")
            raise

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': {'max_steps': self.scheduler.max_steps, 'current_step': self.scheduler.current_step},
            'training_history': self.training_history,
            'best_score': self.early_stopping.best_score,
            'config': { 
                 'model': self.config.get('model'), 
                 'features': self.config.get('features'),
                 'training': self.config.get('training')
                 },
            'metadata': metadata or {}
        }
        
        try:
            torch.save(checkpoint, filepath)
            logger.info(f"{self.model_name}: Model saved to {filepath}")
        except Exception as e:
            logger.error(f"{self.model_name}: Error saving model to {filepath}: {e}")
            raise


class StockPredictorTrainer:
    """Main training class for stock prediction with multiple models."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Model configurations
        self.model_configs = {
            'hybrid': {'enabled': True, 'priority': 1},
            'logistic_regression': {'enabled': True, 'priority': 2},
            'random_forest': {'enabled': True, 'priority': 3},
            'basic_lstm': {'enabled': True, 'priority': 4},
            'bilstm': {'enabled': True, 'priority': 5},
            'bilstm_attention': {'enabled': True, 'priority': 6},
            'transformer': {'enabled': True, 'priority': 7}
        }
        
        # Override with config if available
        if 'models' in config:
            for model_name, model_config in config['models'].items():
                if model_name in self.model_configs:
                    self.model_configs[model_name].update(model_config)
        
        self.results = {}
        self.models = {}
    
    def create_models(self) -> Dict[str, nn.Module]:
        """Create all models based on configuration."""
        models = {}
        
        # Import model classes
        try:
            from ..models.hybrid_model import HybridStockPredictor
            from ..models.baseline_models import BaselineModelFactory
        except ImportError as e:
            logger.error(f"Failed to import model classes: {e}")
            raise
        
        # Create hybrid model first
        if self.model_configs['hybrid']['enabled']:
            try:
                hybrid_model = HybridStockPredictor(self.config)
                models['hybrid'] = hybrid_model
                logger.info("Created hybrid model")
            except Exception as e:
                logger.error(f"Failed to create hybrid model: {e}")
        
        # Create baseline models
        baseline_types = ['logistic_regression', 'random_forest', 'basic_lstm', 
                         'bilstm', 'bilstm_attention', 'transformer']
        
        for model_type in baseline_types:
            if self.model_configs[model_type]['enabled']:
                try:
                    baseline_model = BaselineModelFactory.create_model(model_type, self.config)
                    models[model_type] = baseline_model
                    logger.info(f"Created {model_type} model")
                except Exception as e:
                    logger.error(f"Failed to create {model_type} model: {e}")
        
        return models
    
    def train_all_models(self, train_dataset: StockDataset, val_dataset: StockDataset) -> Dict[str, Any]:
        """Train all models and return results."""
        logger.info("Starting training for all models...")
        
        # Create all models
        self.models = self.create_models()
        
        if not self.models:
            logger.error("No models were successfully created")
            return {'error': 'No models created'}
        
        # Sort models by priority
        sorted_models = sorted(
            self.models.items(),
            key=lambda x: self.model_configs[x[0]]['priority']
        )
        
        # Train each model
        for model_name, model in sorted_models:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name.upper()} model")
            logger.info(f"{'='*50}")
            
            try:
                # Create individual trainer
                trainer = ModelTrainer(model, model_name, self.config, self.device)
                
                # Train the model
                results = trainer.train(train_dataset, val_dataset)
                
                # Store results
                self.results[model_name] = {
                    'results': results,
                    'model': model,
                    'trainer': trainer
                }
                
                # Log results
                if 'error' not in results:
                    best_acc = results.get('best_val_accuracy', 0)
                    logger.info(f"{model_name} completed - Best Val Accuracy: {best_acc:.4f}")
                else:
                    logger.error(f"{model_name} failed: {results['error']}")
                    
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                self.results[model_name] = {'error': str(e)}
        
        # Generate summary
        summary = self.generate_summary()
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(summary)
        
        return {
            'individual_results': self.results,
            'summary': summary,
            'models': self.models
        }
    
    def generate_summary(self) -> str:
        """Generate a summary of all model results."""
        summary_lines = []
        
        # Collect results
        model_results = []
        for model_name, result_data in self.results.items():
            if 'error' in result_data:
                model_results.append({
                    'name': model_name,
                    'accuracy': 0.0,
                    'status': 'FAILED',
                    'error': result_data['error']
                })
            else:
                results = result_data['results']
                model_results.append({
                    'name': model_name,
                    'accuracy': results.get('best_val_accuracy', 0.0),
                    'status': 'SUCCESS',
                    'epochs': results.get('total_epochs', 0)
                })
        
        # Sort by accuracy (descending)
        model_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Generate summary
        summary_lines.append(f"{'Model':<20} {'Accuracy':<10} {'Status':<10} {'Info'}")
        summary_lines.append("-" * 60)
        
        for result in model_results:
            accuracy_str = f"{result['accuracy']:.4f}" if result['accuracy'] > 0 else "N/A"
            info = f"Epochs: {result.get('epochs', 'N/A')}" if result['status'] == 'SUCCESS' else result.get('error', '')[:30]
            summary_lines.append(f"{result['name']:<20} {accuracy_str:<10} {result['status']:<10} {info}")
        
        return "\n".join(summary_lines)
    
    def save_all_models(self, output_dir: str):
        """Save all trained models."""
        import os
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, result_data in self.results.items():
            if 'error' not in result_data and 'trainer' in result_data:
                try:
                    model_path = os.path.join(output_dir, f"{model_name}_{timestamp}.pth")
                    trainer = result_data['trainer']
                    trainer.save_model(model_path, {'model_type': model_name})
                    logger.info(f"Saved {model_name} model to {model_path}")
                except Exception as e:
                    logger.error(f"Failed to save {model_name}: {e}")
    
    def get_best_model(self) -> Tuple[str, nn.Module, Dict]:
        """Get the best performing model."""
        best_name = None
        best_model = None
        best_results = None
        best_accuracy = -1.0
        
        for model_name, result_data in self.results.items():
            if 'error' not in result_data:
                results = result_data['results']
                accuracy = results.get('best_val_accuracy', 0.0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_name = model_name
                    best_model = result_data['model']
                    best_results = results
        
        return best_name, best_model, best_results
        
        # Training parameters
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']
        self.gradient_clip_norm = config['training']['gradient_clip_norm']
        
        # Initialize optimizer
        try:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        except ValueError as e:
             logger.error(f"Error initializing optimizer: {e}. Check model parameters.")
             raise
        
        # --- START REPLACEMENT ---
        # Calculate max_steps based on estimated total steps FIRST
        # Use a placeholder estimate, this will be updated later if possible
        estimated_train_size = self.config.get('training', {}).get('estimated_train_samples', 8000) # Default estimate
        if self.batch_size > 0:
            # Calculate steps per epoch based on estimate
            # Note: This is just for initial setup, it gets refined in the train() method
            estimated_steps_per_epoch = (estimated_train_size + self.batch_size - 1) // self.batch_size # Use ceiling division
        else:
            estimated_steps_per_epoch = 1 # Avoid division by zero
        calculated_max_steps = self.epochs * estimated_steps_per_epoch
        
        # Get max_steps from config, default to 0 if not present
        config_max_steps = config['training'].get('max_steps', 0)
        
        # Determine the final max_steps to use for scheduler initialization
        # Prefer config value ONLY if it's positive AND smaller than the calculation based on epochs/estimate.
        # Otherwise, default to the calculated value.
        if config_max_steps > 0 and config_max_steps < calculated_max_steps:
             final_max_steps_to_use = config_max_steps
             logger.info(f"Using max_steps from config for initial scheduler setup: {final_max_steps_to_use}")
        else:
             final_max_steps_to_use = calculated_max_steps
             # Make sure it's at least 1
             final_max_steps_to_use = max(1, final_max_steps_to_use)
             logger.info(f"Using calculated max_steps for initial scheduler setup: {final_max_steps_to_use} (Config value was {config_max_steps})")
        # --- END REPLACEMENT ---

        # Initialize scheduler using the determined value
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            warmup_steps=config['training'].get('warmup_steps', 0), 
            max_steps=final_max_steps_to_use, # Use the determined value
            base_lr=self.learning_rate
        )
        
        # Store the final max_steps intended (will be updated again in train method)
        self.final_max_steps = self.scheduler.max_steps 
        
        logger.info(f"Scheduler initially initialized with warmup={config['training'].get('warmup_steps', 0)}, max_steps={self.final_max_steps}")
        logger.info(f"Scheduler initialized with warmup={config['training'].get('warmup_steps', 0)}, max_steps={self.final_max_steps}")
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping_patience']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision training
        self.use_mixed_precision = config['training'].get('mixed_precision', False) # Use .get
        self.scaler = None # Initialize scaler to None
        if self.use_mixed_precision and self.device.type == 'cuda': # Check for CUDA
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled.")
        elif self.use_mixed_precision and self.device.type != 'cuda':
             logger.warning("Mixed precision configured but not enabled (requires CUDA device).")
             self.use_mixed_precision = False # Ensure it's disabled
        
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
        """
        all_sequences = []
        all_targets = []
        all_text_embeddings = []
        all_temporal_info = []
        
        processed_count = 0
        skipped_count = 0
        
        for symbol in features_data.keys():
            if symbol not in news_data or not isinstance(features_data[symbol], pd.DataFrame) or features_data[symbol].empty:
                logger.warning(f"Skipping {symbol}: Missing/empty features or news data.")
                skipped_count +=1
                continue
            
            df = features_data[symbol]
            news = news_data[symbol]
            
            if df.empty:
                logger.warning(f"Skipping {symbol}: Feature DataFrame is empty.")
                skipped_count += 1
                continue
                
            try:
                sequences, targets, text_emb, temporal = self._create_sequences(
                    df, news, sequence_length
                )
                
                if len(sequences) > 0:
                    all_sequences.extend(sequences)
                    all_targets.extend(targets)
                    all_text_embeddings.extend(text_emb)
                    all_temporal_info.extend(temporal)
                    processed_count += 1
                else:
                    logger.warning(f"No valid sequences generated for {symbol}.")
                    skipped_count += 1
                    
            except Exception as e:
                logger.error(f"Error creating sequences for {symbol}: {e}")
                logger.debug(traceback.format_exc()) # Log full traceback for debugging
                skipped_count += 1

        logger.info(f"Sequence creation summary: Processed symbols={processed_count}, Skipped symbols={skipped_count}")

        if not all_sequences:
            raise ValueError("No valid sequences created from any symbol. Check data and feature engineering steps.")
        
        # --- MODIFIED: Get num_classes from config ---
        num_classes = self.config['model']['output']['num_classes']
        target_dtype = np.float32 if num_classes == 1 else np.int64
        # --- END MODIFICATION ---
        
        # Convert to arrays
        try:
            X_num = np.array(all_sequences, dtype=np.float32) 
            X_text = np.array(all_text_embeddings, dtype=np.float32) 
            y = np.array(all_targets, dtype=target_dtype) # Use determined dtype
            
            if num_classes > 1:
                logger.info(f"Target distribution (Multiclass): {np.bincount(y) / len(y)}")
            else:
                logger.info(f"Target distribution (Binary): {np.mean(y):.2f} positive class")
        except ValueError as e:
             logger.error(f"Error converting sequence lists to numpy arrays: {e}")
             # --- DEBUG: Log shapes of individual elements if conversion fails ---
             logger.debug("Shapes of first few elements:")
             for i in range(min(5, len(all_sequences))):
                 logger.debug(f"  Sequence {i}: num={np.array(all_sequences[i]).shape}, text={np.array(all_text_embeddings[i]).shape}")
             # ------------------------------------------------------------------
             raise

        # Temporal info
        temporal_dict = None
        if all_temporal_info and all_temporal_info[0] is not None: 
            try:
                 temporal_dict = {
                     'days_ago': np.array([t['days_ago'] for t in all_temporal_info], dtype=np.float32),
                     'event_categories': np.array([t['event_categories'] for t in all_temporal_info], dtype=np.int64) # Categories must be int/long
                 }
                 # --- DEBUG: Check temporal array shapes ---
                 logger.debug(f"Temporal 'days_ago' shape: {temporal_dict['days_ago'].shape}")
                 logger.debug(f"Temporal 'event_categories' shape: {temporal_dict['event_categories'].shape}")
                 # -----------------------------------------
            except Exception as e:
                logger.error(f"Error processing temporal info into arrays: {e}. Disabling temporal features.")
                logger.debug(traceback.format_exc())
                temporal_dict = None 
        else:
             logger.info("No temporal information found or generated. Temporal features will be disabled.")

        # Train/validation split
        val_split = self.config['training']['val_split']
        if not (0 < val_split < 1):
            raise ValueError(f"Invalid validation split value: {val_split}. Must be between 0 and 1.")
        
        split_idx = int(len(X_num) * (1 - val_split))
        if split_idx == 0 or split_idx == len(X_num):
             logger.warning(f"Train/validation split resulted in empty set (split_idx={split_idx}, total={len(X_num)}). Adjust val_split or data size.")
             # Handle this case, e.g., raise error or adjust split
             split_idx = max(1, min(len(X_num) - 1, split_idx)) # Ensure at least one sample in each if possible

        indices = np.arange(len(X_num)) # Use simple indices for splitting
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        logger.info(f"Splitting data: Train size={len(train_indices)}, Validation size={len(val_indices)}")

        # Normalize numerical features using StandardScaler
        from sklearn.preprocessing import StandardScaler

        # 1. Get shapes
        n_samples_train = X_num[train_indices].shape[0]
        n_samples_val = X_num[val_indices].shape[0]
        seq_len = X_num.shape[1]
        n_features = X_num.shape[2]

        # 2. Reshape data for scaling
        # Reshape from [samples, seq_len, features] to [samples * seq_len, features]
        X_num_train_2d = X_num[train_indices].reshape(-1, n_features)
        X_num_val_2d = X_num[val_indices].reshape(-1, n_features)

        # 3. Fit scaler ONLY on training data
        scaler = StandardScaler()
        scaler.fit(X_num_train_2d)

        # 4. Transform both train and validation data
        X_num_train_scaled_2d = scaler.transform(X_num_train_2d)
        X_num_val_scaled_2d = scaler.transform(X_num_val_2d)

        # 5. Reshape back to [samples, seq_len, features]
        X_num_train_scaled = X_num_train_scaled_2d.reshape(n_samples_train, seq_len, n_features)
        X_num_val_scaled = X_num_val_scaled_2d.reshape(n_samples_val, seq_len, n_features)

        logger.info("Applied StandardScaler to numerical features.")

        # Create datasets using the SCALED data
        try:
            # --- MODIFIED: Pass num_classes to StockDataset ---
            train_dataset = StockDataset(
                X_text[train_indices],
                X_num_train_scaled,  # <-- Use scaled data
                y[train_indices],
                num_classes, # Pass num_classes
                {k: v[train_indices] for k, v in temporal_dict.items()} if temporal_dict else None
            )
            
            val_dataset = StockDataset(
                X_text[val_indices],
                X_num_val_scaled,  # <-- Use scaled data
                y[val_indices],
                num_classes, # Pass num_classes
                {k: v[val_indices] for k, v in temporal_dict.items()} if temporal_dict else None
            )
            # --- END MODIFICATION ---
        except Exception as e:
             logger.error(f"Error creating StockDataset objects: {e}")
             logger.debug(traceback.format_exc())
             raise

        logger.info(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def _create_sequences(self, df: pd.DataFrame, news: List[Dict], 
                         sequence_length: int) -> Tuple[List, List, List, List]:
        """Create sequences from DataFrame and news data."""
        sequences = []
        targets = []
        text_embeddings = []
        temporal_info = []
        
        # Explicitly select only NUMERICAL columns
        exclude_cols = ['target', 'Symbol', 'future_return', 'sentiment_probs', 'date'] 
        # --- Select numerical types robustly ---
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # ---------------------------------------
        feature_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if not feature_cols:
             raise ValueError(f"No numerical feature columns found after exclusions in DataFrame for symbol {df['Symbol'].iloc[0] if 'Symbol' in df.columns else 'Unknown'}. Available columns: {df.columns.tolist()}")
        logger.debug(f"Using {len(feature_cols)} numerical features for sequences: {feature_cols[:5]}...")
        
        # Create news embedding lookup by date
        news_lookup = {}
        processed_news_count = 0
        skipped_news_count = 0
        for news_item in news:
             item_date = news_item.get('date')
             date_key = None
             try:
                 if isinstance(item_date, datetime):
                     date_key = item_date.date()
                 elif isinstance(item_date, pd.Timestamp):
                     date_key = item_date.date()
                 elif isinstance(item_date, str):
                      date_key = datetime.fromisoformat(item_date).date()
                 elif hasattr(item_date, 'date'): 
                      date_key = item_date.date()
                 else:
                      raise TypeError(f"Unsupported date type {type(item_date)}")

                 if date_key not in news_lookup:
                     news_lookup[date_key] = []
                 news_lookup[date_key].append(news_item)
                 processed_news_count += 1
             except Exception as parse_error:
                 logger.warning(f"Could not parse/use date '{item_date}' in news lookup: {parse_error}")
                 skipped_news_count += 1
        logger.debug(f"News lookup created: {len(news_lookup)} dates, {processed_news_count} items processed, {skipped_news_count} skipped.")


        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
             logger.warning("DataFrame index is not DatetimeIndex. Converting...")
             try:
                 df.index = pd.to_datetime(df.index)
             except Exception as e:
                 raise ValueError(f"Could not convert DataFrame index to DatetimeIndex: {e}")

        df_values = df[feature_cols].values.astype(np.float32) # Ensure float32
        df_targets = df['target'].values
        df_index_dates = df.index.date # Pre-calculate dates for lookup

        # Check for NaNs/Infs *only* on the selected numerical columns
        # --- More robust check for non-finite values ---
        if not np.all(np.isfinite(df_values)):
            non_finite_mask = ~np.isfinite(df_values)
            rows_with_non_finite = np.where(non_finite_mask.any(axis=1))[0]
            cols_with_non_finite = np.where(non_finite_mask.any(axis=0))[0]
            logger.warning(f"{len(rows_with_non_finite)} rows contain non-finite (NaN/Inf) values in numerical features.")
            logger.warning(f"Example indices: {rows_with_non_finite[:10]}...")
            logger.warning(f"Columns with non-finite values (indices): {cols_with_non_finite}")
            logger.warning(f"Columns names: {[feature_cols[i] for i in cols_with_non_finite]}")
            
            # Fill NaNs/Infs (replace Inf with large number before fillna)
            df_filled = df[feature_cols].replace([np.inf, -np.inf], np.nan)
            df_filled = df_filled.ffill().bfill().fillna(0)
            df_values = df_filled.values.astype(np.float32) # Recast after filling
            logger.info("Non-finite values in numerical features have been filled.")
            if not np.all(np.isfinite(df_values)): # Double check
                 raise ValueError("Non-finite values still present after attempting fill.")
        # ---------------------------------------------

        # Check for NaN/Inf in targets before loop
        non_finite_target_mask = ~np.isfinite(df_targets)
        non_finite_target_indices = np.where(non_finite_target_mask)[0]
        if len(non_finite_target_indices) > 0:
             logger.warning(f"{len(non_finite_target_indices)} non-finite (NaN/Inf) values found in target column. These sequences will be skipped.")

        num_skipped_shape_mismatch = 0
        num_skipped_nan_target = 0

        # --- Efficiently iterate using range and slicing ---
        num_samples = len(df)
        for i in range(sequence_length, num_samples):
            target_idx = i
            start_idx = i - sequence_length
            end_idx = i

            target = df_targets[target_idx]
            
            # Check target validity first
            if not np.isfinite(target):
                num_skipped_nan_target += 1
                continue
                
            # Numerical sequence slice
            seq = df_values[start_idx:end_idx]
            
            # Text embeddings sequence and temporal info
            text_seq = []
            temporal_seq = {'days_ago': [], 'event_categories': []}
            
            current_sequence_date = df_index_dates[target_idx] # Date of the target

            valid_embedding_found = False # Track if we found any valid news embedding
            for j in range(start_idx, end_idx):
                lookup_date = df_index_dates[j]
                days_ago_val = (current_sequence_date - lookup_date).days
                
                if lookup_date in news_lookup:
                    # Average all embeddings for this day
                    day_embeddings = []
                    for news_item in news_lookup[lookup_date]:
                        embedding = news_item.get('finbert_embedding')
                        if embedding is not None and isinstance(embedding, np.ndarray) and embedding.shape == (768,):
                            day_embeddings.append(embedding.astype(np.float32))
                    
                    if day_embeddings:
                        # Average the embeddings
                        avg_embedding = np.mean(day_embeddings, axis=0)
                        text_seq.append(avg_embedding)
                        valid_embedding_found = True
                    else:
                        # No valid embeddings found for this day's articles
                        text_seq.append(np.zeros(768, dtype=np.float32))
                    
                    temporal_seq['days_ago'].append(max(0, days_ago_val)) 
                    temporal_seq['event_categories'].append(0) # Placeholder category
                else:
                    text_seq.append(np.zeros(768, dtype=np.float32)) 
                    temporal_seq['days_ago'].append(max(0, days_ago_val)) # Use actual days ago even if no news
                    temporal_seq['event_categories'].append(0) # Default category
            
            # --- Convert lists to numpy arrays for checking ---
            try:
                seq_np = np.array(seq) # Already a slice, should be np array
                text_seq_np = np.array(text_seq, dtype=np.float32) # Ensure type
                days_ago_np = np.array(temporal_seq['days_ago'], dtype=np.float32) # Ensure type
                event_cat_np = np.array(temporal_seq['event_categories'], dtype=np.int64) # Ensure type

                # --- Rigorous Shape and Content Check ---
                shape_ok = (
                    seq_np.shape == (sequence_length, len(feature_cols)) and
                    text_seq_np.shape == (sequence_length, 768) and
                    days_ago_np.shape == (sequence_length,) and
                    event_cat_np.shape == (sequence_length,)
                )
                content_ok = (
                    np.all(np.isfinite(seq_np)) and
                    np.all(np.isfinite(text_seq_np)) and # Check embeddings too
                    np.all(np.isfinite(days_ago_np)) 
                    # event_cat_np should be integers, isfinite check not needed unless expecting NaNs
                )

                if shape_ok and content_ok:
                    sequences.append(seq_np)
                    targets.append(target) 
                    text_embeddings.append(text_seq_np)
                    temporal_info.append({
                        'days_ago': days_ago_np,
                        'event_categories': event_cat_np
                    })
                # --- Log skipped sequences ---
                elif not shape_ok:
                     num_skipped_shape_mismatch += 1
                     if num_skipped_shape_mismatch < 5: # Log first few occurrences
                          logger.warning(f"Skipping sequence ending at index {i} due to SHAPE mismatch.")
                          # (Log detailed shapes as before if needed)
                elif not content_ok:
                     # This should ideally not happen due to earlier filling, but check anyway
                     logger.warning(f"Skipping sequence ending at index {i} due to non-finite CONTENT.")
                     # Log which array had issues if needed
                     if not np.all(np.isfinite(seq_np)): logger.debug("Non-finite in seq_np")
                     if not np.all(np.isfinite(text_seq_np)): logger.debug("Non-finite in text_seq_np")
                     if not np.all(np.isfinite(days_ago_np)): logger.debug("Non-finite in days_ago_np")
                # -------------------------

            except Exception as check_err:
                 logger.error(f"Error validating sequence arrays at index {i}: {check_err}")
                 num_skipped_shape_mismatch += 1 # Count as shape/conversion error
            # ----------------------------------------------------
        # --- End of Loop ---

        if num_skipped_shape_mismatch > 0:
             logger.warning(f"Total sequences skipped due to shape/content issues: {num_skipped_shape_mismatch}")
        if num_skipped_nan_target > 0:
             logger.warning(f"Total sequences skipped due to NaN targets: {num_skipped_nan_target}")

        return sequences, targets, text_embeddings, temporal_info
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        # --- DEBUG: Track NaNs ---
        nan_loss_batches = 0
        nan_logit_batches = 0
        # --------------------------
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            try:
                text_embeddings = batch['text_embeddings'].to(self.device, non_blocking=True)
                numerical_features = batch['numerical_features'].to(self.device, non_blocking=True)
                targets = batch['targets'].to(self.device, non_blocking=True) # Targets should be Long
                
                temporal_info = None
                if 'temporal_info' in batch and batch['temporal_info']: 
                    temporal_info = {
                        key: value.to(self.device, 
                                      dtype=torch.long if key == 'event_categories' else torch.float, 
                                      non_blocking=True) 
                        for key, value in batch['temporal_info'].items()
                    }
            except Exception as e:
                 logger.error(f"Error moving batch {batch_idx} to device: {e}. Skipping batch.")
                 continue

            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True) 
            
            # Forward pass with mixed precision
            try:
                if self.use_mixed_precision and self.scaler: # Check scaler exists
                    with torch.cuda.amp.autocast():
                        outputs = self.model(text_embeddings, numerical_features, temporal_info)
                        # --- DEBUG: Check logits before loss ---
                        if not torch.isfinite(outputs['logits']).all():
                            logger.error(f"NaN/Inf logits detected BEFORE loss in batch {batch_idx} (Mixed Precision)")
                            nan_logit_batches += 1
                            raise RuntimeError("Non-finite logits detected") # Raise error to stop
                        # --------------------------------------
                        loss = self.criterion(outputs['logits'], targets.long()) 
                    
                    # Backward pass
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.gradient_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer) 
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else: # No mixed precision
                    outputs = self.model(text_embeddings, numerical_features, temporal_info)
                    # --- DEBUG: Check logits before loss ---
                    if not torch.isfinite(outputs['logits']).all():
                        logger.error(f"NaN/Inf logits detected BEFORE loss in batch {batch_idx} (Standard Precision)")
                        nan_logit_batches += 1
                        raise RuntimeError("Non-finite logits detected") # Raise error to stop
                    # --------------------------------------
                    loss = self.criterion(outputs['logits'], targets.long()) 
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    
                    # Optimizer step
                    self.optimizer.step()

                # --- DEBUG: Check loss AFTER backward but before scheduler step ---
                if not torch.isfinite(loss):
                     logger.error(f"NaN/Inf LOSS detected AFTER backward in batch {batch_idx}. LR={self.scheduler.get_lr():.6f}")
                     nan_loss_batches += 1
                     # Decide whether to continue or stop
                     # Option 1: Log and continue (might recover)
                     # Option 2: Raise error to stop training immediately
                     raise RuntimeError("Non-finite loss detected") 
                # -------------------------------------------------------------

                # Update scheduler (AFTER optimizer step)
                self.scheduler.step()

            except RuntimeError as e: # Catch the explicit errors raised above
                 logger.error(f"RuntimeError in batch {batch_idx}: {e}. Skipping batch.")
                 # Clear gradients before next batch if skipping
                 self.optimizer.zero_grad(set_to_none=True) 
                 continue # Skip update calculations for this batch

            except Exception as forward_err:
                 logger.error(f"Unhandled error during forward/backward pass in batch {batch_idx}: {forward_err}")
                 logger.debug(traceback.format_exc())
                 continue # Skip update calculations for this batch


            # Calculate accuracy (only if forward/backward succeeded)
            with torch.no_grad(): 
                _, predicted = torch.max(outputs['logits'], 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)
                # Check loss again just in case, though checked earlier
                if torch.isfinite(loss): 
                    total_loss += loss.item() * targets.size(0) # Weighted loss
                else:
                    # Should not happen if RuntimeError was raised, but as safety
                    logger.warning(f"Loss was non-finite in batch {batch_idx}, not adding to total_loss.")


        # --- DEBUG: Report NaN counts for the epoch ---
        if nan_logit_batches > 0:
             logger.warning(f"Epoch Summary: Encountered NaN/Inf logits in {nan_logit_batches} batches.")
        if nan_loss_batches > 0:
             logger.warning(f"Epoch Summary: Encountered NaN/Inf loss in {nan_loss_batches} batches.")
        # --------------------------------------------

        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan') # Return NaN if no samples
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        # --- DEBUG: Check avg_loss ---
        if not np.isfinite(avg_loss):
             logger.error(f"Average train loss for epoch is NaN/Inf! total_loss={total_loss}, total_samples={total_samples}")
        # -----------------------------

        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Move to device
                try:
                    text_embeddings = batch['text_embeddings'].to(self.device, non_blocking=True)
                    numerical_features = batch['numerical_features'].to(self.device, non_blocking=True)
                    targets = batch['targets'].to(self.device, non_blocking=True) # Targets should be Long
                    
                    temporal_info = None
                    if 'temporal_info' in batch and batch['temporal_info']: 
                        temporal_info = {
                            key: value.to(self.device, 
                                          dtype=torch.long if key == 'event_categories' else torch.float,
                                          non_blocking=True) 
                            for key, value in batch['temporal_info'].items()
                        }
                except Exception as e:
                     logger.error(f"Validation: Error moving batch {batch_idx} to device: {e}. Skipping batch.")
                     continue
                
                # Forward pass
                try:
                    # --- Use autocast even in validation if mixed precision was used in training ---
                    # Although gradients aren't needed, it ensures consistent dtypes for model layers
                    if self.use_mixed_precision:
                         with torch.cuda.amp.autocast():
                             outputs = self.model(text_embeddings, numerical_features, temporal_info)
                             loss = self.criterion(outputs['logits'], targets.long()) 
                    # -----------------------------------------------------------------------------
                    else:
                         outputs = self.model(text_embeddings, numerical_features, temporal_info)
                         loss = self.criterion(outputs['logits'], targets.long()) 

                    # --- DEBUG: Check validation loss ---
                    if not torch.isfinite(loss):
                         logger.warning(f"NaN/Inf validation loss detected in batch {batch_idx}.")
                         # Skip adding to total loss, or handle as needed
                    else:
                         total_loss += loss.item() * targets.size(0) # Weighted loss
                    # ------------------------------------
                    
                    # Collect predictions
                    _, predicted = torch.max(outputs['logits'], 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

                except Exception as val_forward_err:
                     logger.error(f"Validation Error during forward pass in batch {batch_idx}: {val_forward_err}")
                     logger.debug(traceback.format_exc())
                     # Skip batch if forward pass fails during validation
                     continue

        total_samples = len(all_targets)
        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan') # Return NaN if needed
        
        # Calculate metrics
        if not all_targets or not all_predictions or total_samples == 0:
             logger.warning("Validation set produced no valid targets or predictions.")
             return avg_loss, 0.0, {'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}

        # --- DEBUG: Check average validation loss ---
        if not np.isfinite(avg_loss):
             logger.error(f"Average validation loss is NaN/Inf! total_loss={total_loss}, total_samples={total_samples}")
        # -----------------------------------------

        accuracy = accuracy_score(all_targets, all_predictions)
        try:
             f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
             precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
             recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
             # --- DEBUG: Log validation report ---
             logger.debug(f"Validation classification report:\n{classification_report(all_targets, all_predictions, zero_division=0, target_names=['Down', 'Neutral', 'Up'])}") # Assuming 0,1,2
             # ------------------------------------
        except ValueError as metric_err:
             logger.error(f"Error calculating validation metrics: {metric_err}. Only one class might be present.")
             f1, precision, recall = 0.0, 0.0, 0.0

        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }
        
        return avg_loss, accuracy, metrics
    
    def train(self, train_dataset: StockDataset, val_dataset: StockDataset) -> Dict[str, Any]:
        """Main training loop."""
        if len(train_dataset) == 0 or len(val_dataset) == 0:
             logger.error("Training or validation dataset is empty. Cannot start training.")
             return { 'error': 'Empty dataset(s)' } # Simplified error return

        logger.info("Starting training...")
        
        # Create data loaders
        try:
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=self.config['training']['shuffle'],
                collate_fn=collate_fn, num_workers=0, 
                pin_memory=True if self.device.type == 'cuda' else False, 
                drop_last=True # Drop last incomplete batch
            )
            
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        except Exception as loader_err:
             logger.error(f"Failed to create DataLoaders: {loader_err}")
             logger.debug(traceback.format_exc())
             return {'error': f'DataLoader creation failed: {loader_err}'}

        # Re-calculate max_steps based on actual train_loader size
        steps_per_epoch = len(train_loader)
        if steps_per_epoch == 0:
             logger.error("Train loader has zero length after drop_last=True. Check dataset size and batch size.")
             return {'error': 'Train loader empty after drop_last'}
             
        calculated_max_steps = self.epochs * steps_per_epoch
        # Update scheduler's max_steps if necessary
        if self.final_max_steps < calculated_max_steps: # Use stored final_max_steps
             logger.info(f"Updating scheduler max_steps from {self.final_max_steps} to {calculated_max_steps} based on loader size.")
             self.scheduler.max_steps = calculated_max_steps
             self.final_max_steps = calculated_max_steps # Update stored value


        best_val_accuracy = -1.0 # Initialize to negative value
        epochs_run = 0 
        final_val_metrics = {} 
        
        try: 
            for epoch in range(self.epochs):
                epochs_run += 1
                logger.info(f"--- Starting Epoch {epoch+1}/{self.epochs} ---")
                
                # Training
                train_loss, train_accuracy = self.train_epoch(train_loader)
                
                # --- Check for NaN/Inf in training results ---
                if not np.isfinite(train_loss) or not np.isfinite(train_accuracy):
                     logger.error(f"Epoch {epoch+1}: Training resulted in NaN/Inf (Loss: {train_loss}, Acc: {train_accuracy}). Stopping training.")
                     raise RuntimeError("Training unstable (NaN/Inf detected)")
                # --------------------------------------------

                # Validation
                val_loss, val_accuracy, val_metrics = self.validate(val_loader)
                final_val_metrics = val_metrics 

                # --- Check for NaN/Inf in validation results ---
                if not np.isfinite(val_loss) or not np.isfinite(val_accuracy):
                     logger.warning(f"Epoch {epoch+1}: Validation resulted in NaN/Inf (Loss: {val_loss}, Acc: {val_accuracy}). Continuing, but check data/model.")
                     # Treat validation accuracy as very low for early stopping purposes
                     val_accuracy_for_stopping = -1.0 
                else:
                     val_accuracy_for_stopping = val_accuracy
                # ---------------------------------------------
                
                # Update history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['train_accuracy'].append(train_accuracy)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
                self.training_history['learning_rates'].append(self.scheduler.get_lr())
                
                # Log progress
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} Completed - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} | "
                    f"LR: {self.scheduler.get_lr():.6f}"
                )
                
                # Track best validation accuracy 
                if val_accuracy_for_stopping > best_val_accuracy:
                    logger.info(f"New best validation accuracy: {val_accuracy:.4f} (previous: {best_val_accuracy:.4f})")
                    best_val_accuracy = val_accuracy # Store the actual accuracy
                
                # Early stopping (use val_accuracy_for_stopping which handles NaN)
                if self.early_stopping(val_accuracy_for_stopping, self.model):
                    # Early stopping log message is now inside EarlyStopping class
                    break # Exit loop

        except RuntimeError as train_err: # Catch the NaN/Inf error raised in train_epoch
             logger.error(f"Stopping training due to runtime error: {train_err}")
             return { # Return error state
                 'best_val_accuracy': best_val_accuracy, 'final_val_metrics': final_val_metrics,
                 'training_history': self.training_history, 'total_epochs': epochs_run,
                 'error': f"Training stopped due to instability: {train_err}"
             }
        except Exception as train_loop_err:
             logger.error(f"Unhandled error during training loop at epoch {epochs_run}: {train_loop_err}")
             logger.error(traceback.format_exc())
             return { # Return error state
                 'best_val_accuracy': best_val_accuracy, 'final_val_metrics': final_val_metrics,
                 'training_history': self.training_history, 'total_epochs': epochs_run,
                 'error': f"Training loop failed: {train_loop_err}"
             }
        
        # --- End of training loop ---

        # Determine final best accuracy (if weights were restored, use the stored best score)
        final_best_val_acc = self.early_stopping.best_score if self.early_stopping.restored_weights else best_val_accuracy

        final_results = {
            'best_val_accuracy': final_best_val_acc,
            'final_val_metrics': final_val_metrics,
            'training_history': self.training_history,
            'total_epochs': epochs_run
        }
        
        # --- Make log message clearer about which accuracy is reported ---
        logger.info(f"Training completed. Best validation accuracy recorded during training: {final_best_val_acc:.4f} over {epochs_run} epochs.")
        if self.early_stopping.restored_weights:
             logger.info("Model state restored to the point of best validation accuracy.")
        # -----------------------------------------------------------------

        return final_results
    
    def save_model(self, filepath: str, metadata: Optional[Dict] = None):
        """Save model checkpoint."""
        # Ensure directory exists just before saving
        try:
             os.makedirs(os.path.dirname(filepath), exist_ok=True)
        except OSError as e:
             logger.error(f"Could not create directory for saving model at {os.path.dirname(filepath)}: {e}")
             return # Abort saving if directory cannot be created

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            # Save optimizer state only if needed for resuming training
            # 'optimizer_state_dict': self.optimizer.state_dict(), 
            'training_history': self.training_history,
            # Be cautious about saving the full config, might contain sensitive info or large objects
            # Consider saving only relevant parts or parameters used for this specific model
            'config': { 
                 'model': self.config.get('model'), 
                 'features': self.config.get('features'),
                 # Add other relevant sections as needed
                 },
            'metadata': metadata or {}
        }
        
        try:
            torch.save(checkpoint, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model to {filepath}: {e}")
            logger.debug(traceback.format_exc())

    
    def load_model(self, filepath: str):
        """Load model checkpoint."""
        if not os.path.exists(filepath):
             logger.error(f"Model file not found: {filepath}")
             raise FileNotFoundError(f"Model file not found: {filepath}")
        try:
            # --- Load checkpoint onto CPU first, then move model to device ---
            checkpoint = torch.load(filepath, map_location='cpu') 
            
            # Load model state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device) # Move model to target device AFTER loading state
            # ---------------------------------------------------------------

            self.training_history = checkpoint.get('training_history', {})
            loaded_config = checkpoint.get('config') # Get potentially saved config subset
            if loaded_config:
                 logger.info("Loaded configuration subset from checkpoint.")
                 # You might want to compare or merge this with the current self.config if needed

            logger.info(f"Model loaded from {filepath} and moved to {self.device}")
            
            return checkpoint.get('metadata', {})
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
            logger.debug(traceback.format_exc())
            raise

    # --- COMPATIBILITY METHODS FOR MAIN.PY ---
    def train(self, train_dataset: StockDataset, val_dataset: StockDataset) -> Dict[str, Any]:
        """Compatibility method that calls train_all_models and returns best model results."""
        results = self.train_all_models(train_dataset, val_dataset)
        if 'error' in results:
            return results
        
        # Get the best model results for backward compatibility
        best_name, best_model, best_results = self.get_best_model()
        if best_name:
            return {
                'best_model_name': best_name,
                'best_model': best_model,
                'results': best_results,
                'all_results': results
            }
        else:
            return {'error': 'No successful models trained'}
    
    def save_model(self, filepath: str, metadata: Optional[Dict] = None):
        """Compatibility method that saves the best model."""
        best_name, best_model, best_results = self.get_best_model()
        if best_name and best_name in self.results:
            trainer = self.results[best_name]['trainer']
            trainer.save_model(filepath, metadata)
            logger.info(f"Saved best model ({best_name}) to {filepath}")
        else:
            logger.error("No trained model available to save")
    # --- END COMPATIBILITY METHODS ---


class OptunaOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.optimization_config = config['optimization']
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        
        # Study settings
        self.n_trials = self.optimization_config['n_trials']
        self.timeout = self.optimization_config.get('timeout') # Use .get, might not be present
        self.study_name = self.optimization_config['study_name']
        
    def objective(self, trial: optuna.Trial, train_dataset: StockDataset, 
                 val_dataset: StockDataset) -> float:
        """Objective function for Optuna optimization."""
        
        # Sample hyperparameters
        search_spaces = self.optimization_config['search_spaces']
        
        try:
            # Check if space definition is list (categorical) or tuple/list of 2 (range)
            def _suggest(param_name, space):
                if isinstance(space, list) and len(space) > 2 and all(isinstance(x, (int, float, str)) for x in space):
                    return trial.suggest_categorical(param_name, space)
                elif isinstance(space, (list, tuple)) and len(space) == 2:
                    low, high = space
                    if isinstance(low, int) and isinstance(high, int):
                         # Check if range is small, suggest categorical instead if desired
                         if high - low < 5: # Example threshold
                              return trial.suggest_categorical(param_name, list(range(low, high + 1)))
                         else:
                              return trial.suggest_int(param_name, low, high)
                    elif isinstance(low, float) and isinstance(high, float):
                         log_scale = self.optimization_config.get('log_scale', {}).get(param_name, False)
                         return trial.suggest_float(param_name, low, high, log=log_scale)
                    else:
                         raise TypeError(f"Unsupported range types for {param_name}: {type(low)}, {type(high)}")
                else:
                     raise TypeError(f"Invalid search space definition for {param_name}: {space}")

            learning_rate = _suggest('learning_rate', search_spaces['learning_rate'])
            batch_size = _suggest('batch_size', search_spaces['batch_size'])
            hidden_dim = _suggest('hidden_dim', search_spaces['hidden_dim'])
            num_layers = _suggest('num_layers', search_spaces['num_layers'])
            dropout = _suggest('dropout', search_spaces['dropout'])
            attention_heads = _suggest('attention_heads', search_spaces['attention_heads'])
            # sequence_length = _suggest('sequence_length', search_spaces['sequence_length']) # Still complex

        except Exception as suggest_err:
             logger.error(f"Error suggesting parameters in trial {trial.number}: {suggest_err}")
             raise # Re-raise to fail the trial

        # Create a deep copy of the config to avoid modifying the original
        import copy
        trial_config = copy.deepcopy(self.config)

        # Update config with sampled parameters
        trial_config['training']['learning_rate'] = learning_rate
        trial_config['training']['batch_size'] = batch_size
        trial_config['model']['hidden_dim'] = hidden_dim
        trial_config['model']['num_layers'] = num_layers
        trial_config['model']['dropout'] = dropout
        trial_config['model']['attention']['num_heads'] = attention_heads
        # trial_config['model']['sequence_length'] = sequence_length 

        logger.info(f"--- Starting Optuna Trial {trial.number} ---")
        logger.info(f"Parameters: lr={learning_rate:.6f}, bs={batch_size}, hidden={hidden_dim}, layers={num_layers}, dropout={dropout:.3f}, heads={attention_heads}")

        try:
            # Need to import here if HybridStockPredictor isn't globally available
            # Ensure src is in path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.dirname(os.path.dirname(script_dir)) 
            if src_dir not in sys.path: sys.path.insert(0, src_dir)
            from models.hybrid_model import HybridStockPredictor # Relative import
            
            model = HybridStockPredictor(trial_config)
            trainer = StockPredictorTrainer(model, trial_config)
            
            # Use specific epochs for HPO trials if defined
            trainer.epochs = min(self.optimization_config.get('max_epochs_per_trial', 10), trainer.epochs) 
            logger.info(f"Trial {trial.number}: Training for max {trainer.epochs} epochs.")
            
            # Data preparation (assuming sequence length is fixed for HPO)
            train_dataset_trial, val_dataset_trial = train_dataset, val_dataset

            # Train model
            results = trainer.train(train_dataset_trial, val_dataset_trial)

            # --- Check for errors during training in the trial ---
            if 'error' in results:
                 logger.error(f"Trial {trial.number} training failed with error: {results['error']}")
                 # Return a very low score for failed trials
                 return -1.0 
            # ----------------------------------------------------
            
            # Report final best accuracy for pruning
            final_best_accuracy = results['best_val_accuracy']
            # --- Check if accuracy is valid before reporting ---
            if final_best_accuracy is None or not np.isfinite(final_best_accuracy):
                 logger.warning(f"Trial {trial.number} resulted in invalid accuracy ({final_best_accuracy}). Reporting as -1.0.")
                 final_best_accuracy = -1.0 # Report as worst score
            # -------------------------------------------------
            trial.report(final_best_accuracy, step=results['total_epochs']) # Report final value at last epoch step
            if trial.should_prune():
                 logger.info(f"Trial {trial.number} pruned with accuracy {final_best_accuracy:.4f}.")
                 raise optuna.exceptions.TrialPruned()

            logger.info(f"--- Trial {trial.number} Finished: Accuracy = {final_best_accuracy:.4f} ---")
            return final_best_accuracy
            
        except optuna.exceptions.TrialPruned as pr:
             logger.info(f"Trial {trial.number} pruned.")
             raise pr # Re-raise TrialPruned
        except Exception as e:
            logger.error(f"Trial {trial.number} failed unexpectedly: {str(e)}")
            logger.error(traceback.format_exc())
            return -1.0 # Return worse than any possible accuracy
    
    def optimize(self, train_dataset: StockDataset, val_dataset: StockDataset) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        logger.info(f"Starting hyperparameter optimization with Optuna for {self.n_trials} trials...")
        
        # Create study
        seed = self.config.get('environment', {}).get('seed', 42)
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=seed), 
            pruner=MedianPruner(
                 n_startup_trials=self.optimization_config.get('pruner_startup_trials', 5), 
                 n_warmup_steps=self.optimization_config.get('pruner_warmup_steps', 3), # Reduce warmup if epochs per trial is low
                 interval_steps=1 
                 ),
            study_name=self.study_name,
            # storage="sqlite:///optuna_study.db", # Example storage
            # load_if_exists=True
        )
        
        # Run optimization
        try:
            study.optimize(
                lambda trial: self.objective(trial, train_dataset, val_dataset),
                n_trials=self.n_trials,
                timeout=self.timeout if self.timeout and self.timeout > 0 else None, 
                show_progress_bar=True,
                catch=(Exception,) # Catch all exceptions during a trial
            )
        except KeyboardInterrupt:
             logger.warning("Optimization interrupted by user.")
        except Exception as e:
             logger.error(f"Optimization loop failed unexpectedly: {e}")
             logger.error(traceback.format_exc())


        # Get results
        best_params = {}
        best_value = -1.0 # Default to worst score
        try:
            # Check if any COMPLETED trials exist before accessing best_trial
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials:
                best_trial = study.best_trial
                best_params = best_trial.params
                best_value = best_trial.value
                logger.info(f"Optimization completed.")
                logger.info(f"Best Trial ({best_trial.number}): Accuracy = {best_value:.4f}")
                logger.info(f"Best Parameters: {best_params}")
            else:
                 logger.warning("No trials completed successfully. Cannot determine best parameters.")

        except Exception as e: # Catch potential errors accessing best_trial if study is empty/corrupt
             logger.error(f"Error retrieving best trial results: {e}")


        # Save results
        results_path = os.path.join(self.config['reporting']['output_path'], 'optimization')
        os.makedirs(results_path, exist_ok=True)
        
        # Recalculate trial states for summary
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        opt_results_summary = {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials_completed': len(completed_trials),
            'n_trials_pruned': len(pruned_trials),
            'n_trials_failed': len(failed_trials),
            'total_trials_submitted': len(study.trials)
        }
        
        # Save results summary
        try:
            summary_file = os.path.join(results_path, f'{self.study_name}_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(opt_results_summary, f, indent=2)
            logger.info(f"Optimization summary saved to {summary_file}")
        except Exception as e:
             logger.error(f"Failed to save optimization JSON results: {e}")
        
        # Save trials dataframe
        try:
             trials_df = study.trials_dataframe()
             trials_file = os.path.join(results_path, f'{self.study_name}_trials.csv')
             trials_df.to_csv(trials_file, index=False)
             logger.info(f"Optimization trials dataframe saved to {trials_file}")
        except Exception as e: # Catch error if dataframe is empty or fails
             logger.error(f"Failed to save optimization trials dataframe: {e}")

        return opt_results_summary


def main():
    """Test the new multi-model training functionality."""
    import yaml
    
    # Load configuration
    try:
        # Assuming config is in ../configs relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(script_dir), 'configs', 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded from {config_path}")
    except Exception as e:
         print(f"Error loading config: {e}")
         return

    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create dummy data for testing
    n_samples = 200 # Smaller sample for quick test
    seq_len = config['model']['sequence_length'] 
    text_dim = config['model']['embedding_dim'] 
    
    # --- MODIFIED: Use new config key ---
    try:
        num_dim = config['model']['numerical_input_dim'] 
    except KeyError:
        logger.error("Config missing 'model.numerical_input_dim'. Using 83 for test.")
        num_dim = 83 # Match the number of features generated
    n_classes = config['model']['output']['num_classes'] 
    # --- END MODIFICATION --- 
    
    logger.info("Generating dummy data...")
    num_data = np.random.randn(n_samples, seq_len, num_dim).astype(np.float32)
    # Fill NaNs/Infs for testing
    num_data[0, 0, 0] = np.nan
    num_data[1, 1, 1] = np.inf
    # Replace NaNs/Infs with valid values
    num_data = np.nan_to_num(num_data, nan=0.0, posinf=1.0, neginf=-1.0)
    
    text_data = np.random.randn(n_samples, seq_len, text_dim).astype(np.float32)
    
    # --- MODIFIED: Create targets based on num_classes ---
    if n_classes == 1:
        targets = np.random.randint(0, 2, n_samples).astype(np.float32) # Binary
    else:
        targets = np.random.randint(0, n_classes, n_samples).astype(np.int64) # Multiclass
    # --- END MODIFICATION ---
    
    # Simulate temporal data
    days_ago = np.random.randint(0, 30, (n_samples, seq_len)).astype(np.float32)
    event_cats = np.random.randint(0, config['features']['temporal']['event_weight_categories'], (n_samples, seq_len)).astype(np.int64)
    temporal_data = {'days_ago': days_ago, 'event_categories': event_cats}

    # Create datasets
    logger.info("Creating datasets...")
    split = int(0.8 * n_samples)
    
    # --- MODIFIED: Pass num_classes to StockDataset ---
    train_dataset = StockDataset(
        text_data[:split], num_data[:split], targets[:split], 
        n_classes, # Pass num_classes
        {k:v[:split] for k,v in temporal_data.items()}
        )
    val_dataset = StockDataset(
        text_data[split:], num_data[split:], targets[split:],
        n_classes, # Pass num_classes
        {k:v[split:] for k,v in temporal_data.items()}
        )
    # --- END MODIFICATION ---
    logger.info(f"Datasets created: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # Configure for quick testing
    test_config = config.copy()
    test_config['training']['epochs'] = 3  # Small number for testing
    test_config['training']['learning_rate'] = 0.001  # Reasonable learning rate
    test_config['training']['mixed_precision'] = False  # Disable for testing
    test_config['training']['early_stopping_patience'] = 2  # Quick early stopping
    
    # Optionally disable some models for faster testing
    test_config['models'] = {
        'hybrid': {'enabled': True, 'priority': 1},
        'logistic_regression': {'enabled': True, 'priority': 2},
        'basic_lstm': {'enabled': True, 'priority': 3},
        'bilstm': {'enabled': True, 'priority': 4},
        'random_forest': {'enabled': False, 'priority': 5},  # Disable for speed
        'bilstm_attention': {'enabled': False, 'priority': 6},  # Disable for speed
        'transformer': {'enabled': False, 'priority': 7}  # Disable for speed
    }

    
    # Create model
    try:
        from models.hybrid_model import HybridStockPredictor # Relative import should work now
        model = HybridStockPredictor(config)
        logger.info("Model created successfully.")
    except ImportError:
         logger.error("Could not import HybridStockPredictor. Ensure it's in src/models/ and src is in PYTHONPATH.")
         return
    except Exception as e:
         logger.error(f"Error creating model: {e}")
         logger.error(traceback.format_exc())
         return

    # Create trainer
    # --- Temporarily modify config for test if needed ---
    test_config = config.copy()
    test_config['training']['epochs'] = 2
    test_config['training']['learning_rate'] = 0.0001 # Use smaller LR for test
    test_config['training']['mixed_precision'] = False # Disable MP for test
    # ---------------------------------------------------
    try:
        trainer = StockPredictorTrainer(model, test_config)
        logger.info("Trainer created successfully.")
    except Exception as e:
         logger.error(f"Error creating trainer: {e}")
         logger.error(traceback.format_exc())
         return

    
    # Create multi-model trainer
    try:
        trainer = StockPredictorTrainer(test_config)
        logger.info("Multi-model trainer created successfully.")
    except Exception as e:
         logger.error(f"Error creating trainer: {e}")
         import traceback
         logger.error(traceback.format_exc())
         return

    # Test training all models
    logger.info("Starting multi-model training test...")
    try:
        results = trainer.train_all_models(train_dataset, val_dataset)
        
        print("\n" + "="*80)
        print("MULTI-MODEL TRAINING COMPLETED")
        print("="*80)
        
        if 'error' not in results:
            print("\nIndividual Results:")
            for model_name, result_data in results['individual_results'].items():
                if 'error' in result_data:
                    print(f"  {model_name}: FAILED - {result_data['error']}")
                else:
                    best_acc = result_data['results'].get('best_val_accuracy', 0)
                    print(f"  {model_name}: SUCCESS - Best Accuracy: {best_acc:.4f}")
            
            print(f"\nSummary:\n{results['summary']}")
            
            # Get best model
            best_name, best_model, best_results = trainer.get_best_model()
            if best_name:
                print(f"\nBest Model: {best_name} with accuracy {best_results['best_val_accuracy']:.4f}")
            
        else:
            print(f"Error during multi-model training: {results['error']}")
            
        print("="*80)
        
    except Exception as e:
        logger.error(f"Multi-model training failed during test: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    # Add src directory to path to allow relative imports when run directly
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(os.path.dirname(script_dir)) # Go up two levels
    if src_dir not in sys.path:
         sys.path.insert(0, src_dir)

    # --- Set logging level to DEBUG for main test ---
    logging.basicConfig(level=logging.DEBUG, 
                         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # -----------------------------------------------
    logger.info("Running basic trainer test with DEBUG logging...")
    main() # Run the test function
    logger.info("Basic trainer test finished.")