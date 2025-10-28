"""
Utility functions for the stock prediction system.
"""

import os
import sys
import json
import yaml
import pickle
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List, Optional
from datetime import datetime
import random
from pathlib import Path

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Set up logging configuration."""
    logging_config = config.get('logging', {})
    
    # Create logs directory
    log_file = logging_config.get('file', 'logs/stock_predictor.log')
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, logging_config.get('level', 'INFO')),
        format=logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    
    return logger


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str = 'configs/config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file: {e}")


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def create_directory_structure(base_path: str, structure: Dict[str, Any]) -> None:
    """Create directory structure from nested dictionary."""
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_directory_structure(path, content)
        else:
            os.makedirs(path, exist_ok=True)


def save_results(results: Dict[str, Any], output_path: str, 
                format: str = 'json') -> None:
    """Save results to file in specified format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format.lower() == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format.lower() == 'pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    elif format.lower() == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(input_path: str, format: str = None) -> Dict[str, Any]:
    """Load results from file, auto-detecting format if not provided."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Results file not found: {input_path}")
    
    if format is None:
        # Auto-detect format from extension
        extension = Path(input_path).suffix.lower()
        if extension == '.json':
            format = 'json'
        elif extension in ['.pkl', '.pickle']:
            format = 'pickle'
        elif extension in ['.yml', '.yaml']:
            format = 'yaml'
        else:
            format = 'json'  # Default
    
    if format.lower() == 'json':
        with open(input_path, 'r') as f:
            return json.load(f)
    elif format.lower() == 'pickle':
        with open(input_path, 'rb') as f:
            return pickle.load(f)
    elif format.lower() == 'yaml':
        with open(input_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and return information."""
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None
    }
    
    return gpu_info


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def calculate_memory_usage() -> Dict[str, float]:
    """Calculate memory usage information."""
    memory_info = {}
    
    # CPU memory (if psutil is available)
    try:
        import psutil
        process = psutil.Process()
        memory_info['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024
        memory_info['cpu_memory_percent'] = process.memory_percent()
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info['system_memory_total_gb'] = system_memory.total / 1024 / 1024 / 1024
        memory_info['system_memory_used_percent'] = system_memory.percent
    except ImportError:
        memory_info['cpu_memory_mb'] = None
        memory_info['cpu_memory_percent'] = None
    
    # GPU memory
    if torch.cuda.is_available():
        memory_info['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        memory_info['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        memory_info['gpu_memory_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        memory_info['gpu_memory_allocated_mb'] = None
        memory_info['gpu_memory_reserved_mb'] = None
        memory_info['gpu_memory_max_allocated_mb'] = None
    
    return memory_info


def validate_data_shapes(data: Dict[str, Any], expected_shapes: Dict[str, tuple]) -> bool:
    """Validate that data has expected shapes."""
    for key, expected_shape in expected_shapes.items():
        if key not in data:
            raise ValueError(f"Missing required data key: {key}")
        
        actual_shape = data[key].shape if hasattr(data[key], 'shape') else len(data[key])
        
        # Allow None for flexible dimensions
        if expected_shape is not None:
            if isinstance(actual_shape, int):
                actual_shape = (actual_shape,)
            
            if len(actual_shape) != len(expected_shape):
                raise ValueError(f"Shape mismatch for {key}: expected {len(expected_shape)} dimensions, got {len(actual_shape)}")
            
            for i, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
                if expected is not None and expected != actual:
                    raise ValueError(f"Shape mismatch for {key} at dimension {i}: expected {expected}, got {actual}")
    
    return True


def normalize_features(features: np.ndarray, method: str = 'zscore', 
                      fit_params: Optional[Dict[str, Any]] = None) -> tuple:
    """
    Normalize features using specified method.
    
    Args:
        features: Feature array to normalize
        method: Normalization method ('zscore', 'minmax', 'robust')
        fit_params: Pre-computed parameters for normalization
        
    Returns:
        Tuple of (normalized_features, normalization_params)
    """
    if method == 'zscore':
        if fit_params is None:
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            params = {'mean': mean, 'std': std}
        else:
            mean, std = fit_params['mean'], fit_params['std']
            params = fit_params
        
        normalized = (features - mean) / std
        
    elif method == 'minmax':
        if fit_params is None:
            min_val = np.min(features, axis=0)
            max_val = np.max(features, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1  # Avoid division by zero
            params = {'min': min_val, 'range': range_val}
        else:
            min_val, range_val = fit_params['min'], fit_params['range']
            params = fit_params
        
        normalized = (features - min_val) / range_val
        
    elif method == 'robust':
        if fit_params is None:
            median = np.median(features, axis=0)
            mad = np.median(np.abs(features - median), axis=0)
            mad[mad == 0] = 1  # Avoid division by zero
            params = {'median': median, 'mad': mad}
        else:
            median, mad = fit_params['median'], fit_params['mad']
            params = fit_params
        
        normalized = (features - median) / mad
        
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    return normalized, params


def create_experiment_id() -> str:
    """Create unique experiment ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_system_info(config: Dict[str, Any]) -> None:
    """Print system information and configuration summary."""
    print("=" * 80)
    print("STOCK PREDICTION SYSTEM")
    print("=" * 80)
    
    # System info
    gpu_info = check_gpu_availability()
    memory_info = calculate_memory_usage()
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {gpu_info['cuda_available']}")
    if gpu_info['cuda_available']:
        print(f"GPU device: {gpu_info['device_name']}")
        print(f"GPU memory allocated: {memory_info['gpu_memory_allocated_mb']:.1f} MB")
    
    if memory_info['cpu_memory_mb']:
        print(f"CPU memory usage: {memory_info['cpu_memory_mb']:.1f} MB ({memory_info['cpu_memory_percent']:.1f}%)")
    
    print("-" * 80)
    
    # Configuration summary
    print("CONFIGURATION SUMMARY:")
    print(f"Model type: {config['model'].get('fusion', {}).get('fusion_dim', 'N/A')}")
    print(f"Sequence length: {config['model']['sequence_length']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Target symbols: {len(config['data']['symbols'])}")
    print(f"History days: {config['data']['history_days']}")
    
    print("=" * 80)


class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
        
    def update(self, step: int = None, message: str = None) -> None:
        """Update progress."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        progress = self.current_step / self.total_steps
        elapsed = (datetime.now() - self.start_time).total_seconds()
        eta = elapsed / progress - elapsed if progress > 0 else 0
        
        bar_length = 50
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        status_msg = f"\r{self.description}: |{bar}| {progress*100:.1f}% "
        status_msg += f"({self.current_step}/{self.total_steps}) "
        status_msg += f"Elapsed: {format_time(elapsed)} "
        status_msg += f"ETA: {format_time(eta)}"
        
        if message:
            status_msg += f" - {message}"
        
        print(status_msg, end='', flush=True)
        
        if self.current_step >= self.total_steps:
            print()  # New line when complete
    
    def finish(self, message: str = "Complete") -> None:
        """Mark progress as finished."""
        self.current_step = self.total_steps
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\r{self.description}: {message} (Total time: {format_time(elapsed)})")


def main():
    """Test utility functions."""
    # Test configuration loading
    try:
        config = load_config('configs/config.yaml')
        print("Configuration loaded successfully")
        print(f"Number of symbols: {len(config['data']['symbols'])}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
    
    # Test GPU availability
    gpu_info = check_gpu_availability()
    print(f"CUDA available: {gpu_info['cuda_available']}")
    
    # Test memory usage
    memory_info = calculate_memory_usage()
    print(f"Memory info: {memory_info}")
    
    # Test progress tracker
    tracker = ProgressTracker(100, "Testing")
    for i in range(100):
        tracker.update(message=f"Step {i}")
        import time
        time.sleep(0.01)  # Simulate work
    tracker.finish()


if __name__ == "__main__":
    main()