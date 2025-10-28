"""
Explainability module with attention visualization, temporal heatmaps, and event contribution analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logger = logging.getLogger(__name__)

class AttentionVisualizer:
    """Visualizes attention weights from the model."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.save_heatmaps = config['explainability']['attention_vis']['save_heatmaps']
        self.top_k = config['explainability']['attention_vis']['top_k_attention']
        self.output_dir = os.path.join(config['reporting']['output_path'], 'explanations', 'attention')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def visualize_self_attention(self, attention_weights: torch.Tensor, 
                               sequence_labels: List[str], 
                               title: str = "Self-Attention Weights") -> plt.Figure:
        """
        Visualize self-attention weights as a heatmap.
        
        Args:
            attention_weights: (num_heads, seq_len, seq_len) attention weights
            sequence_labels: Labels for sequence positions
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if attention_weights.dim() == 4:  # (batch, heads, seq, seq)
            attention_weights = attention_weights[0]  # Take first batch
        
        num_heads, seq_len, _ = attention_weights.shape
        
        # Average across heads or select specific head
        if num_heads > 1:
            avg_attention = attention_weights.mean(dim=0).cpu().numpy()
        else:
            avg_attention = attention_weights[0].cpu().numpy()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            avg_attention,
            xticklabels=sequence_labels,
            yticklabels=sequence_labels,
            cmap='Blues',
            annot=False,
            cbar=True,
            square=True,
            ax=ax
        )
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Attended to (Key)', fontsize=12)
        ax.set_ylabel('Attending from (Query)', fontsize=12)
        
        plt.tight_layout()
        
        if self.save_heatmaps:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"self_attention_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention heatmap to {filepath}")
        
        return fig
    
    def visualize_cross_attention(self, cross_attention_weights: torch.Tensor,
                                text_labels: List[str], 
                                numerical_labels: List[str],
                                title: str = "Cross-Modal Attention") -> plt.Figure:
        """
        Visualize cross-modal attention between text and numerical features.
        
        Args:
            cross_attention_weights: (num_heads, seq_len, seq_len) cross-attention weights
            text_labels: Labels for text sequence
            numerical_labels: Labels for numerical sequence
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if cross_attention_weights.dim() == 4:  # (batch, heads, seq, seq)
            cross_attention_weights = cross_attention_weights[0]  # Take first batch
        
        num_heads = cross_attention_weights.shape[0]
        
        # Average across heads
        if num_heads > 1:
            avg_attention = cross_attention_weights.mean(dim=0).cpu().numpy()
        else:
            avg_attention = cross_attention_weights[0].cpu().numpy()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(
            avg_attention,
            xticklabels=numerical_labels[:avg_attention.shape[1]],
            yticklabels=text_labels[:avg_attention.shape[0]],
            cmap='Reds',
            annot=False,
            cbar=True,
            ax=ax
        )
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Numerical Features (Key)', fontsize=12)
        ax.set_ylabel('Text Features (Query)', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if self.save_heatmaps:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cross_attention_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cross-attention heatmap to {filepath}")
        
        return fig
    
    def get_top_attended_features(self, attention_weights: torch.Tensor,
                                feature_names: List[str], 
                                k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get top-k most attended features.
        
        Args:
            attention_weights: Attention weights tensor
            feature_names: Names of features
            k: Number of top features to return
            
        Returns:
            List of (feature_name, attention_score) tuples
        """
        if k is None:
            k = self.top_k
        
        # Average attention across heads and sequence if needed
        if attention_weights.dim() > 2:
            attention_scores = attention_weights.mean(dim=tuple(range(attention_weights.dim()-1)))
        else:
            attention_scores = attention_weights.mean(dim=0)
        
        attention_scores = attention_scores.cpu().numpy()
        
        # Get top-k indices
        top_indices = np.argsort(attention_scores)[-k:][::-1]
        
        top_features = [(feature_names[i], attention_scores[i]) for i in top_indices]
        
        return top_features


class TemporalAnalyzer:
    """Analyzes temporal patterns and creates temporal heatmaps."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = os.path.join(config['reporting']['output_path'], 'explanations', 'temporal')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_temporal_heatmap(self, temporal_weights: torch.Tensor,
                              dates: List[datetime],
                              events: List[Dict],
                              title: str = "Temporal Event Importance") -> go.Figure:
        """
        Create interactive temporal heatmap showing event importance over time.
        
        Args:
            temporal_weights: (seq_len,) tensor of temporal weights
            dates: Corresponding dates
            events: List of event dictionaries
            title: Plot title
            
        Returns:
            Plotly figure
        """
        if len(temporal_weights) != len(dates):
            raise ValueError("Temporal weights and dates must have same length")
        
        # Convert to numpy
        weights = temporal_weights.cpu().numpy()
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'date': dates,
            'weight': weights,
            'event_count': [len([e for e in events if e.get('date', datetime.min).date() == d.date()]) for d in dates]
        })
        
        # Create interactive heatmap
        fig = go.Figure()
        
        # Add temporal weights as line plot
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['weight'],
            mode='lines+markers',
            name='Temporal Weight',
            line=dict(color='blue', width=2),
            hovertemplate='Date: %{x}<br>Weight: %{y:.3f}<extra></extra>'
        ))
        
        # Add event markers
        event_dates = [e.get('date', datetime.min) for e in events]
        event_weights = [weights[dates.index(d)] if d in dates else 0 for d in event_dates]
        event_texts = [e.get('title', 'Unknown Event')[:50] + '...' if len(e.get('title', '')) > 50 
                      else e.get('title', 'Unknown Event') for e in events]
        
        fig.add_trace(go.Scatter(
            x=event_dates,
            y=event_weights,
            mode='markers',
            name='News Events',
            marker=dict(
                size=10,
                color='red',
                symbol='star'
            ),
            text=event_texts,
            hovertemplate='Date: %{x}<br>Event: %{text}<br>Weight: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Temporal Weight',
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        return fig
    
    def analyze_decay_pattern(self, model_temporal_weights: nn.Module,
                            max_days: int = 30) -> Dict[str, Any]:
        """
        Analyze the learned temporal decay pattern.
        
        Args:
            model_temporal_weights: Temporal weights module from model
            max_days: Maximum days to analyze
            
        Returns:
            Analysis results dictionary
        """
        days_ago = torch.arange(0, max_days).float()
        event_categories = torch.zeros_like(days_ago).long()  # Assuming category 0
        
        # Calculate temporal weights
        with torch.no_grad():
            temporal_weights = model_temporal_weights.calculate_temporal_weights(
                days_ago.unsqueeze(0), event_categories.unsqueeze(0)
            ).squeeze(0)
        
        # Convert to numpy
        days = days_ago.numpy()
        weights = temporal_weights.numpy()
        
        # Fit exponential decay model
        from scipy.optimize import curve_fit
        
        def exp_decay(x, a, b):
            return a * np.exp(-b * x)
        
        try:
            popt, _ = curve_fit(exp_decay, days, weights)
            fitted_weights = exp_decay(days, *popt)
            
            # Calculate R-squared
            ss_res = np.sum((weights - fitted_weights) ** 2)
            ss_tot = np.sum((weights - np.mean(weights)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate half-life
            half_life = np.log(2) / popt[1] if popt[1] > 0 else np.inf
            
        except Exception as e:
            logger.warning(f"Could not fit exponential decay: {e}")
            popt = [0, 0]
            r_squared = 0
            half_life = np.nan
        
        # Create visualization
        fig = plt.figure(figsize=(10, 6))
        plt.plot(days, weights, 'bo-', label='Learned Weights', markersize=4)
        if popt[1] > 0:
            plt.plot(days, fitted_weights, 'r--', label=f'Fitted Decay (half-life: {half_life:.1f} days)')
        
        plt.xlabel('Days Ago')
        plt.ylabel('Temporal Weight')
        plt.title('Learned Temporal Decay Pattern')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"temporal_decay_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'days': days,
            'weights': weights,
            'fitted_params': popt,
            'r_squared': r_squared,
            'half_life': half_life,
            'plot_path': filepath
        }


class EventContributionAnalyzer:
    """Analyzes contribution of different events to predictions."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = os.path.join(config['reporting']['output_path'], 'explanations', 'events')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def analyze_event_contributions(self, model: nn.Module, 
                                  text_embeddings: torch.Tensor,
                                  numerical_features: torch.Tensor,
                                  events_data: List[Dict],
                                  target_class: int = None) -> Dict[str, Any]:
        """
        Analyze how different events contribute to model predictions.
        
        Args:
            model: Trained model
            text_embeddings: Text embeddings tensor
            numerical_features: Numerical features tensor
            events_data: List of event dictionaries
            target_class: Specific class to analyze (optional)
            
        Returns:
            Event contribution analysis results
        """
        model.eval()
        device = next(model.parameters()).device
        
        text_embeddings = text_embeddings.to(device)
        numerical_features = numerical_features.to(device)
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = model(text_embeddings, numerical_features)
            baseline_probs = torch.softmax(baseline_output['logits'], dim=-1)
        
        if target_class is None:
            target_class = torch.argmax(baseline_probs, dim=-1).item()
        
        baseline_prob = baseline_probs[0, target_class].item()
        
        # Analyze contribution of each event
        event_contributions = []
        
        for i, event in enumerate(events_data):
            # Create masked version (zero out this event's embedding)
            masked_text = text_embeddings.clone()
            if i < masked_text.shape[1]:
                masked_text[0, i, :] = 0  # Zero out event embedding
            
            with torch.no_grad():
                masked_output = model(masked_text, numerical_features)
                masked_probs = torch.softmax(masked_output['logits'], dim=-1)
            
            masked_prob = masked_probs[0, target_class].item()
            contribution = baseline_prob - masked_prob
            
            event_contributions.append({
                'event_index': i,
                'event_title': event.get('title', 'Unknown'),
                'event_date': event.get('date', datetime.min),
                'sentiment_score': event.get('sentiment_score', 0),
                'contribution': contribution,
                'baseline_prob': baseline_prob,
                'masked_prob': masked_prob
            })
        
        # Sort by contribution magnitude
        event_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        # Create visualization
        self._visualize_event_contributions(event_contributions)
        
        return {
            'target_class': target_class,
            'baseline_probability': baseline_prob,
            'event_contributions': event_contributions,
            'total_events': len(events_data)
        }
    
    def _visualize_event_contributions(self, contributions: List[Dict]):
        """Create visualization of event contributions."""
        # Take top 20 events for visualization
        top_contributions = contributions[:20]
        
        titles = [c['event_title'][:30] + '...' if len(c['event_title']) > 30 
                 else c['event_title'] for c in top_contributions]
        contribution_values = [c['contribution'] for c in top_contributions]
        colors = ['green' if c > 0 else 'red' for c in contribution_values]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(titles)), contribution_values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(titles)))
        ax.set_yticklabels(titles)
        ax.set_xlabel('Contribution to Prediction')
        ax.set_title('Event Contributions to Model Prediction')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, contribution_values)):
            ax.text(value + 0.001 if value >= 0 else value - 0.001, 
                   i, f'{value:.3f}', 
                   ha='left' if value >= 0 else 'right', 
                   va='center', fontsize=8)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"event_contributions_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved event contributions plot to {filepath}")


class CounterfactualAnalyzer:
    """Performs counterfactual analysis to understand model behavior."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = os.path.join(config['reporting']['output_path'], 'explanations', 'counterfactual')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_counterfactuals(self, model: nn.Module,
                               text_embeddings: torch.Tensor,
                               numerical_features: torch.Tensor,
                               target_class: int,
                               perturbation_strength: float = 0.1) -> Dict[str, Any]:
        """
        Generate counterfactual examples by perturbing inputs.
        
        Args:
            model: Trained model
            text_embeddings: Original text embeddings
            numerical_features: Original numerical features
            target_class: Target class for counterfactual
            perturbation_strength: Strength of perturbations
            
        Returns:
            Counterfactual analysis results
        """
        model.eval()
        device = next(model.parameters()).device
        
        text_embeddings = text_embeddings.to(device)
        numerical_features = numerical_features.to(device)
        
        # Get original prediction
        with torch.no_grad():
            original_output = model(text_embeddings, numerical_features)
            original_probs = torch.softmax(original_output['logits'], dim=-1)
            original_class = torch.argmax(original_probs, dim=-1).item()
        
        counterfactuals = []
        
        # Try different perturbation strategies
        strategies = [
            'random_text', 'random_numerical', 'sentiment_flip', 
            'feature_scaling', 'temporal_shift'
        ]
        
        for strategy in strategies:
            perturbed_text, perturbed_num = self._apply_perturbation(
                text_embeddings, numerical_features, strategy, perturbation_strength
            )
            
            with torch.no_grad():
                perturbed_output = model(perturbed_text, perturbed_num)
                perturbed_probs = torch.softmax(perturbed_output['logits'], dim=-1)
                perturbed_class = torch.argmax(perturbed_probs, dim=-1).item()
            
            if perturbed_class == target_class and perturbed_class != original_class:
                counterfactuals.append({
                    'strategy': strategy,
                    'original_class': original_class,
                    'counterfactual_class': perturbed_class,
                    'original_probs': original_probs.cpu().numpy(),
                    'counterfactual_probs': perturbed_probs.cpu().numpy(),
                    'perturbation_strength': perturbation_strength
                })
        
        return {
            'original_class': original_class,
            'target_class': target_class,
            'counterfactuals_found': len(counterfactuals),
            'counterfactuals': counterfactuals
        }
    
    def _apply_perturbation(self, text_embeddings: torch.Tensor,
                          numerical_features: torch.Tensor,
                          strategy: str,
                          strength: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply specific perturbation strategy."""
        perturbed_text = text_embeddings.clone()
        perturbed_num = numerical_features.clone()
        
        if strategy == 'random_text':
            noise = torch.randn_like(perturbed_text) * strength
            perturbed_text += noise
            
        elif strategy == 'random_numerical':
            noise = torch.randn_like(perturbed_num) * strength
            perturbed_num += noise
            
        elif strategy == 'sentiment_flip':
            # Flip sign of text embeddings (simplified sentiment flip)
            perturbed_text *= -1
            
        elif strategy == 'feature_scaling':
            # Scale numerical features
            perturbed_num *= (1 + strength)
            
        elif strategy == 'temporal_shift':
            # Shift sequence positions
            shift = max(1, int(strength * perturbed_text.shape[1]))
            perturbed_text = torch.roll(perturbed_text, shifts=shift, dims=1)
            perturbed_num = torch.roll(perturbed_num, shifts=shift, dims=1)
        
        return perturbed_text, perturbed_num


class ExplainabilityPipeline:
    """Main pipeline for model explainability analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.attention_visualizer = AttentionVisualizer(config)
        self.temporal_analyzer = TemporalAnalyzer(config)
        self.event_analyzer = EventContributionAnalyzer(config)
        self.counterfactual_analyzer = CounterfactualAnalyzer(config)
        
    def run_full_analysis(self, model: nn.Module,
                         sample_data: Dict[str, torch.Tensor],
                         events_data: List[Dict],
                         dates: List[datetime]) -> Dict[str, Any]:
        """
        Run complete explainability analysis.
        
        Args:
            model: Trained model
            sample_data: Sample input data
            events_data: Events data for analysis
            dates: Corresponding dates
            
        Returns:
            Complete explainability results
        """
        logger.info("Running explainability analysis...")
        
        results = {}
        
        # Get model outputs with attention weights
        model.eval()
        with torch.no_grad():
            outputs = model(
                sample_data['text_embeddings'],
                sample_data['numerical_features'],
                sample_data.get('temporal_info')
            )
        
        # 1. Attention Analysis
        if outputs.get('text_attention') is not None:
            logger.info("Analyzing attention patterns...")
            
            # Self-attention visualization
            sequence_labels = [f"T-{i}" for i in range(sample_data['text_embeddings'].shape[1])]
            attention_fig = self.attention_visualizer.visualize_self_attention(
                outputs['text_attention'], sequence_labels
            )
            results['attention_visualization'] = attention_fig
            
            # Top attended features
            feature_names = [f"Feature_{i}" for i in range(sample_data['numerical_features'].shape[-1])]
            top_features = self.attention_visualizer.get_top_attended_features(
                outputs['num_attention'], feature_names
            )
            results['top_attended_features'] = top_features
        
        # 2. Temporal Analysis
        if hasattr(model, 'temporal_weights'):
            logger.info("Analyzing temporal patterns...")
            
            # Temporal heatmap
            temporal_weights = torch.randn(len(dates))  # Placeholder
            temporal_fig = self.temporal_analyzer.create_temporal_heatmap(
                temporal_weights, dates, events_data
            )
            results['temporal_heatmap'] = temporal_fig
            
            # Decay pattern analysis
            decay_analysis = self.temporal_analyzer.analyze_decay_pattern(
                model.temporal_weights
            )
            results['decay_analysis'] = decay_analysis
        
        # 3. Event Contribution Analysis
        logger.info("Analyzing event contributions...")
        event_contributions = self.event_analyzer.analyze_event_contributions(
            model,
            sample_data['text_embeddings'],
            sample_data['numerical_features'],
            events_data
        )
        results['event_contributions'] = event_contributions
        
        # 4. Counterfactual Analysis
        logger.info("Generating counterfactuals...")
        target_class = torch.argmax(outputs['logits'], dim=-1).item()
        alternative_class = (target_class + 1) % 3  # Next class
        
        counterfactuals = self.counterfactual_analyzer.generate_counterfactuals(
            model,
            sample_data['text_embeddings'],
            sample_data['numerical_features'],
            alternative_class
        )
        results['counterfactuals'] = counterfactuals
        
        logger.info("Explainability analysis completed")
        return results


def main():
    """Test the explainability functionality."""
    import yaml
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    batch_size = 1
    seq_len = 60
    text_dim = 768
    num_dim = 50
    num_heads = 8
    
    # Test attention visualization
    attention_visualizer = AttentionVisualizer(config)
    
    # Create dummy attention weights
    attention_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
    attention_weights = torch.softmax(attention_weights, dim=-1)
    
    sequence_labels = [f"Day-{i}" for i in range(seq_len)]
    
    fig = attention_visualizer.visualize_self_attention(
        attention_weights, sequence_labels
    )
    print("Created attention visualization")
    
    # Test temporal analysis
    temporal_analyzer = TemporalAnalyzer(config)
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    temporal_weights = torch.exp(-0.1 * torch.arange(30).float())
    events = [{'date': dates[i], 'title': f'Event {i}'} for i in range(0, 30, 5)]
    
    temporal_fig = temporal_analyzer.create_temporal_heatmap(
        temporal_weights, dates.to_list(), events
    )
    print("Created temporal heatmap")


if __name__ == "__main__":
    main()