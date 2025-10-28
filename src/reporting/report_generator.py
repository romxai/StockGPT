"""
Automated reporting module with LaTeX table generation, figure creation, and comprehensive reporting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import json
from datetime import datetime
from jinja2 import Template
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LaTeXTableGenerator:
    """Generates LaTeX tables for research papers."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.latex_config = config['reporting']['latex']
        self.format_style = self.latex_config['format']
        self.caption_style = self.latex_config['caption_style']
        self.output_dir = os.path.join(config['reporting']['paper_path'], 'tables')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_performance_table(self, results: Dict[str, Dict[str, float]], 
                                 caption: str = "Model Performance Comparison") -> str:
        """
        Generate performance comparison table in LaTeX format.
        
        Args:
            results: Dictionary of model_name -> metrics
            caption: Table caption
            
        Returns:
            LaTeX table string
        """
        # Extract metrics
        models = list(results.keys())
        metrics = list(results[models[0]].keys()) if models else []
        
        # Create DataFrame
        df = pd.DataFrame(results).T
        
        # Format numbers
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                df[col] = df[col].apply(lambda x: f"{x:.4f}")
        
        # Generate LaTeX table
        latex_table = self._create_latex_table(
            df, caption, label="tab:performance_comparison"
        )
        
        # Save to file
        filename = "performance_comparison.tex"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"Generated performance table: {filepath}")
        return latex_table
    
    def generate_ablation_table(self, ablation_results: Dict[str, Any], 
                               caption: str = "Ablation Study Results") -> str:
        """Generate ablation study results table."""
        if 'contributions' not in ablation_results:
            return ""
        
        contributions = ablation_results['contributions']
        
        # Create DataFrame
        data = []
        for component, contribution in contributions.items():
            data.append({
                'Component': component.replace('_', ' ').title(),
                'Contribution': f"{contribution:.4f}",
                'Relative Impact': f"{contribution / max(contributions.values()) * 100:.1f}\\%"
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Contribution', ascending=False)
        
        latex_table = self._create_latex_table(
            df, caption, label="tab:ablation_study"
        )
        
        # Save to file
        filename = "ablation_study.tex"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"Generated ablation table: {filepath}")
        return latex_table
    
    def generate_sector_table(self, sector_results: Dict[str, Any], 
                             caption: str = "Multi-Sector Performance Analysis") -> str:
        """Generate multi-sector results table."""
        if 'sector_results' not in sector_results:
            return ""
        
        # Create DataFrame from sector results
        data = []
        for sector, results in sector_results['sector_results'].items():
            data.append({
                'Sector': sector.title(),
                'Accuracy': f"{results.get('accuracy', 0):.4f}",
                'F1-Score': f"{results.get('f1_score', 0):.4f}",
                'Symbols': results.get('n_symbols', 0)
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Accuracy', ascending=False)
        
        # Add summary row
        if 'cross_sector_stats' in sector_results:
            stats = sector_results['cross_sector_stats']
            summary_row = {
                'Sector': '\\textbf{Overall}',
                'Accuracy': f"\\textbf{{{stats.get('mean_accuracy', 0):.4f} ± {stats.get('std_accuracy', 0):.4f}}}",
                'F1-Score': f"\\textbf{{{stats.get('mean_f1', 0):.4f} ± {stats.get('std_f1', 0):.4f}}}",
                'Symbols': '\\textbf{All}'
            }
            df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
        
        latex_table = self._create_latex_table(
            df, caption, label="tab:sector_analysis"
        )
        
        # Save to file
        filename = "sector_analysis.tex"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"Generated sector table: {filepath}")
        return latex_table
    
    def generate_hyperparameter_table(self, optimization_results: Dict[str, Any],
                                     caption: str = "Optimal Hyperparameters") -> str:
        """Generate optimal hyperparameters table."""
        if 'best_params' not in optimization_results:
            return ""
        
        best_params = optimization_results['best_params']
        
        # Create DataFrame
        data = []
        for param, value in best_params.items():
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            data.append({
                'Parameter': param.replace('_', ' ').title(),
                'Optimal Value': formatted_value
            })
        
        df = pd.DataFrame(data)
        
        # Add performance row
        if 'best_value' in optimization_results:
            best_value = optimization_results['best_value']
            perf_row = {
                'Parameter': '\\textbf{Best Accuracy}',
                'Optimal Value': f"\\textbf{{{best_value:.4f}}}"
            }
            df = pd.concat([df, pd.DataFrame([perf_row])], ignore_index=True)
        
        latex_table = self._create_latex_table(
            df, caption, label="tab:hyperparameters"
        )
        
        # Save to file
        filename = "hyperparameters.tex"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"Generated hyperparameters table: {filepath}")
        return latex_table
    
    def generate_significance_table(self, significance_results: Dict[str, Any],
                                   caption: str = "Statistical Significance Tests") -> str:
        """Generate statistical significance test results table."""
        if 'pairwise_comparisons' not in significance_results:
            return ""
        
        comparisons = significance_results['pairwise_comparisons']
        
        # Create DataFrame
        data = []
        for comparison, results in comparisons.items():
            model1, model2 = comparison.split('_vs_')
            t_test = results['paired_t_test']
            wilcoxon = results['wilcoxon_test']
            
            data.append({
                'Comparison': f"{model1} vs {model2}",
                'T-test p-value': f"{t_test['p_value']:.4f}",
                'T-test Significant': 'Yes' if t_test['significant'] else 'No',
                'Wilcoxon p-value': f"{wilcoxon['p_value']:.4f}",
                'Wilcoxon Significant': 'Yes' if wilcoxon['significant'] else 'No',
                'Effect Size': f"{t_test['effect_size']:.4f}"
            })
        
        df = pd.DataFrame(data)
        
        latex_table = self._create_latex_table(
            df, caption, label="tab:significance_tests"
        )
        
        # Save to file
        filename = "significance_tests.tex"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"Generated significance tests table: {filepath}")
        return latex_table
    
    def _create_latex_table(self, df: pd.DataFrame, caption: str, label: str) -> str:
        """Create LaTeX table from DataFrame."""
        # Generate column specification
        n_cols = len(df.columns)
        if n_cols <= 3:
            col_spec = 'l' + 'c' * (n_cols - 1)
        else:
            col_spec = 'l' + 'c' * (n_cols - 1)
        
        # IEEE format template
        template = Template("""
\\begin{table}[htbp]
\\centering
\\caption{{{ caption }}}
\\label{{{ label }}}
\\begin{tabular}{ {{ col_spec }} }
\\hline
{{ header }}
\\hline
{{ rows }}
\\hline
\\end{tabular}
\\end{table}
        """.strip())
        
        # Create header
        header = " & ".join([f"\\textbf{{{col}}}" for col in df.columns]) + " \\\\"
        
        # Create rows
        rows = []
        for _, row in df.iterrows():
            row_str = " & ".join([str(val) for val in row.values]) + " \\\\"
            rows.append(row_str)
        
        rows_str = "\n".join(rows)
        
        latex_table = template.render(
            caption=caption,
            label=label,
            col_spec=col_spec,
            header=header,
            rows=rows_str
        )
        
        return latex_table


class FigureGenerator:
    """Generates publication-quality figures for research papers."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.figure_config = config['reporting']['figures']
        self.dpi = self.figure_config['dpi']
        self.format = self.figure_config['format']
        self.style = self.figure_config['style']
        self.output_dir = os.path.join(config['reporting']['paper_path'], 'figures')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use(self.style)
        sns.set_palette("husl")
        
    def create_performance_comparison(self, results: Dict[str, Dict[str, float]], 
                                    title: str = "Model Performance Comparison") -> str:
        """Create model performance comparison bar chart."""
        models = list(results.keys())
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        
        # Prepare data
        data = []
        for model in models:
            for metric in metrics:
                if metric in results[model]:
                    data.append({
                        'Model': model,
                        'Metric': metric.replace('_', ' ').title(),
                        'Score': results[model][metric]
                    })
        
        df = pd.DataFrame(data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar chart
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            metric_data = df[df['Metric'] == metric.replace('_', ' ').title()]
            scores = [metric_data[metric_data['Model'] == model]['Score'].iloc[0] 
                     if len(metric_data[metric_data['Model'] == model]) > 0 else 0 
                     for model in models]
            
            ax.bar(x + i * width, scores, width, 
                  label=metric.replace('_', ' ').title(), alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"performance_comparison.{self.format}"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated performance comparison figure: {filepath}")
        return filepath
    
    def create_training_curves(self, training_history: Dict[str, List[float]],
                              title: str = "Training and Validation Curves") -> str:
        """Create training and validation curves plot."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        epochs = range(1, len(training_history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, training_history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, training_history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate curve
        ax3.plot(epochs, training_history['learning_rates'], 'g-', label='Learning Rate', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        filename = f"training_curves.{self.format}"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated training curves figure: {filepath}")
        return filepath
    
    def create_confusion_matrices(self, confusion_matrices: Dict[str, np.ndarray],
                                 class_names: List[str] = None,
                                 title: str = "Confusion Matrices") -> str:
        """Create confusion matrices heatmaps for multiple models."""
        if class_names is None:
            class_names = ['Down', 'Neutral', 'Up']
        
        n_models = len(confusion_matrices)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        filename = f"confusion_matrices.{self.format}"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated confusion matrices figure: {filepath}")
        return filepath
    
    def create_ablation_waterfall(self, contributions: Dict[str, float],
                                 title: str = "Ablation Study Contributions") -> str:
        """Create waterfall chart for ablation study."""
        components = list(contributions.keys())
        values = list(contributions.values())
        
        # Sort by contribution
        sorted_data = sorted(zip(components, values), key=lambda x: x[1], reverse=True)
        components, values = zip(*sorted_data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create waterfall chart
        colors = ['green' if v > 0 else 'red' for v in values]
        bars = ax.bar(range(len(components)), values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001 if height >= 0 else height - 0.001,
                   f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        ax.set_xlabel('Components', fontsize=12)
        ax.set_ylabel('Contribution to Performance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels([c.replace('_', ' ').title() for c in components], rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"ablation_waterfall.{self.format}"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated ablation waterfall figure: {filepath}")
        return filepath
    
    def create_sector_performance(self, sector_results: Dict[str, Any],
                                 title: str = "Multi-Sector Performance") -> str:
        """Create sector performance visualization."""
        if 'sector_results' not in sector_results:
            return ""
        
        sectors = list(sector_results['sector_results'].keys())
        accuracies = [results['accuracy'] for results in sector_results['sector_results'].values()]
        f1_scores = [results['f1_score'] for results in sector_results['sector_results'].values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy plot
        bars1 = ax1.bar(sectors, accuracies, alpha=0.7, color='skyblue')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Accuracy by Sector', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # F1-score plot
        bars2 = ax2.bar(sectors, f1_scores, alpha=0.7, color='lightcoral')
        ax2.set_ylabel('F1-Score', fontsize=12)
        ax2.set_title('F1-Score by Sector', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        for ax in [ax1, ax2]:
            ax.set_xticklabels(sectors, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        filename = f"sector_performance.{self.format}"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated sector performance figure: {filepath}")
        return filepath


class ReportGenerator:
    """Generates comprehensive reports from evaluation results."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = config['reporting']['output_path']
        self.latex_generator = LaTeXTableGenerator(config)
        self.figure_generator = FigureGenerator(config)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_comprehensive_report(self, all_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate comprehensive report with all tables and figures."""
        logger.info("Generating comprehensive report...")
        
        generated_files = {}
        
        # Generate LaTeX tables
        if 'performance_results' in all_results:
            latex_table = self.latex_generator.generate_performance_table(
                all_results['performance_results']
            )
            generated_files['performance_table'] = latex_table
        
        if 'ablation_results' in all_results:
            latex_table = self.latex_generator.generate_ablation_table(
                all_results['ablation_results']
            )
            generated_files['ablation_table'] = latex_table
        
        if 'sector_results' in all_results:
            latex_table = self.latex_generator.generate_sector_table(
                all_results['sector_results']
            )
            generated_files['sector_table'] = latex_table
        
        if 'optimization_results' in all_results:
            latex_table = self.latex_generator.generate_hyperparameter_table(
                all_results['optimization_results']
            )
            generated_files['hyperparameter_table'] = latex_table
        
        if 'significance_results' in all_results:
            latex_table = self.latex_generator.generate_significance_table(
                all_results['significance_results']
            )
            generated_files['significance_table'] = latex_table
        
        # Generate figures
        if 'performance_results' in all_results:
            figure_path = self.figure_generator.create_performance_comparison(
                all_results['performance_results']
            )
            generated_files['performance_figure'] = figure_path
        
        if 'training_history' in all_results:
            figure_path = self.figure_generator.create_training_curves(
                all_results['training_history']
            )
            generated_files['training_curves_figure'] = figure_path
        
        if 'confusion_matrices' in all_results:
            figure_path = self.figure_generator.create_confusion_matrices(
                all_results['confusion_matrices']
            )
            generated_files['confusion_matrices_figure'] = figure_path
        
        if 'ablation_results' in all_results and 'contributions' in all_results['ablation_results']:
            figure_path = self.figure_generator.create_ablation_waterfall(
                all_results['ablation_results']['contributions']
            )
            generated_files['ablation_figure'] = figure_path
        
        if 'sector_results' in all_results:
            figure_path = self.figure_generator.create_sector_performance(
                all_results['sector_results']
            )
            generated_files['sector_figure'] = figure_path
        
        # Generate summary report
        summary_report = self._generate_summary_report(all_results)
        summary_path = os.path.join(self.output_dir, 'executive_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary_report)
        generated_files['summary_report'] = summary_path
        
        # Generate JSON report
        json_report = self._generate_json_report(all_results)
        json_path = os.path.join(self.output_dir, 'comprehensive_results.json')
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        generated_files['json_report'] = json_path
        
        # Generate CSV summaries
        csv_files = self._generate_csv_summaries(all_results)
        generated_files.update(csv_files)
        
        logger.info(f"Generated comprehensive report with {len(generated_files)} files")
        return generated_files
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate executive summary report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        summary = f"""
STOCK PREDICTION SYSTEM - EXECUTIVE SUMMARY
Generated: {timestamp}
{'='*60}

OVERVIEW:
This report summarizes the performance of the integrated stock prediction system
that combines FinBERT embeddings with comprehensive technical analysis and 
advanced deep learning architectures.

"""
        
        # Performance summary
        if 'performance_results' in results:
            performance = results['performance_results']
            summary += "PERFORMANCE RESULTS:\n"
            summary += "-" * 20 + "\n"
            
            for model_name, metrics in performance.items():
                summary += f"\n{model_name}:\n"
                for metric, value in metrics.items():
                    summary += f"  {metric.replace('_', ' ').title()}: {value:.4f}\n"
        
        # Ablation study summary
        if 'ablation_results' in results and 'contributions' in results['ablation_results']:
            contributions = results['ablation_results']['contributions']
            summary += f"\nABLATION STUDY:\n"
            summary += "-" * 15 + "\n"
            summary += "Component contributions to model performance:\n"
            
            for component, contribution in sorted(contributions.items(), 
                                                key=lambda x: x[1], reverse=True):
                summary += f"  {component.replace('_', ' ').title()}: {contribution:.4f}\n"
        
        # Sector analysis summary
        if 'sector_results' in results and 'cross_sector_stats' in results['sector_results']:
            stats = results['sector_results']['cross_sector_stats']
            summary += f"\nSECTOR ANALYSIS:\n"
            summary += "-" * 15 + "\n"
            summary += f"Mean Accuracy: {stats.get('mean_accuracy', 0):.4f} ± {stats.get('std_accuracy', 0):.4f}\n"
            summary += f"Mean F1-Score: {stats.get('mean_f1', 0):.4f} ± {stats.get('std_f1', 0):.4f}\n"
            summary += f"Coefficient of Variation: {stats.get('cv_accuracy', 0):.4f}\n"
        
        # Optimization summary
        if 'optimization_results' in results:
            opt_results = results['optimization_results']
            summary += f"\nHYPERPARAMETER OPTIMIZATION:\n"
            summary += "-" * 30 + "\n"
            summary += f"Best Accuracy: {opt_results.get('best_value', 0):.4f}\n"
            summary += f"Optimization Trials: {opt_results.get('n_trials', 0)}\n"
            
            if 'best_params' in opt_results:
                summary += "Optimal Parameters:\n"
                for param, value in opt_results['best_params'].items():
                    summary += f"  {param}: {value}\n"
        
        summary += f"\n{'='*60}\n"
        summary += "End of Summary Report\n"
        
        return summary
    
    def _generate_json_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive JSON report."""
        json_report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'model_type': 'Hybrid BiLSTM + Attention',
                'features': ['FinBERT embeddings', 'Technical indicators', 'Market context'],
                'evaluation_methods': ['Walk-Forward CV', 'Backtesting', 'Statistical tests']
            },
            'results': results
        }
        
        return json_report
    
    def _generate_csv_summaries(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate CSV summary files."""
        csv_files = {}
        
        # Performance results CSV
        if 'performance_results' in results:
            df = pd.DataFrame(results['performance_results']).T
            csv_path = os.path.join(self.output_dir, 'performance_results.csv')
            df.to_csv(csv_path)
            csv_files['performance_csv'] = csv_path
        
        # Sector results CSV
        if 'sector_results' in results and 'sector_results' in results['sector_results']:
            sector_data = []
            for sector, metrics in results['sector_results']['sector_results'].items():
                sector_data.append({'Sector': sector, **metrics})
            
            df = pd.DataFrame(sector_data)
            csv_path = os.path.join(self.output_dir, 'sector_results.csv')
            df.to_csv(csv_path, index=False)
            csv_files['sector_csv'] = csv_path
        
        return csv_files


def main():
    """Test the reporting functionality."""
    import yaml
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    test_results = {
        'performance_results': {
            'Hybrid Model': {'accuracy': 0.8234, 'f1_score': 0.8156, 'precision': 0.8203, 'recall': 0.8234},
            'BiLSTM': {'accuracy': 0.7891, 'f1_score': 0.7834, 'precision': 0.7856, 'recall': 0.7891},
            'Random Forest': {'accuracy': 0.7456, 'f1_score': 0.7398, 'precision': 0.7423, 'recall': 0.7456}
        },
        'ablation_results': {
            'contributions': {
                'finbert_embeddings': 0.0543,
                'attention_mechanism': 0.0234,
                'temporal_weights': 0.0178,
                'market_context': 0.0123,
                'technical_indicators': 0.0089
            }
        },
        'training_history': {
            'train_loss': [0.8, 0.6, 0.4, 0.3, 0.25],
            'val_loss': [0.75, 0.65, 0.45, 0.35, 0.30],
            'train_accuracy': [0.65, 0.72, 0.78, 0.81, 0.83],
            'val_accuracy': [0.63, 0.70, 0.76, 0.79, 0.82],
            'learning_rates': [0.001, 0.0008, 0.0006, 0.0004, 0.0002]
        }
    }
    
    # Test report generation
    report_generator = ReportGenerator(config)
    generated_files = report_generator.generate_comprehensive_report(test_results)
    
    print(f"Generated {len(generated_files)} report files:")
    for file_type, filepath in generated_files.items():
        print(f"  {file_type}: {filepath}")


if __name__ == "__main__":
    main()