#!/usr/bin/env python3
"""
Analyze Validation Results from Classifier Training

This script reads validation results from train_cell_classifier_head.py
and generates comprehensive diagnostic visualizations including:
- Confusion matrices (raw and normalized)
- Class prediction probability distributions
- Per-class performance metrics
- Confidence analysis
- Misclassification analysis

Usage:
    python analyze_validation_results.py --val_results_dir ./log_local/run_date/val_results/
    
Author: CellViT++ Team
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class ValidationAnalyzer:
    """Analyze and visualize validation results from classifier training"""
    
    def __init__(self, val_results_dir: Path, output_dir: Path = None):
        """
        Initialize analyzer
        
        Args:
            val_results_dir: Path to val_results directory
            output_dir: Path to save analysis outputs (default: val_results_dir/analysis)
        """
        self.val_results_dir = Path(val_results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.val_results_dir / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load data
        self.data = self._load_validation_data()
        self.class_names = self._get_class_names()
        self.num_classes = len(self.class_names)
        
        self.logger.info(f"Loaded {len(self.data)} validation samples")
        self.logger.info(f"Classes: {self.class_names}")
    
    def _load_validation_data(self) -> List[Dict]:
        """Load validation results from JSON files"""
        data = []
        json_files = sorted(self.val_results_dir.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(
                f"No JSON files found in {self.val_results_dir}"
            )
        
        self.logger.info(f"Found {len(json_files)} validation result files")
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                result = json.load(f)
                data.append(result)
        
        return data
    
    def _get_class_names(self) -> List[str]:
        """Extract class names from data"""
        # Try to get from first sample
        if self.data:
            first_sample = self.data[0]
            if 'class_names' in first_sample:
                return first_sample['class_names']
            elif 'predictions' in first_sample and isinstance(first_sample['predictions'], dict):
                return list(first_sample['predictions'].keys())
        
        # Fallback: infer from data
        max_label = max(sample['true_label'] for sample in self.data)
        return [f"Class_{i}" for i in range(max_label + 1)]
    
    def extract_labels_and_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract true labels, predicted labels, and prediction probabilities
        
        Returns:
            y_true: True class labels
            y_pred: Predicted class labels
            y_proba: Prediction probabilities (N x C)
        """
        y_true = []
        y_pred = []
        y_proba = []
        
        for sample in self.data:
            y_true.append(sample['true_label'])
            y_pred.append(sample['predicted_label'])
            
            # Handle probability format
            if 'probabilities' in sample:
                probs = sample['probabilities']
            elif 'predictions' in sample:
                probs = list(sample['predictions'].values())
            else:
                # Fallback: one-hot encoded prediction
                probs = [0.0] * self.num_classes
                probs[sample['predicted_label']] = 1.0
            
            y_proba.append(probs)
        
        return (
            np.array(y_true),
            np.array(y_pred),
            np.array(y_proba)
        )
    
    def plot_confusion_matrix(self, normalize: bool = False):
        """Plot confusion matrix"""
        y_true, y_pred, _ = self.extract_labels_and_predictions()
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
            filename = 'confusion_matrix_normalized.png'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
            filename = 'confusion_matrix.png'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved {filename} to {save_path}")
        plt.close()
    
    def plot_probability_distributions(self):
        """Plot prediction probability distributions per class"""
        y_true, y_pred, y_proba = self.extract_labels_and_predictions()
        
        # Create figure with subplots
        n_classes = len(self.class_names)
        fig, axes = plt.subplots(
            n_classes, 1,
            figsize=(12, 4 * n_classes),
            sharex=True
        )
        
        if n_classes == 1:
            axes = [axes]
        
        for class_idx, class_name in enumerate(self.class_names):
            ax = axes[class_idx]
            
            # Get probabilities for this class
            class_probs = y_proba[:, class_idx]
            
            # Separate by true label
            true_positive_probs = class_probs[y_true == class_idx]
            false_positive_probs = class_probs[y_true != class_idx]
            
            # Plot histograms
            ax.hist(
                true_positive_probs,
                bins=50,
                alpha=0.7,
                label=f'True {class_name}',
                color='green',
                edgecolor='black'
            )
            ax.hist(
                false_positive_probs,
                bins=50,
                alpha=0.7,
                label=f'Other Classes',
                color='red',
                edgecolor='black'
            )
            
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f'Prediction Probability Distribution - {class_name}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Prediction Probability', fontsize=12)
        plt.tight_layout()
        
        save_path = self.output_dir / 'probability_distributions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved probability distributions to {save_path}")
        plt.close()
    
    def plot_confidence_analysis(self):
        """Analyze prediction confidence"""
        y_true, y_pred, y_proba = self.extract_labels_and_predictions()
        
        # Get max probability (confidence) for each prediction
        confidences = np.max(y_proba, axis=1)
        correct = (y_true == y_pred)
        
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. Confidence distribution for correct vs incorrect predictions
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(
            confidences[correct],
            bins=50,
            alpha=0.7,
            label='Correct',
            color='green',
            edgecolor='black'
        )
        ax1.hist(
            confidences[~correct],
            bins=50,
            alpha=0.7,
            label='Incorrect',
            color='red',
            edgecolor='black'
        )
        ax1.set_xlabel('Prediction Confidence', fontsize=10)
        ax1.set_ylabel('Count', fontsize=10)
        ax1.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy vs confidence
        ax2 = fig.add_subplot(gs[0, 1])
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        accuracies = []
        counts = []
        
        for i in range(len(bins) - 1):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if mask.sum() > 0:
                accuracies.append(correct[mask].mean())
                counts.append(mask.sum())
            else:
                accuracies.append(0)
                counts.append(0)
        
        ax2.plot(bin_centers, accuracies, 'bo-', linewidth=2, markersize=8)
        ax2.axhline(correct.mean(), color='r', linestyle='--', label=f'Overall Acc: {correct.mean():.3f}')
        ax2.set_xlabel('Confidence Bin', fontsize=10)
        ax2.set_ylabel('Accuracy', fontsize=10)
        ax2.set_title('Accuracy vs Confidence', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Per-class confidence
        ax3 = fig.add_subplot(gs[1, :])
        class_confidences = []
        for class_idx in range(len(self.class_names)):
            class_mask = (y_pred == class_idx)
            if class_mask.sum() > 0:
                class_confidences.append(confidences[class_mask])
            else:
                class_confidences.append([])
        
        bp = ax3.boxplot(
            class_confidences,
            labels=self.class_names,
            patch_artist=True,
            showmeans=True
        )
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax3.set_xlabel('Predicted Class', fontsize=10)
        ax3.set_ylabel('Prediction Confidence', fontsize=10)
        ax3.set_title('Confidence Distribution per Predicted Class', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'confidence_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved confidence analysis to {save_path}")
        plt.close()
    
    def plot_per_class_metrics(self):
        """Plot detailed per-class metrics"""
        y_true, y_pred, _ = self.extract_labels_and_predictions()
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(self.class_names))
        )
        
        # Create DataFrame for visualization
        df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['Precision', 'Recall', 'F1-Score']
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        for ax, metric, color in zip(axes, metrics, colors):
            bars = ax.bar(
                df['Class'],
                df[metric],
                color=color,
                alpha=0.7,
                edgecolor='black'
            )
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'{metric} per Class', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis='y')
            plt.sca(ax)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'per_class_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved per-class metrics to {save_path}")
        plt.close()
        
        # Save as CSV
        csv_path = self.output_dir / 'per_class_metrics.csv'
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved metrics CSV to {csv_path}")
    
    def plot_misclassification_matrix(self):
        """Visualize which classes are confused with each other"""
        y_true, y_pred, _ = self.extract_labels_and_predictions()
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Create misclassification matrix (only off-diagonal elements)
        misclass_matrix = cm.copy()
        np.fill_diagonal(misclass_matrix, 0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            misclass_matrix,
            annot=True,
            fmt='d',
            cmap='Reds',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Misclassification Count'}
        )
        plt.title('Misclassification Matrix (Off-Diagonal)', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path = self.output_dir / 'misclassification_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved misclassification matrix to {save_path}")
        plt.close()
    
    def generate_classification_report(self):
        """Generate and save detailed classification report"""
        y_true, y_pred, _ = self.extract_labels_and_predictions()
        
        # Generate report
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            digits=4
        )
        
        # Save to file
        report_path = self.output_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)
            f.write("\n\n")
            f.write("Overall Accuracy: {:.4f}\n".format((y_true == y_pred).mean()))
            f.write("Total Samples: {}\n".format(len(y_true)))
        
        self.logger.info(f"Saved classification report to {report_path}")
        
        # Also print to console
        print("\n" + "=" * 80)
        print("CLASSIFICATION REPORT")
        print("=" * 80)
        print(report)
        print(f"Overall Accuracy: {(y_true == y_pred).mean():.4f}")
        print(f"Total Samples: {len(y_true)}")
        print("=" * 80 + "\n")
    
    def plot_roc_curves(self):
        """Plot ROC curves for multi-class classification"""
        y_true, _, y_proba = self.extract_labels_and_predictions()
        
        # One-hot encode true labels
        n_classes = len(self.class_names)
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            if y_true_onehot[:, i].sum() > 0:  # Only if class has samples
                fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_proba[:, i])
                auc = roc_auc_score(y_true_onehot[:, i], y_proba[:, i])
                
                ax.plot(
                    fpr,
                    tpr,
                    linewidth=2,
                    label=f'{class_name} (AUC = {auc:.3f})'
                )
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Multi-Class', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / 'roc_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved ROC curves to {save_path}")
        plt.close()
    
    def generate_summary_statistics(self):
        """Generate summary statistics"""
        y_true, y_pred, y_proba = self.extract_labels_and_predictions()
        
        confidences = np.max(y_proba, axis=1)
        correct = (y_true == y_pred)
        
        summary = {
            "Total Samples": len(y_true),
            "Overall Accuracy": float((y_true == y_pred).mean()),
            "Number of Classes": len(self.class_names),
            "Class Names": self.class_names,
            "Mean Confidence": float(confidences.mean()),
            "Mean Confidence (Correct)": float(confidences[correct].mean()),
            "Mean Confidence (Incorrect)": float(confidences[~correct].mean()) if (~correct).sum() > 0 else 0.0,
            "Samples per Class": {
                class_name: int((y_true == i).sum())
                for i, class_name in enumerate(self.class_names)
            }
        }
        
        # Save as JSON
        summary_path = self.output_dir / 'summary_statistics.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, indent=2, fp=f)
        
        self.logger.info(f"Saved summary statistics to {summary_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            elif isinstance(value, list):
                print(f"{key}: {', '.join(value)}")
            else:
                print(f"{key}: {value}")
        print("=" * 80 + "\n")
    
    def run_all_analyses(self):
        """Run all analysis functions"""
        self.logger.info("Starting comprehensive validation analysis...")
        
        # Generate all visualizations and reports
        self.generate_summary_statistics()
        self.generate_classification_report()
        self.plot_confusion_matrix(normalize=False)
        self.plot_confusion_matrix(normalize=True)
        self.plot_probability_distributions()
        self.plot_confidence_analysis()
        self.plot_per_class_metrics()
        self.plot_misclassification_matrix()
        
        try:
            self.plot_roc_curves()
        except Exception as e:
            self.logger.warning(f"Could not generate ROC curves: {e}")
        
        self.logger.info(f"Analysis complete! All outputs saved to: {self.output_dir}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze validation results from classifier training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--val_results_dir',
        type=str,
        required=True,
        help='Path to validation results directory (e.g., log_local/run_date/val_results/)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Path to save analysis outputs (default: val_results_dir/analysis)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Initialize analyzer
    analyzer = ValidationAnalyzer(
        val_results_dir=args.val_results_dir,
        output_dir=args.output_dir
    )
    
    # Run all analyses
    analyzer.run_all_analyses()
    
    print("\n✅ Analysis complete!")
    print(f"📁 Results saved to: {analyzer.output_dir}")


if __name__ == '__main__':
    main()
