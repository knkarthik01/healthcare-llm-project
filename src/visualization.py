"""
Module for visualizing evaluation results from LLM diabetes risk assessments.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

def plot_accuracy_comparison(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot accuracy comparison across methods.
    
    Args:
        results: Dictionary mapping method names to evaluation results
        save_path: Optional path to save the figure
    """
    methods = list(results.keys())
    accuracies = [results[method]['accuracy'] for method in methods]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(methods, accuracies, color='skyblue')
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.ylabel('Accuracy')
    plt.title('Diabetes Risk Prediction Accuracy by Method')
    plt.ylim(0, 1.1)  # Leave room for labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_reasoning_depth(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot reasoning depth comparison across methods.
    
    Args:
        results: Dictionary mapping method names to evaluation results
        save_path: Optional path to save the figure
    """
    methods = list(results.keys())
    reasoning_depths = [results[method]['avg_reasoning_depth'] for method in methods]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(methods, reasoning_depths, color='lightgreen')
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.ylabel('Average Reasoning Depth (1-5)')
    plt.title('Reasoning Depth by Prompting Method')
    plt.ylim(0, 5.5)  # Scale is 1-5
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_factors_mentioned(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot average number of factors mentioned across methods.
    
    Args:
        results: Dictionary mapping method names to evaluation results
        save_path: Optional path to save the figure
    """
    methods = list(results.keys())
    factors = [results[method]['avg_factors_mentioned'] for method in methods]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(methods, factors, color='salmon')
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.ylabel('Average Number of Factors Mentioned')
    plt.title('Diabetes Risk Factors Mentioned by Method')
    plt.ylim(0, max(factors) * 1.2)  # Adjust based on data
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, classes: List[str], method_name: str, save_path: Optional[str] = None):
    """
    Plot confusion matrix for a specific method.
    
    Args:
        cm: Confusion matrix array
        classes: Class labels
        method_name: Name of the method
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Risk')
    plt.ylabel('True Risk')
    plt.title(f'Confusion Matrix - {method_name}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_all_confusion_matrices(results: Dict[str, Dict[str, Any]], save_dir: Optional[str] = None):
    """
    Plot confusion matrices for all methods.
    
    Args:
        results: Dictionary mapping method names to evaluation results
        save_dir: Optional directory to save figures
    """
    for method_name, result in results.items():
        if result.get('confusion_matrix') is not None:
            save_path = None
            if save_dir:
                save_path = f"{save_dir}/confusion_matrix_{method_name.replace(' ', '_')}.png"
            
            plot_confusion_matrix(
                result['confusion_matrix'],
                result['classes'],
                method_name,
                save_path
            )

def plot_comparative_metrics(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot a comparative bar chart of key metrics across methods.
    
    Args:
        results: Dictionary mapping method names to evaluation results
        save_path: Optional path to save the figure
    """
    methods = list(results.keys())
    metrics = ['accuracy', 'avg_reasoning_depth', 'avg_factors_mentioned']
    metric_labels = ['Accuracy', 'Reasoning Depth', 'Factors Mentioned']
    
    # Normalize metrics to 0-1 for comparison
    normalized_data = []
    for metric in metrics:
        values = [results[method].get(metric, 0) for method in methods]
        
        # Normalize differently depending on the metric
        if metric == 'accuracy':
            # Already 0-1
            normalized = values 
        elif metric == 'avg_reasoning_depth':
            # Normalize from 1-5 scale to 0-1
            normalized = [(v - 1) / 4 for v in values]
        else:
            # Normalize based on max value
            max_val = max(values)
            normalized = [v / max_val if max_val > 0 else 0 for v in values]
        
        normalized_data.append(normalized)
    
    # Create DataFrame for plotting
    df = pd.DataFrame(normalized_data, index=metric_labels, columns=methods).T
    
    # Plot
    plt.figure(figsize=(12, 6))
    df.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.ylabel('Normalized Score (0-1)')
    plt.title('Comparative Performance Metrics Across Methods')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def create_comparison_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a comparison table of metrics across methods.
    
    Args:
        results: Dictionary mapping method names to evaluation results
        
    Returns:
        DataFrame containing comparison metrics
    """
    methods = list(results.keys())
    metrics = {
        'Accuracy': [results[method].get('accuracy', 0) for method in methods],
        'Reasoning Depth (1-5)': [results[method].get('avg_reasoning_depth', 0) for method in methods],
        'Factors Mentioned': [results[method].get('avg_factors_mentioned', 0) for method in methods],
        'Unknown Rate': [results[method].get('unknown_rate', 0) for method in methods],
        'Sample Size': [results[method].get('num_evaluations', 0) for method in methods],
    }
    
    # Add precision, recall, F1 by class if available
    classes = results[methods[0]].get('classes', ['Low', 'Medium', 'High'])
    for i, cls in enumerate(classes):
        if results[methods[0]].get('precision') is not None:
            metrics[f'Precision ({cls})'] = [
                results[method].get('precision')[i] if results[method].get('precision') is not None else None
                for method in methods
            ]
            metrics[f'Recall ({cls})'] = [
                results[method].get('recall')[i] if results[method].get('recall') is not None else None
                for method in methods
            ]
            metrics[f'F1 ({cls})'] = [
                results[method].get('f1')[i] if results[method].get('f1') is not None else None
                for method in methods
            ]
    
    df = pd.DataFrame(metrics, index=methods)
    return df

def plot_all_metrics(results: Dict[str, Dict[str, Any]], save_dir: Optional[str] = None):
    """
    Generate and save all evaluation plots.
    
    Args:
        results: Dictionary mapping method names to evaluation results
        save_dir: Optional directory to save figures
    """
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    # Generate individual plots
    plot_accuracy_comparison(results, f"{save_dir}/accuracy_comparison.png" if save_dir else None)
    plot_reasoning_depth(results, f"{save_dir}/reasoning_depth.png" if save_dir else None)
    plot_factors_mentioned(results, f"{save_dir}/factors_mentioned.png" if save_dir else None)
    plot_comparative_metrics(results, f"{save_dir}/comparative_metrics.png" if save_dir else None)
    plot_all_confusion_matrices(results, save_dir)
