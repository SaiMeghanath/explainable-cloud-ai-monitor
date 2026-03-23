"""app/visualize.py

Visualization utilities for LIME and SHAP explanations.

All plot functions accept an optional save_path:
  - If provided, the figure is saved to disk
  - If None, plt.show() is called for interactive display
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap


def plot_lime(explanation, predicted_label: str, save_path: str = None):
    """Bar chart of LIME feature importances for a single prediction.
    
    Positive values (green) push toward predicted class.
    Negative values (red) push away from predicted class.
    
    Args:
        explanation: LimeTabularExplainer explanation object
        predicted_label: Predicted class label string
        save_path: File path to save figure (e.g., 'lime.png'), or None
    """
    label_idx = explanation.available_labels()[0]
    features = explanation.as_list(label=label_idx)
    features_sorted = sorted(features, key=lambda x: x[1])
    
    names = [f[0] for f in features_sorted]
    values = [f[1] for f in features_sorted]
    colors = ['#ef4444' if v < 0 else '#22c55e' for v in values]
    
    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.5 + 1)))
    bars = ax.barh(names, values, color=colors, edgecolor=None, height=0.6)
    
    ax.axvline(0, color='#374151', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Feature importance (LIME)', fontsize=11)
    ax.set_title(f'LIME Explanation - Predicted: {predicted_label}', fontsize=13, fontweight='bold')
    
    pos_patch = mpatches.Patch(color='#22c55e', label='Supports prediction')
    neg_patch = mpatches.Patch(color='#ef4444', label='Opposes prediction')
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9, loc='lower right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_shap_summary(shap_values, X: np.ndarray, feature_names: list, 
                      class_idx: int = 0, save_path: str = None):
    """SHAP summary plot - global feature importance across the dataset.
    
    Shows distribution of SHAP values for each feature across all samples.
    Each dot is one sample; color indicates feature value (blue=low, red=high).
    
    Args:
        shap_values: SHAP values from SHAPExplainer.compute()
        X: Input data (scaled), shape (n_samples, n_features)
        feature_names: List of feature name strings
        class_idx: For multiclass, which class to plot (default: 0)
        save_path: File path to save figure, or None
    """
    values = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
    
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.summary_plot(values, X, feature_names=feature_names, show=False, plot_size=None)
    
    plt.title(f'SHAP Summary Plot - Class {class_idx}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_shap_force(shap_explainer, instance: np.ndarray, scaler,
                    class_idx: int = 0, save_path: str = None):
    """SHAP force plot - local explanation for a single prediction.
    
    Shows how each feature pushes the prediction above or below the base value
    (expected model output).
    
    Args:
        shap_explainer: SHAPExplainer instance
        instance: Single input, shape (1, n_features), unscaled
        scaler: Fitted scaler
        class_idx: For multiclass, which class to explain
        save_path: File path to save figure, or None
    """
    instance_scaled = scaler.transform(instance)
    shap_vals = shap_explainer.compute_single(instance_scaled)
    
    # For multiclass, shap_values is a list of arrays (one per class)
    vals = shap_vals[class_idx] if isinstance(shap_vals, list) else shap_vals
    base = shap_explainer.explainer.expected_value
    base_val = base[class_idx] if isinstance(base, (list, np.ndarray)) else base
    
    shap.initjs()
    force_plot = shap.force_plot(
        base_val, vals[0], instance_scaled[0],
        feature_names=shap_explainer.feature_names,
        matplotlib=True,
        show=False,
    )
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_feature_importance_comparison(lime_features: list, shap_values,
                                       feature_names: list, class_idx: int = 0,
                                       save_path: str = None):
    """Side-by-side comparison of LIME vs SHAP global feature importances.
    
    Args:
        lime_features: List of (feature_name, importance) from LIME
        shap_values: SHAP values array
        feature_names: List of feature name strings
        class_idx: Class index for multiclass SHAP
        save_path: File path to save figure, or None
    """
    # Get SHAP values for the specified class
    vals = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
    shap_importance = np.abs(vals).mean(axis=0)
    shap_dict = dict(zip(feature_names, shap_importance))
    
    # Get LIME importances (absolute values)
    lime_dict = {f: abs(v) for f, v in lime_features}
    
    # Align on shared feature names
    features = list(shap_dict.keys())
    shap_vals_plot = [shap_dict.get(f, 0) for f in features]
    lime_vals_plot = [lime_dict.get(f, 0) for f in features]
    
    x = np.arange(len(features))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, shap_vals_plot, width, label='SHAP', color='#6366f1', alpha=0.85)
    ax.bar(x + width/2, lime_vals_plot, width, label='LIME', color='#f59e0b', alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Mean importance', fontsize=11)
    ax.set_title('Feature Importance: LIME vs SHAP', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()
