"""
Visualization utilities for speaker verification evaluation.

Creates publication-quality plots for thesis:
- Score matrix heatmap
- Score distributions (genuine vs impostor)
- ROC curve
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_score_matrix(score_matrix, speaker_labels, output_path="score_matrix.png"):
    """
    Plot score matrix as a heatmap.

    Visualizes DTW distances between all templates and test utterances.
    Diagonal and near-diagonal regions should show lower distances (same speaker),
    while off-diagonal regions show higher distances (different speakers).

    Args:
        score_matrix: 2D numpy array, shape (n_templates, n_test_utterances)
        speaker_labels: list of speaker IDs for rows
        output_path: path to save PNG file
    """
    plt.figure(figsize=(16, 10))
    
    # Create heatmap
    sns.heatmap(
        score_matrix,
        cmap='RdYlGn_r',  # Red=high distance, Green=low distance
        cbar_kws={'label': 'Normalized DTW Distance'},
        xticklabels=False,
        yticklabels=speaker_labels,
        vmin=np.percentile(score_matrix, 5),
        vmax=np.percentile(score_matrix, 95)
    )
    
    plt.title('Speaker Verification Score Matrix\n(Templates vs Test Utterances)', fontsize=14, fontweight='bold')
    plt.xlabel('Test Utterance Index', fontsize=12)
    plt.ylabel('Speaker Template', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Score matrix visualization saved to {output_path}")


def plot_score_histograms(genuine_scores, impostor_scores, output_path="score_histograms.png"):
    """
    Plot histograms of genuine and impostor score distributions.

    Visualizes how well the system separates genuine (same-speaker) from
    impostor (different-speaker) trials.

    Args:
        genuine_scores: list/array of distances for same-speaker pairs
        impostor_scores: list/array of distances for different-speaker pairs
        output_path: path to save PNG file
    """
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    bins = np.linspace(
        min(np.min(genuine_scores), np.min(impostor_scores)) - 5,
        max(np.max(genuine_scores), np.max(impostor_scores)) + 5,
        30
    )
    
    plt.hist(genuine_scores, bins=bins, alpha=0.6, label=f'Genuine (n={len(genuine_scores)})', color='blue', edgecolor='black')
    plt.hist(impostor_scores, bins=bins, alpha=0.6, label=f'Impostor (n={len(impostor_scores)})', color='red', edgecolor='black')
    
    plt.xlabel('Normalized DTW Distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Score Distributions: Genuine vs Impostor Trials', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\nScore Distribution Statistics:")
    print(f"  Genuine:  mean={np.mean(genuine_scores):.2f}, std={np.std(genuine_scores):.2f}, "
          f"min={np.min(genuine_scores):.2f}, max={np.max(genuine_scores):.2f}")
    print(f"  Impostor: mean={np.mean(impostor_scores):.2f}, std={np.std(impostor_scores):.2f}, "
          f"min={np.min(impostor_scores):.2f}, max={np.max(impostor_scores):.2f}")
    print(f"Score histograms saved to {output_path}")


def plot_roc_curve(far_array, frr_array, eer_metrics=None, output_path="roc_curve.png"):
    """
    Plot ROC (Receiver Operating Characteristic) curve.

    Shows the trade-off between False Acceptance Rate (FAR) and
    False Rejection Rate (FRR) across thresholds.

    Args:
        far_array: array of FAR values
        frr_array: array of FRR values
        eer_metrics: optional dict with 'far_at_eer', 'frr_at_eer' keys
        output_path: path to save PNG file
    """
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(far_array, frr_array, 'b-', linewidth=2.5, label='ROC Curve')
    
    # Plot diagonal (random system)
    plt.plot([0, 1], [1, 0], 'k--', linewidth=1, alpha=0.5, label='Random System')
    
    # Mark EER point if provided
    if eer_metrics:
        plt.plot(
            eer_metrics['far_at_eer'],
            eer_metrics['frr_at_eer'],
            'ro',
            markersize=10,
            label=f"EER = {eer_metrics['far_at_eer']*100:.2f}% "
                  f"(τ={eer_metrics['optimal_threshold']:.2f})"
        )
    
    plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
    plt.ylabel('False Rejection Rate (FRR)', fontsize=12)
    plt.title('ROC Curve: Speaker Verification Performance', fontsize=14, fontweight='bold')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to {output_path}")


def plot_threshold_analysis(thresholds, far_array, frr_array, eer_metrics, output_path="threshold_analysis.png"):
    """
    Plot FAR and FRR as functions of threshold.

    Shows how error rates change with decision threshold,
    highlighting the EER point.

    Args:
        thresholds: array of threshold values
        far_array: array of FAR values
        frr_array: array of FRR values
        eer_metrics: dict with EER metrics
        output_path: path to save PNG file
    """
    plt.figure(figsize=(12, 6))
    
    # Plot FAR and FRR
    plt.plot(thresholds, far_array, 'r-', linewidth=2, label='FAR (False Acceptance Rate)')
    plt.plot(thresholds, frr_array, 'b-', linewidth=2, label='FRR (False Rejection Rate)')
    
    # Mark EER point
    plt.axvline(
        eer_metrics['optimal_threshold'],
        color='green',
        linestyle='--',
        linewidth=2,
        label=f"EER Point (τ={eer_metrics['optimal_threshold']:.2f})"
    )
    
    plt.xlabel('Decision Threshold (τ)', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.title('Error Rates vs Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Threshold analysis plot saved to {output_path}")
