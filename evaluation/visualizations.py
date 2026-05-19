"""
Visualization utilities for speaker verification evaluation.

Creates publication-quality plots for thesis:
- Score matrix heatmap
- Score distributions (genuine vs impostor)
- ROC curve
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import librosa
import librosa.display
from scipy.stats import norm


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
        cbar_kws={'label': 'DTW Distance'},
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


def plot_speaker_distance_matrix(distance_matrix, speaker_labels, output_path="speaker_distance_matrix.png", figsize=(14, 12)):
    """
    Plot a speaker distance matrix heatmap for average DTW distances.

    Args:
        distance_matrix: 2D numpy array of speaker-to-speaker distances
        speaker_labels: list of speaker IDs for rows and columns
        output_path: path to save PNG file
        figsize: size of the output figure
    """
    # Compute vmin/vmax from valid (non-NaN) values only
    valid_values = distance_matrix[np.isfinite(distance_matrix)]
    if len(valid_values) > 0:
        vmin = np.min(valid_values)
        vmax = np.max(valid_values)
    else:
        vmin, vmax = 0, 1
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        distance_matrix,
        xticklabels=speaker_labels,
        yticklabels=speaker_labels,
        cmap='viridis',
        annot=False,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Average DTW Distance'},
        linewidths=0.3,
        linecolor='gray',
        cbar=True
    )
    plt.title('Speaker Distance Matrix (Average DTW)', fontsize=16, fontweight='bold')
    plt.xlabel('Test Speaker', fontsize=13)
    plt.ylabel('Template Speaker', fontsize=13)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Speaker distance matrix saved to {output_path}")


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
    
    plt.xlabel('DTW Distance', fontsize=12)
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


def plot_confusion_matrix(cm_dict, output_path="confusion_matrix.png"):
    """
    Plot a 2x2 confusion matrix heatmap for speaker verification.
    """
    # Orientation: rows = predicted, columns = actual
    matrix = np.array([[cm_dict['TP'], cm_dict['FP']], [cm_dict['FN'], cm_dict['TN']]])
    total = matrix.sum() if matrix.sum() > 0 else 1

    annot = np.empty(matrix.shape, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = int(matrix[i, j])
            percent = 100.0 * count / total
            annot[i, j] = f"{count}\n{percent:.1f}%"

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(matrix, annot=annot, fmt='', cmap='Blues', cbar=False,
                     xticklabels=['Actually Positive (1)', 'Actually Negative (0)'],
                     yticklabels=['Predicted Positive (1)', 'Predicted Negative (0)'],
                     linewidths=0.8, linecolor='black', annot_kws={'fontsize':11, 'weight':'bold'})
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Label', fontsize=12)
    plt.ylabel('Predicted Label', fontsize=12)

    metrics_text = (
        f"Accuracy: {cm_dict['accuracy']*100:.2f}%   "
        f"Sensitivity (TAR): {cm_dict['TAR']*100:.2f}%   "
        f"Specificity (TRR): {cm_dict['TRR']*100:.2f}%   "
        f"F1: {cm_dict['f1_score']:.3f}"
    )
    plt.gcf().text(0.5, -0.08, metrics_text, ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to {output_path}")


def plot_confusion_matrix_report(cm_dict, output_path="confusion_matrix_report.png"):
    """
    Plot a report-ready confusion matrix with larger annotations.
    """
    matrix = np.array([[cm_dict['TP'], cm_dict['FP']], [cm_dict['FN'], cm_dict['TN']]])
    total = matrix.sum() if matrix.sum() > 0 else 1
    annot = np.empty(matrix.shape, dtype=object)
    labels = [['TP', 'FP'], ['FN', 'TN']]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = int(matrix[i, j])
            percent = 100.0 * count / total
            annot[i, j] = f"{labels[i][j]}\n{count}\n{percent:.1f}%"

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        matrix,
        annot=annot,
        fmt='',
        cmap='Blues',
        cbar=False,
        annot_kws={'size': 16, 'weight': 'bold'},
        xticklabels=['Actually Positive (1)', 'Actually Negative (0)'],
        yticklabels=['Predicted Positive (1)', 'Predicted Negative (0)'],
        linewidths=1.0,
        linecolor='black'
    )
    ax.set_title('Confusion Matrix', fontsize=18, fontweight='bold')
    plt.xlabel('Actual Label', fontsize=14)
    plt.ylabel('Predicted Label', fontsize=14)

    metrics_text = (
        f"Accuracy: {cm_dict['accuracy']*100:.2f}%   "
        f"Sensitivity (TAR): {cm_dict['TAR']*100:.2f}%   "
        f"Specificity (TRR): {cm_dict['TRR']*100:.2f}%   "
        f"F1: {cm_dict['f1_score']:.3f}"
    )
    plt.gcf().text(0.5, -0.08, metrics_text, ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Report-style confusion matrix saved to {output_path}")


def plot_confusion_matrix_poster(cm_dict, output_path="confusion_matrix_poster.png"):
    """
    Plot a poster-ready confusion matrix with bold styling and large annotations.
    """
    matrix = np.array([[cm_dict['TP'], cm_dict['FP']], [cm_dict['FN'], cm_dict['TN']]])
    total = matrix.sum() if matrix.sum() > 0 else 1
    annot = np.empty(matrix.shape, dtype=object)
    labels = [['TP', 'FP'], ['FN', 'TN']]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = int(matrix[i, j])
            percent = 100.0 * count / total
            annot[i, j] = f"{labels[i][j]}\n{count}\n{percent:.1f}%"

# Fixed 2x2 color map: TP=orange, FP=teal, FN=teal, TN=orange
    ORANGE = '#E8844A'   # orangish — for TP and TN
    TEAL   = '#0E4F50'   # dark bluegreen — for FP and FN
    cell_colors = np.array([[ORANGE, TEAL], [TEAL, ORANGE]])  # [[TP,FP],[FN,TN]]

    # Build a 2x2 colormap from the fixed colors (values 0,1,2,3 map to cells)
    color_matrix = np.array([[0, 1], [1, 0]], dtype=float)  # 0=orange, 1=teal
    fixed_cmap = LinearSegmentedColormap.from_list('fixed', [ORANGE, TEAL], N=2)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        color_matrix,
        annot=annot,
        fmt='',
        cmap=fixed_cmap,
        vmin=0, vmax=1,
        cbar=False,
        annot_kws={'size': 22, 'weight': 'bold', 'color': 'white'},
        xticklabels=False,
        yticklabels=False,
        linewidths=2.0,
        linecolor='white',
        square=True,
        ax=ax
    )

    # Force all annotation text to white
    for text in ax.texts:
        text.set_color('white')

    # Remove all axes decoration — just the matrix
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(left=False, bottom=False)

    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Poster-ready confusion matrix saved to {output_path}")


def plot_det_curve(far_array, frr_array, eer_metrics=None, output_path="det_curve.png"):
    """
    Plot Detection Error Tradeoff (DET) curve using probit scaling.
    """
    far_array = np.asarray(far_array)
    frr_array = np.asarray(frr_array)
    eps = 1e-6
    x_values = np.clip(far_array, eps, 1 - eps)
    y_values = np.clip(frr_array, eps, 1 - eps)

    use_probit = True
    try:
        x_plot = norm.ppf(x_values)
        y_plot = norm.ppf(y_values)
    except Exception:
        use_probit = False
        x_plot = far_array * 100.0
        y_plot = frr_array * 100.0

    plt.figure(figsize=(10, 8))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2.5, label='DET Curve')

    if eer_metrics is not None:
        eer_x = norm.ppf(np.clip(eer_metrics['far_at_eer'], eps, 1 - eps)) if use_probit else eer_metrics['far_at_eer'] * 100.0
        eer_y = norm.ppf(np.clip(eer_metrics['frr_at_eer'], eps, 1 - eps)) if use_probit else eer_metrics['frr_at_eer'] * 100.0
        plt.plot(eer_x, eer_y, 'ro', markersize=10,
                 label=f"EER = {eer_metrics['eer']*100:.2f}%")

    if use_probit:
        tick_rates = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40])
        tick_positions = norm.ppf(tick_rates / 100.0)
        plt.xticks(tick_positions, [f"{t}%" for t in tick_rates])
        plt.yticks(tick_positions, [f"{t}%" for t in tick_rates])
        plt.xlabel('FAR (%)', fontsize=12)
        plt.ylabel('FRR (%)', fontsize=12)
    else:
        plt.xlabel('False Acceptance Rate (%)', fontsize=12)
        plt.ylabel('False Rejection Rate (%)', fontsize=12)

    plt.title('DET Curve — standard NIST SRE evaluation format', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"DET curve saved to {output_path}")


def plot_mfcc_comparison(audio_paths_dict, output_path="mfcc_comparison.png"):
    """
    Plot MFCC heatmaps for a set of audio examples for speaker verification analysis.
    """
    from core.features import extract_mfcc
    
    labels = list(audio_paths_dict.keys())
    count = len(labels)
    cols = min(2, count)
    rows = int(np.ceil(count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = np.array(axes).reshape(-1)

    for idx, (label, path) in enumerate(audio_paths_dict.items()):
        audio, sr = librosa.load(path, sr=16000)
        audio = librosa.util.normalize(audio)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        mfcc = extract_mfcc(audio, sr=sr, n_mfcc=13, include_deltas=False, apply_cmvn=False)

        ax = axes[idx]
        im = ax.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
        ax.set_title(label, fontsize=11)
        ax.set_ylabel('MFCC Coefficient Index')
        ax.set_xlabel('Time Frame')
        ax.set_yticks(range(13))
        ax.set_yticklabels(range(13))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Coefficient Value')

    for idx in range(count, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('MFCC Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"MFCC comparison plot saved to {output_path}")


def plot_dtw_alignment(audio_path_1, audio_path_2, label_1, label_2, output_path="dtw_alignment.png"):
    """
    Plot DTW alignment cost matrix and optimal warping path for two audio recordings.
    """
    from core.dtw import dtw_cost_matrix

    audio1, sr1 = librosa.load(audio_path_1, sr=16000)
    audio1 = librosa.util.normalize(audio1)
    audio1, _ = librosa.effects.trim(audio1, top_db=20)
    mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr1, n_mfcc=13, hop_length=512, n_fft=2048)

    audio2, sr2 = librosa.load(audio_path_2, sr=16000)
    audio2 = librosa.util.normalize(audio2)
    audio2, _ = librosa.effects.trim(audio2, top_db=20)
    mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr2, n_mfcc=13, hop_length=512, n_fft=2048)

    cost_matrix, path = dtw_cost_matrix(mfcc1.T, mfcc2.T)
    path_y = [p[0] for p in path]
    path_x = [p[1] for p in path]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, height_ratios=[1, 2], figure=fig)

    ax_mfcc1 = fig.add_subplot(gs[0, 0])
    ax_mfcc2 = fig.add_subplot(gs[0, 1])
    ax_cost = fig.add_subplot(gs[1, 0])
    ax_path = fig.add_subplot(gs[1, 1])

    ax_mfcc1.plot(mfcc1[0], color='navy')
    ax_mfcc1.set_title(f'{label_1} — first MFCC coefficient')
    ax_mfcc1.set_xlabel('Frame')
    ax_mfcc1.set_ylabel('Coefficient Value')

    ax_mfcc2.plot(mfcc2[0], color='darkgreen')
    ax_mfcc2.set_title(f'{label_2} — first MFCC coefficient')
    ax_mfcc2.set_xlabel('Frame')
    ax_mfcc2.set_ylabel('Coefficient Value')

    im_cost = ax_cost.imshow(cost_matrix, origin='lower', aspect='auto', cmap='viridis')
    fig.colorbar(im_cost, ax=ax_cost, label='Accumulated DTW Cost')
    ax_cost.set_title('DTW Accumulated Cost Matrix')
    ax_cost.set_xlabel('Sequence 2 Index')
    ax_cost.set_ylabel('Sequence 1 Index')

    ax_path.imshow(cost_matrix, origin='lower', aspect='auto', cmap='viridis')
    ax_path.plot(path_x, path_y, color='red', linewidth=2)
    ax_path.set_title('DTW Optimal Warping Path')
    ax_path.set_xlabel('Sequence 2 Index')
    ax_path.set_ylabel('Sequence 1 Index')

    plt.suptitle(f'DTW Alignment: {label_1} vs {label_2}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"DTW alignment visualization saved to {output_path}")


def plot_dtw_warping_only(audio_path_1, audio_path_2, label_1, label_2, output_path="dtw_warping_only.png"):
    """
    Plot only the two MFCC waveforms and the optimal DTW warping path (no cost matrix).
    """
    from core.dtw import dtw_cost_matrix

    audio1, sr1 = librosa.load(audio_path_1, sr=16000)
    audio1 = librosa.util.normalize(audio1)
    audio1, _ = librosa.effects.trim(audio1, top_db=20)
    mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr1, n_mfcc=13, hop_length=512, n_fft=2048)

    audio2, sr2 = librosa.load(audio_path_2, sr=16000)
    audio2 = librosa.util.normalize(audio2)
    audio2, _ = librosa.effects.trim(audio2, top_db=20)
    mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr2, n_mfcc=13, hop_length=512, n_fft=2048)

    cost_matrix, path = dtw_cost_matrix(mfcc1.T, mfcc2.T)
    path_y = [p[0] for p in path]
    path_x = [p[1] for p in path]

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, height_ratios=[1, 2], figure=fig)

    ax_mfcc1 = fig.add_subplot(gs[0, 0])
    ax_mfcc2 = fig.add_subplot(gs[0, 1])
    ax_path  = fig.add_subplot(gs[1, :])  # full-width bottom panel

    ax_mfcc1.plot(mfcc1[0], color='navy')
    ax_mfcc1.set_title(f'{label_1} — first MFCC coefficient')
    ax_mfcc1.set_xlabel('Frame')
    ax_mfcc1.set_ylabel('Coefficient Value')
    ax_mfcc1.grid(True, alpha=0.3)

    ax_mfcc2.plot(mfcc2[0], color='darkgreen')
    ax_mfcc2.set_title(f'{label_2} — first MFCC coefficient')
    ax_mfcc2.set_xlabel('Frame')
    ax_mfcc2.set_ylabel('Coefficient Value')
    ax_mfcc2.grid(True, alpha=0.3)

    ax_path.imshow(cost_matrix, origin='lower', aspect='auto', cmap='viridis')
    ax_path.plot(path_x, path_y, color='red', linewidth=2, label='Warping path')
    ax_path.set_title('DTW Optimal Warping Path')
    ax_path.set_xlabel(f'Sequence 2 — {label_2} (frames)')
    ax_path.set_ylabel(f'Sequence 1 — {label_1} (frames)')
    ax_path.legend(fontsize=10, loc='upper left')

    plt.suptitle(f'DTW Alignment: {label_1} vs {label_2}', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"DTW warping path plot saved to {output_path}")


def plot_feature_extraction_stages(audio_path, speaker_label, output_path="feature_extraction_stages.png"):
    """
    Produce a 4-panel figure showing progressive feature extraction stages for thesis Section 4.3.

    Panels: raw waveform with silence boundaries, static MFCC (13),
    MFCC+delta+delta-delta (39), and after CMVN normalization (39).
    """
    from core.audio_utils import load_audio, normalize_audio, trim_silence
    from core.features import extract_mfcc
    from matplotlib.transforms import blended_transform_factory

    audio, sr = load_audio(audio_path, sr=16000)
    audio = normalize_audio(audio)

    # Obtain trim boundaries on the normalized (non-trimmed) signal
    _, trim_index = trim_silence(audio, sr=sr)
    trim_start_s = trim_index[0] / sr
    trim_end_s = trim_index[1] / sr

    # Trim for feature extraction
    trimmed = audio[trim_index[0]:trim_index[1]]

    mfcc_static = extract_mfcc(trimmed, sr=sr, n_mfcc=13, include_deltas=False, apply_cmvn=False)
    mfcc_delta  = extract_mfcc(trimmed, sr=sr, n_mfcc=13, include_deltas=True,  apply_cmvn=False)
    mfcc_cmvn   = extract_mfcc(trimmed, sr=sr, n_mfcc=13, include_deltas=True,  apply_cmvn=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # --- Panel 1: Waveform ---
    ax1 = axes[0]
    t = np.arange(len(audio)) / sr
    ax1.plot(t, audio, color='steelblue', linewidth=0.5)
    ax1.axvline(trim_start_s, color='red', linestyle='--', linewidth=1.5, label='Silence boundary')
    ax1.axvline(trim_end_s,   color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('Audio Waveform with Silence Boundaries', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Static MFCC (13 coefficients) ---
    ax2 = axes[1]
    im2 = ax2.imshow(mfcc_static, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_xlabel('Frame Index', fontsize=11)
    ax2.set_ylabel('Coefficient Index', fontsize=11)
    ax2.set_title('13 Static MFCC Coefficients', fontsize=12, fontweight='bold')
    ax2.set_yticks(range(0, 13, 2))
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label('Coefficient Value', fontsize=10)

    # --- Panel 3: MFCC + Δ + ΔΔ (39 dimensions) ---
    ax3 = axes[2]
    im3 = ax3.imshow(mfcc_delta, aspect='auto', origin='lower', cmap='viridis')
    ax3.axhline(12.5, color='white', linestyle='--', linewidth=1.5)
    ax3.axhline(25.5, color='white', linestyle='--', linewidth=1.5)
    ax3.set_xlabel('Frame Index', fontsize=11)
    ax3.set_ylabel('Feature Dimension', fontsize=11)
    ax3.set_title('39-Dimensional Feature Vector (MFCC + Δ + ΔΔ)', fontsize=12, fontweight='bold')
    cbar3 = fig.colorbar(im3, ax=ax3)
    cbar3.set_label('Coefficient Value', fontsize=10)
    # Block labels in axes coordinates (y: 0=bottom, 1=top with origin='lower')
    for yc, lbl in [(6/38, 'MFCC'), (19/38, 'Δ'), (32/38, 'ΔΔ')]:
        ax3.text(-0.04, yc, lbl, transform=ax3.transAxes,
                 va='center', ha='right', fontsize=10, clip_on=False)

    # --- Panel 4: After CMVN (39 dimensions) ---
    ax4 = axes[3]
    im4 = ax4.imshow(mfcc_cmvn, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-3, vmax=3)
    ax4.axhline(12.5, color='white', linestyle='--', linewidth=1.5)
    ax4.axhline(25.5, color='white', linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Frame Index', fontsize=11)
    ax4.set_ylabel('Feature Dimension', fontsize=11)
    ax4.set_title('After CMVN Normalization (zero mean, unit variance)', fontsize=12, fontweight='bold')
    cbar4 = fig.colorbar(im4, ax=ax4)
    cbar4.set_label('Coefficient Value', fontsize=10)
    for yc, lbl in [(6/38, 'MFCC'), (19/38, 'Δ'), (32/38, 'ΔΔ')]:
        ax4.text(-0.04, yc, lbl, transform=ax4.transAxes,
                 va='center', ha='right', fontsize=10, clip_on=False)

    plt.suptitle(f'Feature Extraction Stages — Speaker: {speaker_label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Feature extraction stages plot saved to {output_path}")
