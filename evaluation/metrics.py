"""
Evaluation metrics for speaker verification.

Implements standard metrics used in speaker verification research:
- FAR (False Acceptance Rate)
- FRR (False Rejection Rate)
- ROC (Receiver Operating Characteristic) curve
- EER (Equal Error Rate)
"""

import numpy as np


def compute_far_frr(genuine_scores, impostor_scores, threshold):
    """
    Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR)
    at a given threshold.

    FAR: percentage of impostor pairs incorrectly accepted as genuine
    FRR: percentage of genuine pairs incorrectly rejected

    Args:
        genuine_scores: list/array of distances for same-speaker pairs
        impostor_scores: list/array of distances for different-speaker pairs
        threshold: decision threshold

    Returns:
        dict with keys:
            - 'far': False Acceptance Rate (0-1)
            - 'frr': False Rejection Rate (0-1)
    """
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # False accepts: impostor scores below threshold (incorrectly accepted)
    false_accepts = np.sum(impostor_scores < threshold)
    far = false_accepts / len(impostor_scores)

    # False rejects: genuine scores above threshold (incorrectly rejected)
    false_rejects = np.sum(genuine_scores >= threshold)
    frr = false_rejects / len(genuine_scores)

    return {'far': far, 'frr': frr}


def compute_roc_curve(genuine_scores, impostor_scores, n_thresholds=100):
    """
    Compute Receiver Operating Characteristic (ROC) curve.

    Sweeps thresholds and computes FAR/FRR pairs.

    Args:
        genuine_scores: list/array of distances for same-speaker pairs
        impostor_scores: list/array of distances for different-speaker pairs
        n_thresholds: number of threshold points to evaluate

    Returns:
        dict with keys:
            - 'far': array of FAR values
            - 'frr': array of FRR values
            - 'thresholds': threshold values used
    """
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # Create threshold sweep across all score ranges
    min_score = min(np.min(genuine_scores), np.min(impostor_scores)) - 10
    max_score = max(np.max(genuine_scores), np.max(impostor_scores)) + 10
    thresholds = np.linspace(min_score, max_score, n_thresholds)

    far_array = []
    frr_array = []

    for threshold in thresholds:
        metrics = compute_far_frr(genuine_scores, impostor_scores, threshold)
        far_array.append(metrics['far'])
        frr_array.append(metrics['frr'])

    return {
        'far': np.array(far_array),
        'frr': np.array(frr_array),
        'thresholds': thresholds
    }


def compute_eer(genuine_scores, impostor_scores):
    """
    Compute Equal Error Rate (EER) and optimal threshold.

    EER is the threshold where FAR ≈ FRR. Lower EER indicates better performance.

    Args:
        genuine_scores: list/array of distances for same-speaker pairs
        impostor_scores: list/array of distances for different-speaker pairs

    Returns:
        dict with keys:
            - 'eer': Equal Error Rate (0-1, reported as percentage in thesis)
            - 'optimal_threshold': threshold where FAR ≈ FRR
            - 'far_at_eer': FAR value at EER
            - 'frr_at_eer': FRR value at EER
    """
    roc = compute_roc_curve(genuine_scores, impostor_scores, n_thresholds=1000)
    far_array = roc['far']
    frr_array = roc['frr']
    thresholds = roc['thresholds']

    # Find threshold where FAR and FRR are closest
    diff = np.abs(far_array - frr_array)
    eer_idx = np.argmin(diff)

    eer = (far_array[eer_idx] + frr_array[eer_idx]) / 2.0

    return {
        'eer': eer,
        'optimal_threshold': thresholds[eer_idx],
        'far_at_eer': far_array[eer_idx],
        'frr_at_eer': frr_array[eer_idx]
    }


def compute_min_tdcf(genuine_scores, impostor_scores, P_fa=0.05, P_miss=0.05):
    """
    Compute minimum Tandem Detection Cost Function (TDCF).

    Alternative metric used in NIST SRE evaluations.
    Allows weighting of false accepts vs false rejects based on application.

    Args:
        genuine_scores: list/array of distances for same-speaker pairs
        impostor_scores: list/array of distances for different-speaker pairs
        P_fa: cost weight for false accepts
        P_miss: cost weight for false rejects

    Returns:
        dict with keys:
            - 'min_tdcf': minimum TDCF value
            - 'optimal_threshold': threshold achieving min TDCF
    """
    roc = compute_roc_curve(genuine_scores, impostor_scores, n_thresholds=1000)
    far_array = roc['far']
    frr_array = roc['frr']
    thresholds = roc['thresholds']

    # TDCF = P_fa * FAR + P_miss * FRR
    tdcf_array = P_fa * far_array + P_miss * frr_array
    min_idx = np.argmin(tdcf_array)

    return {
        'min_tdcf': tdcf_array[min_idx],
        'optimal_threshold': thresholds[min_idx]
    }
