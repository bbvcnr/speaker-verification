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


def compute_confusion_matrix(genuine_scores, impostor_scores, threshold):
    """
    Compute confusion matrix counts and rates at a given threshold.

    Args:
        genuine_scores: list/array of distances for same-speaker pairs
        impostor_scores: list/array of distances for different-speaker pairs
        threshold: decision threshold

    Returns:
        dict with TP, TN, FP, FN, TAR, TRR, FAR, FRR, precision, f1_score
    """
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    tp = int(np.sum(genuine_scores < threshold))
    fn = int(np.sum(genuine_scores >= threshold))
    tn = int(np.sum(impostor_scores >= threshold))
    fp = int(np.sum(impostor_scores < threshold))

    tar = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    trr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tar
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TAR': tar,
        'TRR': trr,
        'FAR': far,
        'FRR': frr,
        'precision': precision,
        'accuracy': accuracy,
        'f1_score': f1_score
    }


def compute_fisher_discriminant_ratio(genuine_scores, impostor_scores):
    """
    Compute Fisher's discriminant ratio for genuine and impostor score distributions.

    Higher values indicate better separability.
    """
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    mu_g = np.mean(genuine_scores)
    mu_i = np.mean(impostor_scores)
    var_g = np.var(genuine_scores)
    var_i = np.var(impostor_scores)

    denominator = var_g + var_i
    if denominator == 0:
        return float('inf')

    return ((mu_i - mu_g) ** 2) / denominator


def compute_bhattacharyya_distance(genuine_scores, impostor_scores):
    """
    Compute Bhattacharyya distance between two Gaussian score distributions.
    """
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    mu_g = np.mean(genuine_scores)
    mu_i = np.mean(impostor_scores)
    sigma_g2 = np.var(genuine_scores)
    sigma_i2 = np.var(impostor_scores)

    # Stabilize zero variance
    sigma_g2 = max(sigma_g2, 1e-9)
    sigma_i2 = max(sigma_i2, 1e-9)

    first_term = 0.25 * np.log(0.25 * (sigma_g2 / sigma_i2 + sigma_i2 / sigma_g2 + 2.0))
    second_term = 0.25 * ((mu_i - mu_g) ** 2) / (sigma_g2 + sigma_i2)

    return first_term + second_term


def compute_per_speaker_eer(data_dir, templates_dir, audio_utils, features, dtw):
    """
    Compute per-speaker equal error rate for each person in the dataset.

    Genuine trials are test recordings against the speaker's own template.
    Impostor trials use all other speaker templates.
    """
    import os
    from core.verification import create_template

    persons = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    persons = sorted(persons)
    results = {}

    # Precompute impostor templates for all speakers
    templates = {}
    for person in persons:
        person_path = os.path.join(templates_dir or data_dir, person)
        files = sorted([os.path.join(person_path, f) for f in os.listdir(person_path) if f.endswith('.wav')])
        if files:
            templates[person] = create_template(files)

    for person in persons:
        person_path = os.path.join(data_dir, person)
        files = sorted([f for f in os.listdir(person_path) if f.endswith('.wav')])
        if len(files) < 2:
            results[person] = {'eer': None, 'note': 'insufficient data'}
            continue

        genuine_scores = []
        impostor_scores = []

        for test_file in files:
            test_path = os.path.join(person_path, test_file)
            audio, _ = audio_utils.load_audio(test_path, sr=16000)
            audio = audio_utils.normalize_audio(audio)
            audio, _ = audio_utils.trim_silence(audio, sr=16000)
            test_mfcc = features.extract_mfcc(audio, sr=16000)

            enrollment_files = [f for f in files if f != test_file]
            enrollment_paths = [os.path.join(person_path, f) for f in enrollment_files]
            own_template = create_template(enrollment_paths)

            genuine_dist = dtw.dtw_distance(own_template.T, test_mfcc.T)
            if np.isfinite(genuine_dist):
                genuine_scores.append(genuine_dist)

            for other_person, other_template in templates.items():
                if other_person == person:
                    continue
                impostor_dist = dtw.dtw_distance(other_template.T, test_mfcc.T)
                if np.isfinite(impostor_dist):
                    impostor_scores.append(impostor_dist)

        if len(genuine_scores) < 2 or len(impostor_scores) == 0:
            results[person] = {'eer': None, 'note': 'insufficient data'}
            continue

        eer_data = compute_eer(genuine_scores, impostor_scores)
        results[person] = {
            'eer': eer_data['eer'],
            'n_genuine': len(genuine_scores),
            'n_impostor': len(impostor_scores)
        }

    return results
