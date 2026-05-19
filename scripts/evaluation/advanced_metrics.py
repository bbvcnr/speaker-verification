import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path so we can import core and evaluation modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import core.audio_utils as audio_utils
import core.features as features
from core.verification import create_template
import core.dtw as dtw
from data_handlers import load_dataset_config
from evaluation.metrics import (
    compute_confusion_matrix,
    compute_fisher_discriminant_ratio,
    compute_bhattacharyya_distance,
    compute_per_speaker_eer,
    compute_roc_curve,
    compute_eer,
)
from evaluation.visualizations import (
    plot_score_histograms,
    plot_roc_curve,
    plot_threshold_analysis,
    plot_det_curve,
    plot_mfcc_comparison,
    plot_dtw_alignment,
    plot_confusion_matrix,
    plot_confusion_matrix_report,
    plot_confusion_matrix_poster,
    plot_speaker_distance_matrix,
    plot_feature_extraction_stages,
)


def load_and_process_audio(path):
    audio, sr = audio_utils.load_audio(path, sr=16000)
    audio = audio_utils.normalize_audio(audio)
    audio, _ = audio_utils.trim_silence(audio, sr=sr)
    return audio, sr


def build_dataset_scores(dataset_path):
    np.random.seed(42)
    persons = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    persons = sorted(persons)

    genuine_scores = []
    impostor_scores = []

    for person in persons:
        person_path = os.path.join(dataset_path, person)
        files = sorted([f for f in os.listdir(person_path) if f.endswith('.wav')])
        if len(files) < 2:
            continue

        for test_file in files:
            enrollment_files = [f for f in files if f != test_file]
            enrollment_paths = [os.path.join(person_path, f) for f in enrollment_files]
            template = create_template(enrollment_paths)

            test_path = os.path.join(person_path, test_file)
            test_audio, _ = load_and_process_audio(test_path)
            test_mfcc = features.extract_mfcc(test_audio, sr=16000)

            genuine_dist = dtw.dtw_distance(template.T, test_mfcc.T)
            if np.isfinite(genuine_dist):
                genuine_scores.append(genuine_dist)

            for other_person in persons:
                if other_person == person:
                    continue
                other_path = os.path.join(dataset_path, other_person)
                other_files = [f for f in os.listdir(other_path) if f.endswith('.wav')]
                if not other_files:
                    continue
                random_file = np.random.choice(other_files)
                impostor_path = os.path.join(other_path, random_file)
                impostor_audio, _ = load_and_process_audio(impostor_path)
                impostor_mfcc = features.extract_mfcc(impostor_audio, sr=16000)
                impostor_dist = dtw.dtw_distance(template.T, impostor_mfcc.T)
                if np.isfinite(impostor_dist):
                    impostor_scores.append(impostor_dist)

    return genuine_scores, impostor_scores


def compute_speaker_distance_matrix(dataset_path):
    """
    Compute an average speaker-to-speaker DTW matrix for a dataset.

    The diagonal entries are based on a held-out same-speaker utterance,
    while off-diagonal entries are average distances from one speaker's
    template to all utterances of another speaker.
    """
    persons = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    persons = sorted(persons)
    n = len(persons)
    speaker_matrix = np.full((n, n), np.nan)

    for i, person in enumerate(persons):
        person_path = os.path.join(dataset_path, person)
        files = sorted([f for f in os.listdir(person_path) if f.endswith('.wav')])
        if len(files) < 2:
            continue

        template_files = [os.path.join(person_path, f) for f in files[:-1]]
        held_out_path = os.path.join(person_path, files[-1])
        template = create_template(template_files)

        held_audio, _ = load_and_process_audio(held_out_path)
        held_mfcc = features.extract_mfcc(held_audio, sr=16000)
        speaker_matrix[i, i] = dtw.dtw_distance(template.T, held_mfcc.T)

        for j, other_person in enumerate(persons):
            if i == j:
                continue

            other_path = os.path.join(dataset_path, other_person)
            other_files = sorted([f for f in os.listdir(other_path) if f.endswith('.wav')])
            if not other_files:
                continue

            distances = []
            for other_file in other_files:
                other_audio, _ = load_and_process_audio(os.path.join(other_path, other_file))
                other_mfcc = features.extract_mfcc(other_audio, sr=16000)
                distances.append(dtw.dtw_distance(template.T, other_mfcc.T))

            if distances:
                speaker_matrix[i, j] = float(np.mean(distances))

    return persons, speaker_matrix


def main(dataset_name='custom_dataset'):
    cfg = load_dataset_config(dataset_name)
    dataset_path = cfg.get('data_path', os.path.join('data', dataset_name))
    results_dir = os.path.join(cfg.get('results_path', os.path.join('evaluation_results', dataset_name)), 'advanced')
    os.makedirs(results_dir, exist_ok=True)

    threshold_config_path = cfg.get('threshold_config', os.path.join('config', f'{dataset_name}_threshold.json'))
    with open(threshold_config_path, 'r') as f:
        threshold_config = json.load(f)

    threshold = float(threshold_config.get('threshold', 0.0))
    dataset_label = dataset_name.replace('_', ' ').title()

    genuine_scores, impostor_scores = build_dataset_scores(dataset_path)
    roc_data = compute_roc_curve(genuine_scores, impostor_scores)
    eer_data = compute_eer(genuine_scores, impostor_scores)
    cm_data = compute_confusion_matrix(genuine_scores, impostor_scores, threshold)
    fisher_dr = compute_fisher_discriminant_ratio(genuine_scores, impostor_scores)
    bhatta = compute_bhattacharyya_distance(genuine_scores, impostor_scores)
    per_speaker = compute_per_speaker_eer(dataset_path, dataset_path, audio_utils, features, dtw)

    plot_score_histograms(genuine_scores, impostor_scores, output_path=os.path.join(results_dir, 'score_histograms.png'))
    plot_roc_curve(roc_data['far'], roc_data['frr'], eer_metrics=eer_data, output_path=os.path.join(results_dir, 'roc_curve.png'))
    plot_threshold_analysis(roc_data['thresholds'], roc_data['far'], roc_data['frr'], eer_metrics=eer_data, output_path=os.path.join(results_dir, 'threshold_analysis.png'))
    plot_confusion_matrix(cm_data, output_path=os.path.join(results_dir, 'confusion_matrix.png'))
    plot_confusion_matrix_report(cm_data, output_path=os.path.join(results_dir, 'confusion_matrix_report.png'))
    plot_confusion_matrix_poster(cm_data, output_path=os.path.join(results_dir, 'confusion_matrix_poster.png'))

    speaker_labels, speaker_distance_matrix = compute_speaker_distance_matrix(dataset_path)
    plot_speaker_distance_matrix(
        speaker_distance_matrix,
        speaker_labels,
        output_path=os.path.join(results_dir, 'speaker_distance_matrix.png')
    )

    roc_data_det = compute_roc_curve(genuine_scores, impostor_scores, n_thresholds=5000)
    plot_det_curve(roc_data_det['far'], roc_data_det['frr'], eer_metrics=eer_data, output_path=os.path.join(results_dir, 'det_curve.png'))

    mfcc_examples = {}
    example_paths = [
        ('Genuine A (nejra/sample_1)', os.path.join(dataset_path, 'nejra', 'sample_1.wav')),
        ('Genuine B (nejra/sample_5)', os.path.join(dataset_path, 'nejra', 'sample_5.wav')),
        ('Impostor (alem/sample_1)', os.path.join(dataset_path, 'alem', 'sample_1.wav')),
    ]
    for label, path in example_paths:
        if os.path.exists(path):
            mfcc_examples[label] = path

    wrong_phrase_path = os.path.join('data', 'experiments', 'nejra_wrong', 'sample_1.wav')
    if os.path.exists(wrong_phrase_path):
        mfcc_examples['Wrong phrase (nejra_wrong/sample_1)'] = wrong_phrase_path

    if mfcc_examples:
        plot_mfcc_comparison(mfcc_examples, output_path=os.path.join(results_dir, 'mfcc_comparison.png'))

    alignment_examples = [
        (
            os.path.join(dataset_path, 'nejra', 'sample_1.wav'),
            os.path.join(dataset_path, 'nejra', 'sample_5.wav'),
            'Nejra sample 1',
            'Nejra sample 5',
            'dtw_alignment_genuine.png'
        ),
        (
            os.path.join(dataset_path, 'nejra', 'sample_1.wav'),
            os.path.join(dataset_path, 'alem', 'sample_1.wav'),
            'Nejra sample 1',
            'Alem sample 1',
            'dtw_alignment_impostor.png'
        ),
    ]
    for p1, p2, label1, label2, output_name in alignment_examples:
        if os.path.exists(p1) and os.path.exists(p2):
            plot_dtw_alignment(
                p1,
                p2,
                label1,
                label2,
                output_path=os.path.join(results_dir, output_name),
            )

    # Build speaker→files mapping for feature extraction plot
    selected_speakers = {}
    for _person in sorted(os.listdir(dataset_path)):
        _person_path = os.path.join(dataset_path, _person)
        if os.path.isdir(_person_path):
            _files = sorted([
                os.path.join(_person_path, f)
                for f in os.listdir(_person_path) if f.endswith('.wav')
            ])
            if _files:
                selected_speakers[_person] = _files

    # Feature extraction stages — for thesis Section 4.3
    first_speaker = sorted(selected_speakers.keys())[0]
    first_audio = sorted(selected_speakers[first_speaker])[0]
    plot_feature_extraction_stages(
        audio_path=first_audio,
        speaker_label=first_speaker,
        output_path=os.path.join(results_dir, "feature_extraction_stages.png")
    )

    metrics_output = {
        'threshold': threshold,
        'confusion_matrix': cm_data,
        'fisher_discriminant_ratio': fisher_dr,
        'bhattacharyya_distance': bhatta,
        'eer': eer_data['eer'],
        'optimal_threshold': eer_data['optimal_threshold'],
        'per_speaker_eer': per_speaker,
        'summary': {
            'n_genuine_trials': len(genuine_scores),
            'n_impostor_trials': len(impostor_scores),
        },
        'generated_at': datetime.now().isoformat(),
    }

    with open(os.path.join(results_dir, 'advanced_metrics.json'), 'w') as json_file:
        json.dump(metrics_output, json_file, indent=2)

    print('╔════════════════════════════════════════════════╗')
    print(f'║     ADVANCED METRICS — {dataset_label}          ║')
    print('╠════════════════════════════════════════════════╣')
    print(f'║ Threshold: {threshold:.2f}'.ljust(48) + '║')
    print('║'.ljust(48) + '║')
    print('║ CONFUSION MATRIX:                            ║')
    print(f'║   TP: {cm_data["TP"]}  FN: {cm_data["FN"]}  TAR: {cm_data["TAR"]*100:.1f}%'.ljust(48) + '║')
    print(f'║   FP: {cm_data["FP"]}  TN: {cm_data["TN"]}  FAR: {cm_data["FAR"]*100:.1f}%'.ljust(48) + '║')
    print(f'║   F1 Score: {cm_data["f1_score"]:.2f}'.ljust(48) + '║')
    print('║'.ljust(48) + '║')
    print('║ SEPARABILITY METRICS:                        ║')
    print(f'║   Fisher Discriminant Ratio: {fisher_dr:.2f}'.ljust(48) + '║')
    print(f'║   Bhattacharyya Distance:    {bhatta:.2f}'.ljust(48) + '║')
    print('║'.ljust(48) + '║')
    print('║ PER-SPEAKER EER:                             ║')
    for person, stats in per_speaker.items():
        if stats.get('eer') is None:
            line = f'║   {person}: insufficient data'.ljust(48) + '║'
        else:
            line = f'║   {person}: {stats["eer"]*100:.1f}%  ({stats["n_genuine"]} genuine, {stats["n_impostor"]} impostor)'.ljust(48) + '║'
        print(line)
    print('╚════════════════════════════════════════════════╝')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute advanced metrics for a dataset')
    parser.add_argument('--dataset', default='custom_dataset', help='Dataset name from config/dataset_config.json')
    args = parser.parse_args()
    main(dataset_name=args.dataset)
