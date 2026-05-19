import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path so we can import core and evaluation modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_handlers import CustomDatasetHandler, load_dataset_config
from core.verification import create_template
from core.audio_utils import load_audio, normalize_audio, trim_silence
from core.features import extract_mfcc
from core.dtw import dtw_distance
from evaluation.metrics import compute_roc_curve, compute_eer

def tune_threshold(dataset_name='custom_dataset'):
    cfg = load_dataset_config(dataset_name)
    dataset_path = cfg.get('data_path', f'data/{dataset_name}')
    results_dir = os.path.join(cfg.get('results_path', f'evaluation_results/{dataset_name}'), 'advanced')
    config_path = cfg.get('threshold_config', f'config/{dataset_name}_threshold.json')
    enrollment_count = cfg.get('enrollment_count', 3)
    test_count = cfg.get('test_count', 2)
    impostor_speakers_per_test = cfg.get('impostor_speakers_per_test', 5)
    random_seed = cfg.get('random_seed', 42)
    dataset_label = dataset_name.replace('_', ' ').title()

    # Create directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    speaker_recordings = CustomDatasetHandler.load_speaker_recordings(dataset_path)
    selected_speakers = CustomDatasetHandler.select_speakers(
        speaker_recordings,
        min_recordings=enrollment_count + test_count,
        random_seed=random_seed,
    )
    templates, test_sets = CustomDatasetHandler.split_enrollment_test(
        selected_speakers,
        enrollment_count=enrollment_count,
        test_count=test_count,
        random_seed=random_seed,
    )
    genuine_trials, impostor_trials = CustomDatasetHandler.create_verification_trials(
        templates,
        test_sets,
        impostor_speakers_per_test=impostor_speakers_per_test,
        random_seed=random_seed,
    )

    genuine_scores = []
    impostor_scores = []
    person_files_count = {speaker: len(files) for speaker, files in selected_speakers.items()}

    def process_file(path):
        audio, _ = load_audio(path, sr=16000)
        audio = normalize_audio(audio)
        audio, _ = trim_silence(audio, sr=16000)
        return extract_mfcc(audio, sr=16000)

    for trial in genuine_trials:
        template_files = templates[trial['template_speaker']]
        template = create_template(template_files)
        test_mfcc = process_file(trial['test_file'])
        dist = dtw_distance(template.T, test_mfcc.T)
        if np.isfinite(dist):
            genuine_scores.append(dist)

    for trial in impostor_trials:
        template_files = templates[trial['template_speaker']]
        template = create_template(template_files)
        test_mfcc = process_file(trial['test_file'])
        dist = dtw_distance(template.T, test_mfcc.T)
        if np.isfinite(dist):
            impostor_scores.append(dist)

    # Compute metrics
    roc_data = compute_roc_curve(genuine_scores, impostor_scores)
    eer_data = compute_eer(genuine_scores, impostor_scores)

    # Save metrics only
    metrics = {
        "eer": eer_data['eer'],
        "optimal_threshold": eer_data['optimal_threshold'],
        "n_genuine_trials": len(genuine_scores),
        "n_impostor_trials": len(impostor_scores)
    }
    with open(os.path.join(results_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save threshold config
    config = {
        "threshold": eer_data['optimal_threshold'],
        "eer": eer_data['eer'],
        "date": datetime.now().isoformat()
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Compute stats
    genuine_mean = np.mean(genuine_scores) if genuine_scores else float('nan')
    genuine_std = np.std(genuine_scores) if genuine_scores else float('nan')
    impostor_mean = np.mean(impostor_scores) if impostor_scores else float('nan')
    impostor_std = np.std(impostor_scores) if impostor_scores else float('nan')
    separation = impostor_mean - genuine_mean

    num_persons = len(selected_speakers)
    num_persons_few_samples = sum(1 for count in person_files_count.values() if count < 5)

    # Print summary
    print("=" * 40)
    print(f"{dataset_label} THRESHOLD TUNING RESULTS")
    print("=" * 40)
    print(f"Number of persons:          {num_persons}")
    print(f"Genuine trials:      {len(genuine_scores)}")
    print(f"Impostor trials:     {len(impostor_scores)}")
    print()
    print(f"Genuine score mean:  {genuine_mean:.2f} ± {genuine_std:.2f}")
    print(f"Impostor score mean: {impostor_mean:.2f} ± {impostor_std:.2f}")
    print(f"Score separation:    {separation:.2f}")
    print()
    print(f"EER:                 {eer_data['eer']*100:.1f}%")
    print(f"Optimal threshold:   {eer_data['optimal_threshold']:.2f}")
    print()
    if num_persons_few_samples > 0:
        print(f"⚠ {num_persons_few_samples} selected persons have fewer than 5 samples — results may be unreliable.")
    print("⚠ Warning: Smaller speaker counts and short enrollment sets can distort EER estimates.")
    print("  For production: use more speakers and more enrollment/test samples per speaker.")
    print("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tune threshold for text-dependent verification')
    parser.add_argument('--dataset', default='custom_dataset', help='Dataset name from config/dataset_config.json')
    args = parser.parse_args()
    tune_threshold(dataset_name=args.dataset)