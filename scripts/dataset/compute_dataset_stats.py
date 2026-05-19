"""
Compute per-dataset statistics for thesis Section 6.1.

Usage:
    python scripts/dataset/compute_dataset_stats.py
    python scripts/dataset/compute_dataset_stats.py --dataset heysnips
"""

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import librosa
from data_handlers import load_dataset_config


def _estimate_snr(audio, sr):
    """Return (snr_db, has_noise) for one audio signal."""
    speech, _ = librosa.effects.trim(audio, top_db=20)
    trimmed_samples = len(speech)
    total_samples = len(audio)
    noise_samples = total_samples - trimmed_samples
    if noise_samples < 10:
        return None, False
    signal_power = np.mean(speech ** 2)
    # Collect all silence frames (before and after speech region)
    _, trim_idx = librosa.effects.trim(audio, top_db=20)
    noise_region = np.concatenate([audio[:trim_idx[0]], audio[trim_idx[1]:]])
    if len(noise_region) == 0:
        return None, False
    noise_power = np.mean(noise_region ** 2)
    if noise_power < 1e-12 or signal_power < 1e-12:
        return None, False
    snr_db = 10 * math.log10(signal_power / noise_power)
    return snr_db, True


def _collect_wav_files(data_path):
    return sorted(glob.glob(os.path.join(data_path, '**', '*.wav'), recursive=True))


def compute_stats(dataset_name, cfg):
    data_path = cfg.get('data_path', os.path.join('data', dataset_name))

    if not os.path.isdir(data_path):
        print(f"[WARN] Data path not found: {data_path} — skipping {dataset_name}")
        return None

    speakers = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    n_speakers = len(speakers)

    wav_files = _collect_wav_files(data_path)
    n_recordings = len(wav_files)

    durations = []
    snr_values = []
    total = n_recordings

    for i, fpath in enumerate(wav_files, 1):
        print(f"  Processing {i}/{total}...", end='\r')
        try:
            duration = librosa.get_duration(path=fpath)
            durations.append(duration)

            audio, sr = librosa.load(fpath, sr=16000, mono=True)
            snr, has_noise = _estimate_snr(audio, sr)
            if has_noise and snr is not None:
                snr_values.append(snr)
        except Exception as e:
            print(f"\n  [WARN] Could not process {fpath}: {e}")

    print()  # newline after progress indicator

    stats = {
        'dataset': dataset_name,
        'n_speakers': n_speakers,
        'n_recordings': n_recordings,
        'duration_min_s': round(float(np.min(durations)), 4) if durations else None,
        'duration_mean_s': round(float(np.mean(durations)), 4) if durations else None,
        'duration_max_s': round(float(np.max(durations)), 4) if durations else None,
        'mean_snr_db': round(float(np.mean(snr_values)), 2) if snr_values else None,
        'snr_files_used': len(snr_values),
    }

    # Hey Snips gender distribution
    if dataset_name == 'heysnips':
        gender_counts = _find_heysnips_gender(data_path)
        stats['gender_distribution'] = gender_counts

    return stats


def _find_heysnips_gender(data_path):
    candidates = [
        os.path.join(data_path, 'info.json'),
    ] + glob.glob(os.path.join(data_path, '*.json'))

    for json_path in candidates:
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            # Try common structures: list of speaker dicts or dict keyed by speaker id
            counts = {'male': 0, 'female': 0, 'other': 0}
            found = False
            entries = data if isinstance(data, list) else data.values() if isinstance(data, dict) else []
            for entry in entries:
                if isinstance(entry, dict) and 'gender' in entry:
                    g = str(entry['gender']).lower()
                    if g in counts:
                        counts[g] += 1
                    else:
                        counts['other'] += 1
                    found = True
            if found:
                return counts
        except Exception:
            continue

    return 'gender metadata not available'


def print_table(stats):
    print()
    print(f"{'='*55}")
    print(f"  Dataset: {stats['dataset']}")
    print(f"{'='*55}")
    print(f"  {'Total speakers':<30} {stats['n_speakers']}")
    print(f"  {'Total recordings':<30} {stats['n_recordings']}")
    if stats['duration_min_s'] is not None:
        print(f"  {'Min duration (s)':<30} {stats['duration_min_s']:.3f}")
        print(f"  {'Mean duration (s)':<30} {stats['duration_mean_s']:.3f}")
        print(f"  {'Max duration (s)':<30} {stats['duration_max_s']:.3f}")
    if stats['mean_snr_db'] is not None:
        print(f"  {'Estimated SNR (dB)':<30} {stats['mean_snr_db']:.2f}  (from {stats['snr_files_used']} files)")
    else:
        print(f"  {'Estimated SNR (dB)':<30} N/A (no silence detected)")
    if 'gender_distribution' in stats:
        gd = stats['gender_distribution']
        if isinstance(gd, dict):
            print(f"  {'Gender distribution':<30} M={gd.get('male', 0)}  F={gd.get('female', 0)}")
        else:
            print(f"  {'Gender distribution':<30} {gd}")
    print(f"{'='*55}")
    print()


def save_json(stats, cfg, dataset_name):
    results_path = cfg.get('results_path', os.path.join('evaluation_results', dataset_name))
    os.makedirs(results_path, exist_ok=True)
    out_path = os.path.join(results_path, 'dataset_stats.json')
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute dataset statistics')
    parser.add_argument('--dataset', default=None, help='Dataset name from config/dataset_config.json (omit for all)')
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent.parent / 'config' / 'dataset_config.json'
    with open(config_path, 'r') as f:
        all_configs = json.load(f)

    SKIP_DATASETS = {'speech_commands'}

    if args.dataset:
        if args.dataset not in all_configs:
            print(f"Unknown dataset: {args.dataset}")
            sys.exit(1)
        datasets_to_process = {args.dataset: all_configs[args.dataset]}
    else:
        datasets_to_process = {k: v for k, v in all_configs.items() if k not in SKIP_DATASETS}

    for dataset_name, cfg in datasets_to_process.items():
        print(f"\nComputing stats for '{dataset_name}'...")
        stats = compute_stats(dataset_name, cfg)
        if stats is None:
            continue
        print_table(stats)
        save_json(stats, cfg, dataset_name)


if __name__ == '__main__':
    main()
