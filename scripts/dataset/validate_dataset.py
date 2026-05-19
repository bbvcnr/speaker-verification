import argparse
import os
import sys
from pathlib import Path

# Add project root to path so we can import core and evaluation modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import librosa
import numpy as np
from data_handlers import load_dataset_config
from core.audio_utils import load_audio, trim_silence

def validate_custom_dataset(dataset_path):
    """
    Validate the custom dataset in the given path.
    """
    report = []
    summary = {
        'total_persons': 0,
        'total_files': 0,
        'ok_files': 0,
        'warning_files': 0,
        'error_files': 0
    }

    # Scan for persons (directories)
    if not os.path.exists(dataset_path):
        report.append("ERROR: Dataset path does not exist.")
        return report, summary

    persons = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    summary['total_persons'] = len(persons)

    for person in sorted(persons):
        person_path = os.path.join(dataset_path, person)
        report.append(f"\nPerson: {person}")
        files = [f for f in os.listdir(person_path) if f.endswith('.wav')]
        summary['total_files'] += len(files)

        for file in sorted(files):
            file_path = os.path.join(person_path, file)
            status, reason = validate_file(file_path)
            report.append(f"  {file}: {status} - {reason}")

            if status == 'OK':
                summary['ok_files'] += 1
            elif status == 'WARNING':
                summary['warning_files'] += 1
            elif status == 'ERROR':
                summary['error_files'] += 1

    # Summary
    report.append("\nSummary:")
    report.append(f"  Total persons: {summary['total_persons']}")
    report.append(f"  Total files: {summary['total_files']}")
    report.append(f"  OK files: {summary['ok_files']}")
    report.append(f"  Warning files: {summary['warning_files']}")
    report.append(f"  Error files: {summary['error_files']}")

    # Recommendations
    if summary['error_files'] > 0 or summary['warning_files'] > 0:
        report.append("\nRecommendations:")
        if summary['error_files'] > 0:
            report.append("  - Fix ERROR files: Re-record samples that cannot be loaded or have wrong format.")
        if summary['warning_files'] > 0:
            report.append("  - Review WARNING files: Check duration, energy, and silence trimming.")

    return report, summary

def validate_file(file_path):
    """
    Validate a single audio file.
    Returns: (status, reason)
    """
    try:
        audio, sr = load_audio(file_path)
    except Exception as e:
        return 'ERROR', f"Cannot load audio: {str(e)}"

    # Check sample rate
    if sr != 16000:
        return 'ERROR', f"Sample rate is {sr}, expected 16000"

    # Check channels (mono)
    if audio.ndim != 1:
        return 'ERROR', f"Audio has {audio.ndim} channels, expected mono (1 channel)"

    # Duration
    duration = len(audio) / sr
    if not (0.5 <= duration <= 3.0):
        return 'WARNING', f"Duration {duration:.2f}s, expected 0.5-3.0s"

    # RMS energy
    rms = np.sqrt(np.mean(audio**2))
    if rms < 0.01:
        return 'WARNING', f"RMS energy {rms:.4f}, too low (possibly silent)"

    # Trimmed duration
    trimmed_audio, _ = trim_silence(audio, sr)
    trimmed_duration = len(trimmed_audio) / sr
    if trimmed_duration < 0.3:
        return 'WARNING', f"Trimmed duration {trimmed_duration:.2f}s, too short after silence removal"

    return 'OK', "All checks passed"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate a speaker verification dataset')
    parser.add_argument('--dataset', default='custom_dataset', help='Dataset name from config/dataset_config.json')
    args = parser.parse_args()
    cfg = load_dataset_config(args.dataset)
    dataset_path = cfg.get('data_path', f'data/{args.dataset}')
    report, summary = validate_custom_dataset(dataset_path)

    # Print to console
    for line in report:
        print(line)

    # Save to file
    report_path = os.path.join(dataset_path, "validation_report.txt")
    with open(report_path, 'w') as f:
        for line in report:
            f.write(line + '\n')

    print(f"\nReport saved to {report_path}")