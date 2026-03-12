"""
Batch evaluation script for speaker verification.

This script processes multiple speakers from LibriSpeech dev-clean,
creates multi-recording templates, and computes similarity scores
for same-speaker pairs using normalized DTW.

Output: batch_scores.csv with columns: speaker, test_file, distance, same_speaker
"""

import os
import csv
import numpy as np
from verification import create_template, verify_speaker

def batch_evaluate_speakers(base_path="data/LibriSpeech/dev-clean", output_file="batch_scores.csv"):
    """
    Perform batch evaluation on multiple speakers.
    - base_path: path to LibriSpeech dev-clean
    - output_file: CSV file to save results
    """
    # Get list of speakers (directories in dev-clean)
    speakers = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    speakers.sort()  # For reproducibility

    # Limit to first 3 speakers for demo (remove [:3] to process all)
    speakers = speakers[:3]

    results = []

    for speaker in speakers:
        speaker_path = os.path.join(base_path, speaker)
        # Get all chapter directories for this speaker
        chapters = [d for d in os.listdir(speaker_path) if os.path.isdir(os.path.join(speaker_path, d))]

        all_recordings = []
        for chapter in chapters:
            chapter_path = os.path.join(speaker_path, chapter)
            # Get all .flac files
            flac_files = [f for f in os.listdir(chapter_path) if f.endswith('.flac')]
            flac_paths = [os.path.join(chapter_path, f) for f in flac_files]
            all_recordings.extend(flac_paths)

        if len(all_recordings) < 4:
            print(f"Skipping speaker {speaker}: only {len(all_recordings)} recordings")
            continue

        # Split recordings: first half for template, second half for testing
        split_idx = len(all_recordings) // 2
        template_paths = all_recordings[:split_idx]
        test_paths = all_recordings[split_idx:]

        print(f"Processing speaker {speaker}: {len(template_paths)} template, {len(test_paths)} test recordings")

        # Create multi-recording template
        template_mfcc = create_template(template_paths)

        # Test against each test recording
        for test_path in test_paths:
            distance, _ = verify_speaker(template_mfcc, test_path, normalize_dtw=True)
            results.append({
                'speaker': speaker,
                'test_file': os.path.basename(test_path),
                'distance': round(distance, 2),
                'same_speaker': True  # All pairs are same-speaker in this setup
            })

    # Save to CSV
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['speaker', 'test_file', 'distance', 'same_speaker']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Batch evaluation complete. Results saved to {output_file}")
    print(f"Total pairs evaluated: {len(results)}")

    # Print basic statistics
    distances = [r['distance'] for r in results]
    print(f"Distance stats - Mean: {np.mean(distances):.2f}, Std: {np.std(distances):.2f}, "
          f"Min: {np.min(distances):.2f}, Max: {np.max(distances):.2f}")

if __name__ == "__main__":
    batch_evaluate_speakers()