"""
Main script for the text-dependent speaker verification prototype.

This script demonstrates the single-pair verification pipeline:
1. Load and preprocess audio
2. Extract MFCC features (with delta and delta-delta)
3. Create a template from one or more recordings
4. Verify a test recording against the template using normalized DTW
5. Apply a threshold-based decision

For batch evaluation across multiple speakers, use:
    python batch_evaluation.py
    
This will generate batch_scores.csv for Phase 2 evaluation (FAR, FRR, EER, ROC).

Usage: Run this script to test the pipeline with sample LibriSpeech files.
"""

import os
from audio_utils import load_audio, normalize_audio, trim_silence
from features import extract_mfcc
from verification import create_template, verify_speaker

def main():
    # Define paths to sample audio files from LibriSpeech dev-clean
    # Note: Adjust these paths based on your actual file locations
    # Example: Using files from speaker 1272 (same speaker for template and test)
    base_path = "data/LibriSpeech/dev-clean"
    template_path = os.path.join(base_path, "1272", "128104", "1272-128104-0000.flac")
    test_path = os.path.join(base_path, "1272", "128104", "1272-128104-0001.flac")  # Same speaker

    # Alternative test with different speaker (uncomment to test rejection)
    # test_path = os.path.join(base_path, "1462", "170138", "1462-170138-0000.flac")  # Different speaker

    # Check if files exist
    if not os.path.exists(template_path):
        print(f"Template file not found: {template_path}")
        return
    if not os.path.exists(test_path):
        print(f"Test file not found: {test_path}")
        return

    print("Starting speaker verification pipeline...")

    # Step 1-3: Load, preprocess, and extract MFCC for template
    print("Creating template from:", template_path)
    template_mfcc = create_template(template_path)
    print(f"Template MFCC shape: {template_mfcc.shape}")

    # Step 4-7: Verify test audio against template
    print("Verifying test audio:", test_path)
    distance, decision = verify_speaker(template_mfcc, test_path)
    print(f"DTW Distance: {distance:.2f}")
    print(f"Decision: {decision}")

    # Optional: Show MFCC shapes for test audio
    audio, sr = load_audio(test_path)
    audio = normalize_audio(audio)
    audio, _ = trim_silence(audio, sr)
    test_mfcc = extract_mfcc(audio, sr)
    print(f"Test MFCC shape: {test_mfcc.shape}")

if __name__ == "__main__":
    main()