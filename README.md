# Text-Dependent Speaker Verification Prototype

A classical signal-processing based speaker verification system using MFCC features and Dynamic Time Warping (DTW), designed for a bachelor's thesis. This implementation prioritizes clarity and modularity over optimization.

## Overview

This project implements a text-dependent speaker verification pipeline that verifies whether a speaker matches a previously stored template using:
- **MFCC features** with delta and delta-delta coefficients (39 total dimensions)
- **Normalized Dynamic Time Warping (DTW)** for sequence similarity
- **Multi-recording templates** for robust speaker models

## Dataset

Uses the **LibriSpeech dataset** (dev-clean subset):
- 84 unique speakers
- Multiple recordings per speaker (utterances from audiobooks)
- Each recording: `.flac` format, 16 kHz sample rate

## Project Structure

```
speaker-verification/
├── audio_utils.py              # Audio loading, normalization, silence trimming
├── features.py                 # MFCC extraction with delta coefficients
├── dtw.py                      # Dynamic Time Warping distance computation
├── verification.py             # Template creation, verification logic
├── main.py                     # Single-pair verification demo
├── batch_evaluation.py         # Batch processing for multiple speakers
├── batch_scores.csv            # Generated scores from batch evaluation
└── README.md                   # This file
```

## Phase 1 Features:

#### 1. **Normalized DTW Distance**
- Divides distance by warping path length (~n + m) for length-independent comparisons
- Makes distances comparable across utterances of different durations
- Optional parameter: `normalize=True` in `dtw_distance()`

#### 2. **Delta & Delta-Delta MFCC Features**
- **Static MFCCs** (13 coefficients): Spectral characteristics
- **Delta MFCCs** (13 coefficients): Velocity (rate of change)
- **Delta-Delta MFCCs** (13 coefficients): Acceleration
- **Total: 39 features** per time frame
- Captures temporal dynamics for better speaker discrimination

#### 3. **Multi-Template Support**
- `create_template()` accepts single path or list of paths
- Extracts MFCCs from multiple recordings
- Pads shorter sequences to max length
- Averages to create robust speaker representation
- Reduces noise from single recordings

#### 4. **Batch Evaluation Script**
- Processes multiple speakers from LibriSpeech
- Creates templates from first half of recordings per speaker
- Tests against second half (same-speaker pairs)
- Generates `batch_scores.csv` with normalized DTW distances
- Computes basic statistics (mean, std, min, max)

## Usage

### Single-Pair Verification (Demo)

```bash
python main.py
```

**Output:**
```
Template MFCC shape: (39, 154)
Test MFCC shape: (39, 120)
DTW Distance: 19033.92
Decision: REJECT
```

### Batch Evaluation (For Thesis Data)

```bash
python batch_evaluation.py
```

Processes 3 speakers by default (remove `[:3]` to process all 84 speakers).

**Output:**
```
Processing speaker 1272: 36 template, 37 test recordings
Processing speaker 1462: 47 template, 47 test recordings
Processing speaker 1673: 21 template, 21 test recordings
Batch evaluation complete. Results saved to batch_scores.csv
Total pairs evaluated: 105
Distance stats - Mean: 89.88, Std: 17.27, Min: 53.21, Max: 138.68
```

### Generated Data: `batch_scores.csv`

```csv
speaker,test_file,distance,same_speaker
1272,1272-135031-0021.flac,91.07,True
1272,1272-135031-0022.flac,67.05,True
1272,1272-141231-0000.flac,69.99,True
```

## Module Documentation

### `audio_utils.py`

Functions for audio I/O and preprocessing:

```python
load_audio(file_path, sr=None)
    Load audio file, optionally resample to target sample rate
    Returns: (audio_array, sample_rate)

normalize_audio(audio)
    Normalize audio to [-1, 1] range for consistent processing
    Returns: normalized_audio

trim_silence(audio, sr, top_db=20)
    Remove leading/trailing silence
    Returns: (trimmed_audio, (start, end) indices)
```

### `features.py`

MFCC feature extraction with temporal dynamics:

```python
extract_mfcc(audio, sr, n_mfcc=13, hop_length=512, n_fft=2048, include_deltas=True)
    Extract MFCC features with optional delta coefficients
    - When include_deltas=True (default): returns (39, time_frames)
      - Rows 0-12: static MFCCs
      - Rows 13-25: delta (velocity)
      - Rows 26-38: delta-delta (acceleration)
    - When include_deltas=False: returns (13, time_frames)
    Returns: MFCC matrix of shape (n_features, time_frames)
```

**Why MFCC + Deltas?**
- MFCCs mimic human hearing (mel scale)
- Delta coefficients capture temporal dynamics
- Standard in speech recognition and speaker verification

### `dtw.py`

Dynamic Time Warping for sequence alignment:

```python
dtw_distance(seq1, seq2, normalize=False)
    Compute DTW distance between two sequences
    - Allows non-linear time warping (handles speaking rate variations)
    - normalize=True: divides by path length for fair comparison
    Returns: distance (scalar, lower = more similar)
```

**Why DTW?**
- Handles temporal variations in speech
- Robust to speaking rate differences
- Perfect for comparing utterances of different lengths
- Classical, well-understood method for speech processing

### `verification.py`

Speaker verification logic:

```python
create_template(audio_paths, sr=16000)
    Create speaker template from one or more recordings
    - Accepts single path (str) or list of paths
    - Averages padded MFCCs across recordings
    Returns: averaged_mfcc_matrix

verify_speaker(template_mfcc, test_audio_path, sr=16000, threshold=1000, normalize_dtw=False)
    Verify test audio against template
    - Computes DTW distance
    - Applies threshold-based decision
    Returns: (distance, decision)
```

### `batch_evaluation.py`

Batch processing pipeline:

```python
batch_evaluate_speakers(base_path="data/LibriSpeech/dev-clean", output_file="batch_scores.csv")
    Process multiple speakers, generate similarity scores
    - Splits recordings: first half for template, second half for testing
    - Creates multi-templates
    - Computes normalized DTW for all test pairs
    Saves: batch_scores.csv with (speaker, test_file, distance, same_speaker)
```

## Current Results (Phase 1)

### Batch Evaluation Statistics (105 same-speaker pairs)

| Metric | Value |
|--------|-------|
| Mean Distance | 89.88 |
| Std Deviation | 17.27 |
| Minimum | 53.21 |
| Maximum | 138.68 |

**Interpretation:**
- Same-speaker pairs cluster around mean ≈ 90
- Current threshold (1000) is very conservative
- Need Phase 2 evaluation to optimize threshold

## Why This Approach?

### Classical Signal Processing (No Deep Learning)
- Interpretable pipeline at each stage
- Clear signal processing principles
- No dependence on large labeled datasets for training

### MFCC + DTW
- **Established method** in speech processing
- **Robust** to noise and variations
- **Efficient** for real-time applications
- **Well-documented** in literature

### Multi-Template Averaging
- **Reduces noise** from single recordings
- **Captures speaker variability** across utterances
- **Improves reliability** over time

### Normalized DTW
- **Length-independent** comparisons
- **Fair evaluation** across different utterance lengths
- **Better threshold generalization** across speakers

## Phase 2 Planned Components

### Evaluation Metrics (evaluation.py)

**Primary Metrics:**

1. **FAR (False Acceptance Rate)**
   - Percentage of impostor pairs accepted as genuine
   - Formula: FAR(τ) = #false_accepts / #impostor_pairs
   - Common impostor pairs: different speakers tested together

2. **FRR (False Rejection Rate)**
   - Percentage of genuine pairs rejected
   - Formula: FRR(τ) = #false_rejects / #genuine_pairs
   - Genuine pairs: same speaker test pairs (already in batch_scores.csv)

3. **ROC Curve (Receiver Operating Characteristic)**
   - Plot: FRR vs FAR across all thresholds
   - Shows performance trade-off
   - Visual assessment of system quality

4. **EER (Equal Error Rate)**
   - Threshold where FAR ≈ FRR
   - Single-number metric for comparison
   - Lower EER = better system

**Implementation Plan:**

```python
# evaluation.py module structure:

compute_far_frr(genuine_scores, impostor_scores, threshold)
    # Given lists of same-speaker and different-speaker distances
    # Compute FAR and FRR at specific threshold
    
compute_roc_curve(genuine_scores, impostor_scores)
    # Generate FAR/FRR across all thresholds
    # Return arrays for plotting
    
compute_eer(genuine_scores, impostor_scores)
    # Find threshold where FAR ≈ FRR
    # Return EER value and optimal threshold
    
plot_roc_curve(far_array, frr_array, output_path)
    # Visualize ROC curve using matplotlib
```

### Impostor Pair Generation

**Current state:** `batch_scores.csv` contains 105 genuine (same-speaker) pairs

**Need for Phase 2:** Impostor (different-speaker) pairs

**Approach:**
1. Extend batch evaluation to test speakers against other speakers' templates
2. Or: generate on-demand during evaluation (test speaker A's recordings against speaker B's template)
3. Goal: ~1000+ impostor pairs for robust statistics

### Threshold Analysis

**Plan:**
- Sweep thresholds from min to max of batch_scores.csv distances
- For each threshold: compute FAR, FRR
- Plot score distributions (genuine vs impostor)
- Identify EER point
- Create threshold recommendation for thesis

## Expected Phase 2 Outputs

1. **evaluation.py** module with FAR/FRR/EER/ROC functions
2. **evaluation_results.csv** with metrics across thresholds
3. **roc_curve.png** visualization
4. **score_distributions.png** showing genuine/impostor separation
5. **threshold_analysis.txt** with recommendations

## Requirements

```
librosa>=0.9.2
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
soundfile>=0.11.0
```

Install with:
```bash
pip install -r requirements.txt
```

Or activate the provided venv:
```bash
source venv/Scripts/activate  # Linux/macOS
venv\Scripts\activate         # Windows
```

## Next Steps

1. **Phase 2 (Scheduled):** Implement evaluation framework
   - FAR, FRR, EER computation
   - ROC curve generation
   - Threshold optimization
   - Score distribution analysis

2. **Phase 3 (Optional):** Advanced features
   - Sakoe-Chiba band constraint for DTW
   - Speaker-adaptive thresholds
   - Noise robustness testing
   - Comparative baselines

## References

- **LibriSpeech Dataset:** http://www.openslr.org/12
- **MFCC Features:** Davis & Mermelstein (1980)
- **DTW:** Sakoe & Chiba (1978)
- **Speaker Verification:** Reynolds (2002), NIST SRE evaluations


## License

Educational use
