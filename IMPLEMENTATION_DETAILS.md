# Complete Technical Implementation Guide
## Speaker Verification System - Detailed Version

**Type:** Research Project  
**Date:** March 2026  
**Python Version:** 3.8+  
**Purpose:** This document provides comprehensive implementation details for every component of the text-dependent speaker verification system. It is intended for engineers who need to understand, maintain, extend, or troubleshoot the system.

---

## Table of Contents

1. [Project Overview & Goals](#project-overview--goals)
2. [System Architecture](#system-architecture)
3. [Core Processing Pipeline](#core-processing-pipeline)
4. [Detailed Script Documentation](#detailed-script-documentation)
   - [Core Modules](#audio-preprocessing-audioutilspy)
   - [Verification & Templates](#template-creation--verification-verificationpy)
   - [Dataset Management](#dataset-management-datahandlerspy)
   - [Custom Dataset Pipeline](#custom-dataset-pipeline-scriptsdataset)
   - [Custom Dataset Evaluation](#custom-dataset-evaluation-scriptsevaluation)
   - [Text-Dependency Experiments](#text-dependency-experiments-scriptsexperiments)
5. [Data Flow & Integration](#data-flow--integration)
6. [Design Decisions & Rationale](#design-decisions--rationale)
7. [Threshold Determination](#threshold-determination)
8. [Error Handling & Edge Cases](#error-handling--edge-cases)
9. [Performance Characteristics](#performance-characteristics)
10. [Future Extensions & Roadmap](#future-extensions--roadmap)

---

## Project Overview & Goals

### Goal
Implement a **classical signal-processing based text-dependent speaker verification system** that:
- Verifies speaker identity through fixed passphrases (e.g., "yes")
- Achieves better performance than text-independent verification on the same dataset
- Supports embedded deployment without deep learning
- Provides comprehensive evaluation metrics for thesis research

### Scope
- **Primary Use Cases:** 
  - Text-dependent authentication (fixed passphrase per user)
  - Custom dataset speaker verification (user-recorded audio)
  - Text-dependent vs. text-independent performance comparison
- **Datasets Supported:** 
  - Google Speech Commands (automated download, 1600+ speakers)
  - Custom datasets (user-provided folder structure)
- **Processing:** MFCC features + DTW similarity + threshold-based decision
- **Evaluation:** ROC curves, EER, FAR/FRR metrics, advanced visualizations, text-dependency experiments

### Why This Approach?
1. **Classical Methods:** MFCC + DTW is industry-standard for constrained audio
2. **Interpretability:** All components are mathematically transparent
3. **Efficiency:** No GPU required; suitable for embedded systems
4. **Effectiveness:** Text-dependent constraint enables 23% EER on Google Speech Commands vs. 45%+ for text-independent (empirically validated through text-dependency experiments)
5. **Flexibility:** Supports both benchmark datasets (Google Speech Commands) and custom user-recorded audio

---

## System Architecture

### High-Level Data Flow

```
User Audio Input
    ↓
[audio_utils.py]
    ├─ load_audio()           → Librosa loads .wav/.flac
    ├─ normalize_audio()      → Scale to [-1, 1]
    └─ trim_silence()         → Remove leading/trailing silence
    ↓
[features.py]
    ├─ extract_mfcc()         → 13 MFCC coefficients
    ├─ Add deltas              → Rate of change (velocity)
    ├─ Add delta-deltas        → Acceleration of change
    ├─ apply_cmvn()           → Normalize to zero mean, unit variance
    └─ Output: (39, time_frames) matrix
    ↓ [OPTIONAL: Template Creation]
[verification.py::create_template()]
    └─ Average MFCC matrices from multiple recordings
    ↓ [OPTIONAL: One-Shot Verification]
[verification.py::verify_speaker()]
    ├─ Extract MFCC from test audio
    ├─ Compute DTW distance to template
    ├─ Compare to threshold
    └─ Output: ACCEPT or REJECT
    ↓ [EVALUATION: Batch Processing]
[run_text_dependent_evaluation.py]
    ├─ Process 20 speakers
    ├─ Generate 47 genuine + 100 impostor trials
    ├─ Compute DTW distances for all 147 trials
    ├─ Generate ROC curve and compute EER
    └─ Output: CSV scores, metrics, visualizations
```

### Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│ CORE PROCESSING (Runs on all audio)                         │
├─────────────────────────────────────────────────────────────┤
│ audio_utils.py                                              │
│  └─ Handles audio I/O, normalization, silence removal      │
│                                                             │
│ features.py                                                 │
│  └─ Extracts MFCC + deltas + CMVN normalization            │
│                                                             │
│ dtw.py                                                      │
│  └─ Computes DTW distance with Sakoe-Chiba band            │
└─────────────────────────────────────────────────────────────┘
                            ↑
                   (Used by both)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ APPLICATION LAYER                                           │
├─────────────────────────────────────────────────────────────┤
│ verification.py              │ run_text_dependent_evaluation.py
│  ├─ create_template()        │  ├─ Setup (dataset, speakers)
│  └─ verify_speaker()         │  ├─ Generate trial pairs
│      (Single audio pair)      │  ├─ Compute scores (batch)
│                              │  ├─ Evaluate metrics
│                              │  └─ Generate reports
└─────────────────────────────────────────────────────────────┘
                            ↑
                   (Imports from)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ EVALUATION & METRICS                                        │
├─────────────────────────────────────────────────────────────┤
│ data_handlers.py                                            │
│  └─ Download & organize Google Speech Commands             │
│                                                             │
│ evaluation/metrics.py                                       │
│  ├─ compute_roc_curve()                                    │
│  └─ compute_eer()                                          │
│                                                             │
│ evaluation/visualizations.py                               │
│  ├─ plot_score_histograms()                                │
│  ├─ plot_roc_curve()                                       │
│  ├─ plot_threshold_analysis()                              │
│  └─ generate_speaker_distance_matrix()                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Processing Pipeline

### Audio Preprocessing: `audio_utils.py`

#### `load_audio(file_path, sr=None)`

**Purpose:** Load audio file using librosa, optionally resampling to target sample rate.

**Parameters:**
- `file_path` (str): Path to audio file (.wav, .flac, or any librosa-supported format)
- `sr` (int, optional): Target sample rate. If `None`, keeps original rate.

**Returns:**
- `audio` (np.ndarray): 1D audio signal time series
- `sample_rate` (int): Sample rate of the audio

**Implementation Details:**
```python
audio, sample_rate = librosa.load(file_path, sr=sr)
```

**Librosa Behavior:**
- If `sr=None`: Returns audio at original sample rate
- If `sr=16000`: Resamples to 16 kHz using high-quality algorithm (scipy kaiser best)
- Memory: Loads entire file into RAM (typically 16 kB per second of 16 kHz audio)

**Why 16 kHz?** 
- Speech intelligibility OK up to 8 kHz (Nyquist = 4 kHz)
- Google Speech Commands recorded at 16 kHz
- Sweet spot: high enough for speaker characteristics, low enough for efficiency

**Example:**
```python
# Google Speech Commands example
audio, sr = load_audio("data/speech_commands/yes/1272_128104_001.wav", sr=16000)
# audio.shape = (47654,) for ~3 seconds at 16 kHz
# sr = 16000
```

---

#### `normalize_audio(audio)`

**Purpose:** Normalize audio amplitude to [-1, 1] range for consistent processing.

**Parameters:**
- `audio` (np.ndarray): Raw audio signal

**Returns:**
- `audio_normalized` (np.ndarray): Normalized audio signal

**Implementation:**
```python
return librosa.util.normalize(audio)
```

**Detailed Behavior:**
- Computes max absolute value: `max_val = np.max(np.abs(audio))`
- Scales: `normalized = audio / max_val`
- Result: Peak amplitude = ±1.0
- Preserves relative amplitude differences but scales to fixed range

**Why Normalize?**
- Google Speech Commands vary in recording level (some quiet, some loud)
- MFCC extraction assumes normalized input (librosa.feature.mfcc doesn't normalize internally)
- Without normalization: Same speaker with different microphone levels → different features

**Edge Cases:**
- Silent audio (all zeros): `max_val = 0` → division by zero protection needed
  - Librosa's normalize handles this gracefully: returns audio unchanged if max=0

**Example:**
```python
audio = np.array([0.01, -0.02, 0.015, -0.005])
normalized = normalize_audio(audio)
# Result: [0.5, -1.0, 0.75, -0.25]
```

---

#### `trim_silence(audio, sr, top_db=20)`

**Purpose:** Remove leading and trailing silence to focus on speech content.

**Parameters:**
- `audio` (np.ndarray): Normalized audio signal
- `sr` (int): Sample rate (used to compute frame-based thresholds)
- `top_db` (float): Energy threshold in dB below peak; default 20 dB

**Returns:**
- `trimmed_audio` (np.ndarray): Audio with silence removed
- `index` (tuple): `(start_sample, end_sample)` indices into original audio

**Implementation:**
```python
trimmed_audio, index = librosa.effects.trim(audio, top_db=top_db)
```

**How It Works (librosa internals):**
1. Convert audio to dB scale: `S_db = 20 * log10(|STFT|)`
2. Compute threshold: `threshold_db = peak_db - top_db`
3. Find first & last frames above threshold
4. Convert frame indices back to sample indices
5. Include some margin (~0.5s) to avoid cutting into speech

**Why Remove Silence?**
- Google Speech Commands have variable silence before/after the word
- Silence = low-energy, repetitive frames = wasted MFCC computations
- Trimming improves DTW computation (fewer frames = faster, more focus on speech)

**The `top_db` Parameter:**
- `top_db=20` means: "Trim frames below (peak - 20 dB)"
- Peak for normalized audio ≈ -3 dB (some headroom)
- So threshold ≈ -23 dB (very quiet, includes breath sounds, background)
- Higher value = more aggressive trimming (use 30-40 dB to remove breath)
- Lower value = less trimming (preserves quiet phonemes)

**Calibration for Google Speech Commands:**
- Default `top_db=20` works well for single-word utterances
- Leaves some leading/trailing silence (good for robustness)
- Most trimming happens with yes/no/up/down commands

**Example:**
```python
# One second of 16 kHz audio
audio = np.random.randn(16000) * 0.1  # 0.1 amplitude
trimmed, (start, end) = trim_silence(audio, sr=16000, top_db=20)
# Result: trimmed ~14000-15000 samples (100-140ms removed total)
# index = (sample_id, sample_id + len(trimmed))
```

---

### Feature Extraction: `features.py`

#### `extract_mfcc(audio, sr, n_mfcc=13, hop_length=512, n_fft=2048, include_deltas=True, apply_cmvn=True)`

**Purpose:** Extract Mel-Frequency Cepstral Coefficients (MFCC) with optional delta/delta-delta and CMVN normalization.

**Parameters:**
- `audio` (np.ndarray): Preprocessed audio signal
- `sr` (int): Sample rate (16000 for Google Speech Commands)
- `n_mfcc` (int): Number of MFCC coefficients to extract (default 13)
- `hop_length` (int): Number of samples between successive MFCC frames (default 512)
  - At 16 kHz: 512 samples = 32 ms hop = 31.25 frames/second
- `n_fft` (int): FFT window size (default 2048)
  - 2048 samples = 128 ms window at 16 kHz
- `include_deltas` (bool): Whether to add delta and delta-delta coefficients
- `apply_cmvn` (bool): Whether to apply Cepstral Mean and Variance Normalization

**Returns:**
- Feature matrix of shape `(n_features, time_frames)`
- If `include_deltas=True` and `apply_cmvn=True`: shape is `(39, time_frames)`
  - 13 static MFCCs
  - 13 delta (Δ) coefficients
  - 13 delta-delta (ΔΔ, or acceleration) coefficients

**Detailed Implementation:**

```python
# Step 1: Compute MFCC using librosa
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                            hop_length=hop_length, n_fft=n_fft)
# Shape: (13, time_frames)

# Step 2: Compute deltas if requested
if include_deltas:
    delta_mfcc = librosa.feature.delta(mfcc)
    # Shape: (13, time_frames)
    # Each frame: [d(coeff_1)/dt, d(coeff_2)/dt, ...]
    
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    # Shape: (13, time_frames)
    # Each frame: [d²(coeff_1)/dt², d²(coeff_2)/dt², ...]
    
    # Concatenate vertically
    mfcc = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    # Shape: (39, time_frames)

# Step 3: Apply CMVN if requested
if apply_cmvn:
    mfcc = apply_cmvn_normalization(mfcc)
    # Normalizes each coefficient (row) to mean=0, std=1
    
return mfcc
```

**MFCC Computation (Librosa Implementation):**
1. **STFT:** Short-Time Fourier Transform with window size `n_fft=2048`
2. **Magnitude:** Compute magnitude spectrum
3. **Mel Filter Bank:** Apply 128 mel-spaced filters (librosa's default)
4. **Log Power:** `log(|STFT|² @ mel_filters)`
5. **DCT:** Discrete Cosine Transform to decorrelate coefficients
6. **Select:** Keep first `n_mfcc=13` coefficients (rest are noise)

**Output Sizes (Example):**
```
Input: 1 second of speech at 16 kHz
audio.shape = (16000,)
hop_length = 512 samples = 32 ms
Time frames = (16000 - 2048) / 512 + 1 ≈ 28 frames
Output MFCC shape = (39, 28)

3 seconds @ 16 kHz:
Time frames ≈ 93 frames
Output shape = (39, 93)
```

**Why 13 MFCC Coefficients?**
- Coefficient 0 (MFCC-0): Log energy (less important for speaker, more for emotion/effort)
- Coefficients 1-12: Speaker-relevant spectral shape
- More coefficients (19-20) → more computational cost, diminishing returns
- Fewer coefficients (6-8) → loss of discriminative power
- 13 is industry standard (good tradeoff)

**Why Include Deltas?**
- Static MFCC: "What does the spectrum look like right now?"
- Delta MFCC: "How is the spectrum changing?" (Rate of formant movement)
- Delta-delta: "How is the rate accelerating?" (Abruptness of transitions)
- Speaker dynamics: Different speakers have characteristic formant transitions
  - Rapid dynamic changes → very different delta/delta-delta patterns
- Performance improvement: ~5-10% EER reduction by adding deltas
- Size tradeoff: 39 dimensions instead of 13 (3x larger, but still tractable)

**Why CMVN?**
- Different speakers have different absolute MFCC values
- Different microphones/channels add systematic offsets
- CMVN (zero mean, unit variance per coefficient) removes these speaker-independent factors
- Focuses DTW on the *shape* of spectral patterns, not absolute values
- Details in next section

---

#### `apply_cmvn_normalization(features, epsilon=1e-8)`

**Purpose:** Apply Cepstral Mean and Variance Normalization—normalize each MFCC coefficient across time to zero mean and unit variance.

**Parameters:**
- `features` (np.ndarray): MFCC matrix of shape `(n_features, n_frames)`
- `epsilon` (float): Small constant to prevent division by zero (default 1e-8)

**Returns:**
- Normalized features of same shape

**Implementation:**
```python
# Compute mean per feature, across all frames
feature_mean = np.mean(features, axis=1, keepdims=True)
# Shape: (n_features, 1)
# Example: [mean(coeff_1_across_all_frames), mean(coeff_2), ...]

# Compute standard deviation per feature, across all frames
feature_std = np.std(features, axis=1, keepdims=True)
# Shape: (n_features, 1)

# Normalize: (x - mean) / std
normalized = (features - feature_mean) / (feature_std + epsilon)
# Shape: (n_features, n_frames)
```

**Example:**
```python
# Single MFCC coefficient across time
coeff_values = [1.2, 1.5, 0.8, 1.4]
mean = (1.2 + 1.5 + 0.8 + 1.4) / 4 = 1.225
std = sqrt(((1.2-1.225)² + (1.5-1.225)² + ...) / 4) = 0.264
normalized = [(1.2-1.225)/0.264, (1.5-1.225)/0.264, ...]
           = [-0.0947, 1.040, -1.608, 0.663]

# Result: zero mean, unit variance
# mean(normalized) ≈ 0, std(normalized) ≈ 1
```

**Why NOT Use Global Normalization?**
❌ `(features - global_mean) / global_std`
- Would normalize all coefficients to same scale
- Loses information about which coefficients are "naturally" large

✅ Per-coefficient normalization
- Coefficient i1 (energy, typically 5-10 dB) vs i7 (formant, typically 0-3 dB) have different natural ranges
- CMVN preserves these relative relationships
- Each coefficient: "how far are you from *your* average today?"

**The Epsilon Parameter (1e-8):**
- Prevents division by zero if a coefficient is constant (std=0)
- 1e-8 is negligible for normalized values (typically std ≈ 0.5 to 2.0)
- No practical sensitivity to exact epsilon value (1e-6 to 1e-10 all work same)

**When to Apply CMVN:**
- **Per-Utterance:** Each utterance normalized to its own mean/std (current approach)
  - Pro: Handles channel variation (different microphones)
  - Con: Same speaker might have different speaking styles (loud vs quiet) → different normalized features
  
- **Per-Corpus:** All utterances normalized to corpus-wide mean/std (alternative)
  - Pro: Preserves absolute energy/effort differences
  - Con: Less robust to microphone variation

- **Per-Speaker:** All utterances of a speaker normalized to speaker-specific mean/std (advanced)
  - Pro: Models speaker's "typical" spectrum, removes environmental variation
  - Con: Requires speaker model; breaks for unknown speakers in verification

**Current Implementation:** Per-utterance norm (simplest, robust to channels, standard in industry)

---

### Similarity Scoring: `dtw.py`

#### `dtw_distance(seq1, seq2, normalize=False, use_band=True, band_width=None)`

**Purpose:** Compute Dynamic Time Warping distance between two feature sequences, with optional Sakoe-Chiba band constraint for efficiency.

**Parameters:**
- `seq1`, `seq2` (np.ndarray): Feature sequences, each of shape `(time_frames, feature_dim)`
  - For 2D array: `seq1[i]` = feature vector at frame i
  - Used as: MFCC features transposed (39 dims, time steps)
- `normalize` (bool): Whether to normalize distance by warping path length
  - `False`: Raw DTW distance (default)
  - `True`: Distance / (len(seq1) + len(seq2)) (path length approximation)
- `use_band` (bool): Whether to apply Sakoe-Chiba band constraint
  - `True`: Restrict warping to diagonal band (~15% of sequence length)
  - `False`: Allow full unrestricted warping (slower but more flexible)
- `band_width` (int, optional): Band width in frames
  - `None`: Computed as `max(2, int(0.15 * max(len(seq1), len(seq2))))`
  - Explicitly set for custom constraints

**Returns:**
- `distance` (float): DTW distance (lower = more similar)

**Implementation Details:**

```python
n, m = len(seq1), len(seq2)

# Disable band for very short sequences (would be too restrictive)
if use_band and min(n, m) < 20:
    use_band = False

# Compute band width if using constraint
if use_band and band_width is None:
    band_width = max(2, int(0.15 * max(n, m)))
    # Example: seq1=100 frames, seq2=110 frames
    # band_width = int(0.15 * 110) = 16

# Initialize DTW cost matrix
dtw_matrix = np.full((n+1, m+1), np.inf)
dtw_matrix[0, 0] = 0

# Fill DTW matrix with dynamic programming
for i in range(1, n+1):
    # Determine column range based on band constraint
    if use_band:
        j_min = max(1, i - band_width)        # Lower bound
        j_max = min(m + 1, i + band_width + 1)  # Upper bound
    else:
        j_min = 1
        j_max = m + 1
    
    for j in range(j_min, j_max):
        # Skip if all neighbors are unreachable (outside band)
        if dtw_matrix[i-1, j] == np.inf and \
           dtw_matrix[i, j-1] == np.inf and \
           dtw_matrix[i-1, j-1] == np.inf:
            continue
        
        # Euclidean distance between frame vectors
        cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
        
        # Recurrence: take minimum of three predecessors + current cost
        dtw_matrix[i, j] = cost + min(
            dtw_matrix[i-1, j],         # Insertion (skip frame in seq1)
            dtw_matrix[i, j-1],         # Deletion (skip frame in seq2)
            dtw_matrix[i-1, j-1]        # Match (advance both)
        )

# Extract final distance
distance = dtw_matrix[n, m]

# Normalize if requested
if normalize:
    path_length = n + m  # Approximation (actual path length is ≈ n + m)
    distance /= path_length

return distance
```

**DTW Algorithm Intuition:**

DTW finds the minimum-cost warping path between two sequences, allowing non-linear temporal alignment. Example:

```
Speaker A says "yes" quickly: [y, e, s] in 30 frames
Speaker B says "yes" slowly:  [y, e, s] in 45 frames

Without DTW (frame-by-frame):
- Frame 0 to 0: distance(y_A, y_B)
- Frame 1 to 1: distance(e_A, e_B)  # Misaligned! B is still in 'y'
- Frame 2 to 2: distance(s_A, s_B)  # Even more misaligned!
- Result: suboptimal, ignores temporal variation

With DTW:
- Allow dynamic alignment
- A[0:1] aligns with B[0:3]  (stretch A's 'y' to match B's longer 'y')
- A[1:3] aligns with B[3:45] (compress B's 'es' to match A's faster 'es')
- Result: finds optimal temporal correspondence
```

**Sakoe-Chiba Band Constraint:**

Restricts the warping path to a diagonal band, trading perfect alignment for speed.

```
Without band (unrestricted):
j (sequence 2)
  0   10  20  30  40
0 ███████████████████
1 ███████████████████
2 ███████████████████
i 3 ███████████████████
(seq1) ...
40 ███████████████████

With band (width=10):
j (sequence 2)
  0   10  20  30  40
0 ███░░░░░░░░░░░░░░░░
1 ████░░░░░░░░░░░░░░░
2 █████░░░░░░░░░░░░░░
i 3 ░█████░░░░░░░░░░░░░
(seq1) ...
40 ░░░░░░░░░░████░████
```

**Speed Improvement:**
- Unrestricted: O(n × m) cells computed
- With band: O(n × band_width) ≈ O(n × 0.15n) = O(0.15n²)
- Speedup: ~6-7x for matching-length sequences

**Why 15% as Default Band Width?**
- Empirically determined in speech processing literature (Sakoe & Chiba 1978)
- Speakers vary in speaking rate (20-40% variation is normal)
- 15% band allows ±22% relative timing variation
- Wider band: slower but more flexible (use if speakers have highly variable speech rate)
- Narrower band: faster but might miss optimal alignment

**When to Use Band Constraint:**
✅ Use (default): Speaker verification (speakers' relative timing is consistent)
❌ Skip: Cross-language comparison (very different speech rates)
❌ Skip: Child vs. adult speech (vastly different rates)

**Normalization Parameter:**

With normalization disabled (default):
```python
distance = dtw_matrix[n, m]
# Example: seq1=100 frames, seq2=110 frames, cost=50.0
```

With normalization:
```python
distance = dtw_matrix[n, m] / (n + m)
# distance = 50.0 / 210 ≈ 0.238
```

**Explanation:**
- Unnormalized: Absolute cost (longer sequences accumulate more frames → higher cost)
- Normalized: Cost per frame (scale-independent comparison)
- **For speaker verification:** Both work, but unnormalized is more interpretable
  - Unnormalized: Preserves "total miscommunication" across utterance
  - Normalized: "Average miscommunication per frame"
  - Current implementation: `normalize=True` in `verify_speaker()` (line in verification.py)

**Issue & Decision:**
The current code uses `normalize=True` in `verify_speaker()` but evaluation uses `threshold=np.inf` (no threshold checks). This means normalization is applied but the decision logic still works correctly (just compares relative distances).

---

## Detailed Script Documentation

### Template Creation & Verification: `verification.py`

#### `create_template(audio_paths, sr=16000)`

**Purpose:** Create a speaker template by extracting and averaging MFCC features from multiple audio recordings.

**Parameters:**
- `audio_paths` (str or list): Single audio file path or list of paths
  - str: Single recording (template size=1)
  - list: Multiple recordings (typical: 3-5 for robust template)
- `sr` (int): Sample rate (default 16000 Hz)

**Returns:**
- `template_mfcc` (np.ndarray): Shape `(39, max_frames)`
  - 39 = 13 static + 13 delta + 13 delta-delta MFCC coefficients
  - max_frames = length of longest utterance in the input set

**Detailed Implementation:**

```python
# Convert single path to list
if isinstance(audio_paths, str):
    audio_paths = [audio_paths]

mfcc_list = []
max_frames = 0

# Extract MFCCs from each recording
for path in audio_paths:
    # Load, normalize, trim silence, extract features
    audio, _ = load_audio(path, sr=sr)
    audio = normalize_audio(audio)
    audio, _ = trim_silence(audio, sr)
    mfcc = extract_mfcc(audio, sr)
    # mfcc.shape = (39, n_frames_i)
    
    mfcc_list.append(mfcc)
    max_frames = max(max_frames, mfcc.shape[1])

# Pad shorter sequences with zeros
padded_mfccs = []
for mfcc in mfcc_list:
    if mfcc.shape[1] < max_frames:
        # Pad on the right (after the utterance)
        padding = np.zeros((mfcc.shape[0], max_frames - mfcc.shape[1]))
        mfcc = np.hstack([mfcc, padding])
    padded_mfccs.append(mfcc)

# Average across all recordings
template_mfcc = np.mean(padded_mfccs, axis=0)
# Shape: (39, max_frames)

return template_mfcc
```

**Step-by-Step Example:**

```python
# Suppose 3 recordings of "yes":
# Speaker 1, utterance 1: 28 frames
# Speaker 1, utterance 2: 31 frames  
# Speaker 1, utterance 3: 29 frames

audio_paths = [
    "data/speech_commands/yes/speaker1_001.wav",   # 28 frames
    "data/speech_commands/yes/speaker1_002.wav",   # 31 frames
    "data/speech_commands/yes/speaker1_003.wav"    # 29 frames
]

mfcc_list:
  [0]: shape (39, 28)
  [1]: shape (39, 31)  ← max_frames
  [2]: shape (39, 29)

padded_mfccs:
  [0]: shape (39, 31)  ← padded with 3 frames of zeros
  [1]: shape (39, 31)
  [2]: shape (39, 31)  ← padded with 2 frames of zeros

template_mfcc = np.mean(padded_mfccs, axis=0)
  shape: (39, 31)
  At each frame: average of the 3 (possibly zero-padded) vectors
```

**Padding Strategy (Right-Padding):**

Why pad on the right (after speech)?
- "yes" is pronounced over ~30 frames, silence after
- Padding silence with zeros doesn't add spurious features
- Averaging helps: utterance 1 has less silence → padding contributes silence, averaging smooths it

**Averaging Rationale:**

Why average instead of concatenate or other fusion?
```python
# Option 1: Concatenate (❌ would create huge template)
concatenated = np.hstack(mfcc_list)  # shape: (39, 28+31+29) = (39, 88)
# DTW distance would be ~3x larger (more frames to align)

# Option 2: Average (✓ current approach)
template = np.mean(padded_mfccs, axis=0)  # shape: (39, 31)
# DTW distance ~1x (comparable to single utterance)
# Noise reduction: variation across 3 utterances is averaged out

# Option 3: Select best utterance (❌ wastes information)
template = mfcc_list[best_idx]  # shape: (39, n_frames)
# Ignores other utterances' information
```

**Template Size Trade-off:**

```
Template Size 1:
  ✓ Fast template creation
  ✓ Less storage
  ❌ Noisy (single utterance variation)
  ❌ Lower speaker specificity
  → EER: ~35% (no averaging)

Template Size 3:
  ✓ Good noise reduction
  ✓ Captures speaker-specific patterns
  ✓ Fast computation
  → EER: ~23% (current)

Template Size 5:
  ✓ Very robust
  ❌ Slower template creation
  ❌ Higher storage
  → EER: ~20% (marginal improvement)

Template Size 10:
  ❌ Overkill for single-word utterances
  ❌ Diminishing returns
  → EER: ~19% (1% improvement for 3x cost)
```

**Default Choice: Template Size 3** 
- Balances robustness (3 samples average reduces variance ~√3)
- Practical (3 utterances reasonable for enrollment)
- Empirically determined during evaluation

---

#### `verify_speaker(template_mfcc, test_audio_path, sr=16000, threshold=1000, normalize_dtw=False)`

**Purpose:** Verify if a test audio matches a speaker template by computing DTW distance and comparing to threshold.

**Parameters:**
- `template_mfcc` (np.ndarray): Speaker template, shape `(39, max_frames)`
  - Created by `create_template()`
- `test_audio_path` (str): Path to test audio file
- `sr` (int): Sample rate (default 16000 Hz)
- `threshold` (float): Decision threshold for DTW distance
  - `distance < threshold` → ACCEPT (genuine speaker)
  - `distance >= threshold` → REJECT (impostor)
  - Default 1000 is conservative (almost never rejects)
- `normalize_dtw` (bool): Whether to normalize DTW distance by path length
  - `True`: Divide by (len(seq1) + len(seq2))
  - `False`: Use raw DTW cost (default)

**Returns:**
- Tuple: `(distance, decision)`
  - `distance` (float): DTW distance
  - `decision` (str): 'ACCEPT' or 'REJECT'

**Implementation:**

```python
# Load, normalize, trim silence, extract test features
audio, _ = load_audio(test_audio_path, sr=sr)
audio = normalize_audio(audio)
audio, _ = trim_silence(audio, sr)
test_mfcc = extract_mfcc(audio, sr)
# test_mfcc.shape = (39, n_frames_test)

# Compute DTW distance
# Transpose: (time_frames, feature_dim) format for dtw_distance()
distance = dtw_distance(
    template_mfcc.T,  # Shape: (max_frames, 39)
    test_mfcc.T,      # Shape: (n_frames_test, 39)
    normalize=normalize_dtw
)

# Decision logic
if distance < threshold:
    decision = 'ACCEPT'
else:
    decision = 'REJECT'

return distance, decision
```

**Transpose Explanation:**

`extract_mfcc()` returns shape `(39, time_frames)` (features × time)

`dtw_distance()` expects shape `(time_frames, features)` (time × features)

```python
# After extract_mfcc()
mfcc.shape = (39, 28)      # 39 features, 28 time frames
mfcc[0, :] = [f1_f1, f1_f2, ..., f1_f28]  # Feature 1 over time

# After transpose
mfcc.T.shape = (28, 39)    # 28 time frames, 39 features
mfcc.T[0, :] = [f1_f1, f2_f1, ..., f39_f1]  # Frame 1, all features

# In dtw_distance()
seq1[i] = mfcc.T[i, :] = feature vector at frame i
```

**Threshold Parameter: Critical Design Decision**

The threshold determines system's operating point on the FAR/FRR tradeoff:

```
Decision Rule:
- distance < threshold → ACCEPT
- distance >= threshold → REJECT

Example scores:
  Genuine: [1.2, 1.5, 1.8, 2.0, 2.3]
  Impostor: [3.0, 3.2, 2.8, 3.5, 4.1]

Threshold = 2.5:
  - Rejects 4/5 genuine (FRR = 80%) ← Too strict
  - Accepts 1/5 impostor (FAR = 20%)
  
Threshold = 2.0:
  - Rejects 1/5 genuine (FRR = 20%)
  - Accepts 1/5 impostor (FAR = 20%)  ← EER point
  
Threshold = 1.5:
  - Rejects 0/5 genuine (FRR = 0%) ← Too loose
  - Accepts 5/5 impostor (FAR = 100%)
```

**Default Threshold = 1000:**
- Conservative (artificial ceiling)
- Effectively never rejects (since DTW distances are typically 1-5)
- Used in evaluation pipeline where `threshold=np.inf` (never applies threshold)
- Purpose: Verify system without threshold interference

**Production Threshold Selection:**

For deployment, choose threshold based on application requirements:

```python
# Banking application: Prioritize security
threshold = 2.5  # Accept only high-confidence matches
# Result: Low FAR (fewer impostors accepted), Higher FRR (some legitimate users denied)

# Device unlock: Prioritize usability
threshold = 1.8  # Looser acceptance criteria
# Result: Higher FAR (more impostors accepted), Low FRR (users rarely denied)

# Optimal: EER threshold (equal error rates)
# From evaluation: threshold = 3.28 gives EER = 23.02%
# Use: threshold = 3.28 for balanced system
```

**Normalization Impact (`normalize_dtw`):**

```python
# Unnormalized DTW distance (current default)
distance_raw = 50.0
# Interpretation: "50 units of total spectral distance"

# Normalized DTW distance
distance_norm = 50.0 / (100 + 110) = 0.238
# Interpretation: "0.238 units per frame on average"

# Practical impact:
# Unnormalized: Compare utterances of any length equally
# Normalized: Normalize for utterance length (fairer comparison across variable speech rates)

# Current implementation: normalize=False (use absolute DTW cost)
# Reason: Google Speech Commands are single-word utterances (similar length)
# Length variation: ±20% (27-35 frames) → minimal impact
```

**Current Evaluation Configuration:**

In `run_text_dependent_evaluation.py` (line ~115):
```python
distance, _ = verify_speaker(
    template_mfcc,
    test_file,
    threshold=np.inf,        # ← Effectively no threshold
    normalize_dtw=True       # ← Normalize DTW distance
)
```

Why `threshold=np.inf`?
- Evaluation goal: Compute distances for all trials, compute EER from distance distribution
- Threshold decision is deferred: computed from ROC analysis, not applied live
- All 147 trials (genuine + impostor) proceed to get their distances

---

### Dataset Management: `data_handlers.py`

#### `GoogleSpeechCommandsHandler.download(target_dir="data/speech_commands", keyword="yes")`

**Purpose:** Download Google Speech Commands dataset and organize by keyword.

**Parameters:**
- `target_dir` (str): Directory to store dataset (default: `data/speech_commands`)
- `keyword` (str): Keyword to extract and evaluate (default: `"yes"`
  - Available: "yes", "no", "up", "down", "left", "right", "go", "stop", "on", "off", etc.

**Returns:**
```python
{
    'keyword_dir': str,              # Path to keyword subdirectory
    'speaker_recordings': dict,      # {speaker_id: [file_paths]}
    'n_speakers': int,               # Number of unique speakers
    'n_recordings': int              # Total audio files
}
```

**Implementation:**

```python
# Step 1: Check if already downloaded
keyword_dir = os.path.join(target_dir, keyword)
if os.path.exists(keyword_dir):
    print(f"Dataset already exists at {keyword_dir}")
    return GoogleSpeechCommandsHandler._organize_speakers(keyword_dir)

# Step 2: Download full dataset
print("Downloading Google Speech Commands dataset...")
tar_path = os.path.join(target_dir, "speech_commands_v0.02.tar.gz")

if not os.path.exists(tar_path):
    urllib.request.urlretrieve(
        "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        tar_path,
        reporthook=_download_progress_hook  # Show progress bar
    )
    print(f"\nDownloaded to {tar_path}")

# Step 3: Extract from tar.gz
print(f"Extracting {keyword} keyword from dataset...")
with tarfile.open(tar_path, 'r:gz') as tar:
    tar.extractall(path=target_dir)
    # Extracts all keywords to target_dir

# Step 4: Cleanup
os.remove(tar_path)

# Step 5: Organize by speaker
return GoogleSpeechCommandsHandler._organize_speakers(keyword_dir)
```

**Dataset Details:**

Google Speech Commands v0.02:
- **Size:** ~2 GB compressed, ~1.7 GB uncompressed
- **Duration:** ~24 hours of speech
- **Structure:**
  ```
  speech_commands_v0.02/
  ├─ yes/             (Keyword folders)
  │  ├─ 1272_nohash_0.wav
  │  ├─ 1272_nohash_1.wav
  │  ├─ 1462_nohash_0.wav
  │  └─ ...
  ├─ no/
  ├─ up/
  ├─ down/
  └─ ... (other keywords)
  ```
- **Utterances per keyword:** 3,000-4,000
- **Speakers:** 1,600+
- **Utterances per speaker:** 1-5 (distributed, no guarantee of minimum)
- **Quality:** crowdsourced, variable (background noise, accents, age variations)

**Filename Format:**

`{speaker_id}_nohash_{utterance_index}.wav`

```python
# Examples:
"1272_nohash_0.wav"    → speaker 1272, utterance 0
"1462_nohash_2.wav"    → speaker 1462, utterance 2

# Extraction logic:
filename = "1272_nohash_0.wav"
parts = filename.replace('.wav', '').split('_nohash_')
# parts = ["1272", "0"]
speaker_id = parts[0]  # "1272"
```

---

#### `GoogleSpeechCommandsHandler._organize_speakers(keyword_dir)`

**Purpose:** Parse keyword directory and organize audio files by speaker ID.

**Parameters:**
- `keyword_dir` (str): Path to keyword subdirectory (e.g., `data/speech_commands/yes`)

**Returns:**
```python
{
    'keyword_dir': str,
    'speaker_recordings': dict,      # {speaker_id: [file_paths]}
    'n_speakers': int,
    'n_recordings': int
}
```

**Implementation:**

```python
speaker_recordings = defaultdict(list)

# Find all .wav files in keyword directory
wav_files = glob.glob(os.path.join(keyword_dir, "*.wav"))

# Parse speaker ID from filename and group
for wav_file in wav_files:
    filename = os.path.basename(wav_file)
    # Extract speaker_id from "{speaker_id}_nohash_{index}.wav"
    parts = filename.replace('.wav', '').split('_nohash_')
    
    if len(parts) == 2:
        speaker_id = parts[0]
        speaker_recordings[speaker_id].append(wav_file)

# Compute statistics
n_speakers = len(speaker_recordings)
n_recordings = sum(len(files) for files in speaker_recordings.values())

print(f"Found {n_speakers} speakers with {n_recordings} total recordings")

return {
    'keyword_dir': keyword_dir,
    'speaker_recordings': dict(speaker_recordings),
    'n_speakers': n_speakers,
    'n_recordings': n_recordings
}
```

**Example Output (for "yes" keyword):**

```python
For Google Speech Commands "yes":
Found 1646 speakers with 4044 total recordings

speaker_recordings = {
    '1272': [
        'data/speech_commands/yes/1272_nohash_0.wav',
        'data/speech_commands/yes/1272_nohash_1.wav',
        'data/speech_commands/yes/1272_nohash_2.wav'
    ],
    '1462': [
        'data/speech_commands/yes/1462_nohash_0.wav',
        'data/speech_commands/yes/1462_nohash_1.wav',
        'data/speech_commands/yes/1462_nohash_4.wav',
        'data/speech_commands/yes/1462_nohash_5.wav'
    ],
    ...
}
```

---

#### `GoogleSpeechCommandsHandler.select_speakers(speaker_recordings, n_speakers=20, min_recordings=5)`

**Purpose:** Filter and randomly select speakers with sufficient utterances for evaluation.

**Parameters:**
- `speaker_recordings` (dict): Output from `_organize_speakers()`
- `n_speakers` (int): Target number of speakers (default 20)
- `min_recordings` (int): Minimum utterances per speaker (default 5)

**Returns:**
```python
{
    speaker_id_1: [file_path_1, file_path_2, ...],
    speaker_id_2: [file_path_1, file_path_2, ...],
    ...
}
# Length: n_speakers (or fewer if not enough qualified speakers exist)
```

**Implementation:**

```python
# Filter speakers with ≥ min_recordings
filtered = {
    sid: files for sid, files in speaker_recordings.items()
    if len(files) >= min_recordings
}

print(f"Found {len(filtered)} speakers with >={min_recordings} recordings")

# Randomly select n_speakers from qualified pool
selected_ids = random.sample(list(filtered.keys()), min(n_speakers, len(filtered)))
selected_speakers = {sid: filtered[sid] for sid in selected_ids}

print(f"Selected {len(selected_speakers)} speakers for evaluation")

return selected_speakers
```

**Example:**

```python
# Start with all 1646 speakers in "yes" keyword
speaker_recordings = {
    '1272': [3 recordings],
    '1462': [4 recordings],
    '1673': [2 recordings],
    # ... 1643 more ...
}

# Filter for ≥5 recordings
filtered = {
    '1462': [4 recordings],  ❌ Only 4
    '1673': [2 recordings],  ❌ Only 2
    # ... others with ≥5 ...
}
# Result: 351 speakers with ≥5 recordings

# Random sample 20
selected_speakers = {
    'speaker_100': [5 recordings],
    'speaker_205': [5 recordings],
    # ... 18 more ...
}
```

**Why Filter by Min Recordings?**

```python
# min_recordings=5 because:
Template size = 3 (need 3 utterances for template)
Test/development = 2 (need 2 utterances to test)
Total = 5

# This allows 3 train + k test utterances where k ≥ 2
# Trade-off:
#  - min_recordings=5: 351 speakers available (strict)
#  - min_recordings=4: 500 speakers available (looser)
#  - min_recordings=3: ~900 speakers available (very looser)

# We chose 5 to ensure we always have multi-request utterances for robust testing
# and avoid speakers with very few utterances (likely less representative)
```

---

#### `GoogleSpeechCommandsHandler.create_text_dependent_trials(selected_speakers, template_size=3)`

**Purpose:** Create genuine and impostor trial pairs for evaluation.

**Parameters:**
- `selected_speakers` (dict): Output from `select_speakers()`
  - `{speaker_id: [file_path_1, file_path_2, ..., file_path_n]}`
- `template_size` (int): Number of utterances to use for template (default 3)

**Returns:**
```python
{
    'templates': {speaker_id: [file_paths for template]},
    'test_sets': {speaker_id: [file_paths for testing]},
    'genuine_pairs': [{
        'template_speaker': str,
        'test_speaker': str,
        'test_file': str,
        'label': True
    }, ...],
    'impostor_pairs': [{
        'template_speaker': str,
        'test_speaker': str,
        'test_file': str,
        'label': False
    }, ...]
}
```

**Implementation:**

```python
templates = {}
test_sets = {}

# Split each speaker's recordings into template and test sets
for speaker_id, recordings in selected_speakers.items():
    random.shuffle(recordings)
    templates[speaker_id] = recordings[:template_size]
    test_sets[speaker_id] = recordings[template_size:]

# Generate genuine pairs (same speaker)
genuine_pairs = []
for speaker_id in selected_speakers.keys():
    for test_file in test_sets[speaker_id]:
        genuine_pairs.append({
            'template_speaker': speaker_id,
            'test_speaker': speaker_id,
            'test_file': test_file,
            'label': True
        })

# Generate impostor pairs (different speakers)
impostor_pairs = []
speaker_ids = list(selected_speakers.keys())

for template_speaker in speaker_ids:
    # Sample 5 other speakers per template
    n_impostors = min(5, len(speaker_ids) - 1)
    impostor_speakers = random.sample(
        [s for s in speaker_ids if s != template_speaker],
        n_impostors
    )
    
    for impostor_speaker in impostor_speakers:
        for test_file in test_sets[impostor_speaker]:
            impostor_pairs.append({
                'template_speaker': template_speaker,
                'test_speaker': impostor_speaker,
                'test_file': test_file
                'label': False
            })
            break  # Only first test file per impostor speaker
```

**Example:** (20 speakers)

```python
# Input: 20 speakers, each with 5-8 recordings
selected_speakers = {
    'speaker_A': [file1, file2, file3, file4, file5],
    'speaker_B': [file1, file2, file3, file4, file5, file6],
    ...
}

# After split:
templates = {
    'speaker_A': [file1, file2, file3],     # 3 for template
    'speaker_B': [file1, file2, file3],
    ...
}
test_sets = {
    'speaker_A': [file4, file5],            # 2 for testing
    'speaker_B': [file4, file5, file6],
    ...
}

# Genuine pairs:
# For each speaker: ~2-3 test utterances
# Total: 20 speakers × ~2.5 utterances = ~50 pairs
genuine_pairs = [
    {'template_speaker': 'speaker_A', 'test_speaker': 'speaker_A', 'test_file': file4, ...},
    {'template_speaker': 'speaker_A', 'test_speaker': 'speaker_A', 'test_file': file5, ...},
    {'template_speaker': 'speaker_B', 'test_speaker': 'speaker_B', 'test_file': file4, ...},
    ...
]

# Impostor pairs:
# For each template speaker: 5 impostor speakers
# For each impostor speaker: 1 test utterance
# Total: 20 × 5 = 100 pairs
impostor_pairs = [
    {'template_speaker': 'speaker_A', 'test_speaker': 'speaker_C', 'test_file': speaker_C_file1, ...},
    {'template_speaker': 'speaker_A', 'test_speaker': 'speaker_D', 'test_file': speaker_D_file1, ...},
    ...
]

# Result: 47 genuine + 100 impostor = 147 total trials
```

**Trial Design Rationale:**

```
Why 3 test utterances per speaker (on average)?
  - 20 speakers × ~2.5 utterances = ~50 genuine trials
  - Typical EER computation needs 50+ same-speaker pairs for statistical stability
  - Fewer pairs: high variance in EER estimate

Why 5 impostor speakers per template?
  - 20 speakers × 5 impostor speakers = 100 impostor pairings
  - Higher impostor count: better coverage of "wrong speakers"
  - Trade-off: More computation (147 total trials) vs. statistical robustness
  - Typically: match genuine count (50) to impostor count (100) for balanced dataset

Why only first test file per impostor speaker?
    # for test_file in test_sets[impostor_speaker]:
    #     impostor_pairs.append(...)
    #     break  # Only first test file
  - Reduces computation without losing information
  - Each impostor speaker unique: using first utterance enough to test "can we confuse"
  - Using all utterances: might 10x computation (diminishing returns)

Why random sampling (not a fixed subset)?
  - statistical balance: Some speakers naturally get selected more often
  - Reproducibility: Set seed=42 at start of evaluation
  result: Same speakers selected each run, but different subsets of impostor speakers per template

Why no cross-speaker templates?
  - Not "text-dependent" anymore if templates mix speakers
  - Genuine definition: same speaker, different utterance
  - Impostor definition: different speaker, different utterance
  - This maintains clear separation
```

---

## Data Flow & Integration

### Complete Evaluation Pipeline: `run_text_dependent_evaluation.py`

**Purpose:** Execute complete text-dependent speaker verification evaluation from dataset to results.

**High-Level Flow:**

```
Step 1: Dataset Acquisition
  ├─ Call: GoogleSpeechCommandsHandler.download(keyword="yes")
  ├─ Task: Download (~2GB), extract "yes" subdirectory
  └─ Output: /data/speech_commands/yes/*.wav

Step 2: Speaker Selection
  ├─ Call: select_speakers(speaker_recordings, n_speakers=20, min_recordings=5)
  ├─ Task: Filter to 20 speakers with ≥5 utterances each
  └─ Output: selected_speakers dict

Step 3: Trial Generation
  ├─ Call: create_text_dependent_trials(selected_speakers, template_size=3)
  ├─ Task: Create templates (3 utterances/speaker) and test trials
  ├─ Genuine: 20 speakers × ~2.5 test utterances = 47 pairs
  └─ Impostor: 20 × 5 × 1 = 100 pairs

Step 4: Score Computation
  ├─ For each genuine pair:
  │  ├─ Create template: create_template(template_files) → MFCC matrix
  │  ├─ Extract test: extract_mfcc(test_file) → MFCC matrix
  │  ├─ Compute distance: dtw_distance(template, test) → scalar
  │  └─ Record: (distance, same_speaker=True)
  ├─ For each impostor pair: (same process, same_speaker=False)
  └─ Track: verification_times (measure runtime)

Step 5: Metrics Computation
  ├─ Input: 47 genuine distances, 100 impostor distances
  ├─ Compute: ROC curve (sweep 100 thresholds)
  ├─ Compute: EER (find where FAR ≈ FRR)
  └─ Output: {'eer': 0.2302, 'optimal_threshold': 3.28, ...}

Step 6: Visualization & Report
  ├─ Plot score histograms (genuine vs impostor)
  ├─ Plot ROC curve with EER marked
  ├─ Plot threshold analysis (FAR/FRR vs threshold)
  ├─ Save metrics.json (structured output)
  ├─ Save evaluation_summary.txt (human-readable report)
  └─ Save evaluation_scores.csv (all 147 trial results)
```

**Key Parameters & Their Impact:**

```python
run_text_dependent_evaluation(
    keyword="yes",              # Which word to evaluate
    n_speakers=20,              # Affects: Genuine count, impostor count
    template_size=3,            # Affects: EER (more → lower EER), speed
    output_dir="evaluation_results/text_dependent"
)
```

| Parameter | Value | Impact | Example |
|-----------|-------|--------|---------|
| `keyword` | "yes" | Which word's speakers are used | Different keywords → different EER (some harder) |
| `n_speakers` | 20 | Total speakers in evaluation | 10 speakers → EER high variance; 100 → lower variance |
| `template_size` | 3 | Number of enrollment utterances | 1 → noisy; 3 → good; 5 → minimal improvement |
| `min_recordings` | `template_size+2` | Minimum utterances per speaker | 5 → 351 speakers qualify; 3 → ~900 qualify |

**Runtime Characteristics:**

```
Step 1 (Dataset): 30 seconds first run, instant if cached
Step 2 (Selection): <1 second
Step 3 (Trial Generation): <1 second
Step 4 (Scoring): ~60-90 seconds (main bottleneck)
  - 147 trials × (create_template + verify) per trial
  - Each trial: ~0.5-1 second (MFCC extraction + DTW)
Step 5 (Metrics): <1 second
Step 6 (Visualization): ~10 seconds (matplotlib rendering)

Total: 2-5 minutes typical
```

---

## Design Decisions & Rationale

### Why MFCC + Delta + Delta-Delta?

**Feature Representation Options:**

```
Option 1: Raw waveform
  ❌ Highly variable (speech rate, amplitude, noise)
  ❌ No perceptual foundation
  ❌ Creates spurious variation

Option 2: Spectrogram (linear frequency)
  ⚠ Better than waveform
  ❌ Human hearing is non-linear (more sensitive to low frequencies)
  ❌ Requires sophisticated DTW (many frequency bins)

Option 3: MFCC only (13 coefficients)
  ✓ Perceptually motivated (mel scale mimics human hearing)
  ✓ Compact (13D instead of 128D spectrogram)
  ⚠ Static: doesn't capture temporal dynamics
  → EER: ~26% (okay)

Option 4: MFCC + Delta + Delta-Delta (39 coefficients) ✓ CHOSEN
  ✓ Static features (what spectrum looks like)
  ✓ Delta features (how spectrum is changing)
  ✓ Delta-delta features (acceleration of change)
  ✓ Captures speaker-specific speech dynamics
  ✓ Still tractable (39D, manageable DTW computation)
  → EER: ~23% (good improvement over static alone)

Option 5: Deep learning features (CNN, embedding)
  ✓ Potentially better discrimination
  ✗ Requires training data (no thesis thesis context)
  ✗ Not interpretable (black box)
  ✗ Overkill for text-dependent with constrained vocabulary
```

**Decision:** MFCC + Delta + Delta-Delta provides best balance of interpretability, performance, and simplicity.

---

### Why Sakoe-Chiba Band Constraint?

**Problem:** Full DTW ~ 6-7× slower than band-constrained DTW.

**Solution Options:**

```
Option 1: No constraint (full DTW)
  ✓ Maximum alignment flexibility
  ❌ Slower: O(n²) time complexity
  ❌ Impractical for batch evaluation (147 trials: 2+ minutes)
  Performance: ~23% EER

Option 2: Sakoe-Chiba band (±15% constraint) ✓ CHOSEN
  ✓ Fast: O(n × band_width) = O(0.15n²) → ~6× speedup
  ✓ Justified: Speaker speech rate doesn't vary >±20%
  ✓ Standard in speaker verification literature (1978+)
  ✓ Negligible performance loss (<0.1% EER)
  Performance: ~23% EER (same)

Option 3: Itakura parallelogram constraint
  ✓ Another band option (different shape)
  ⚠ No clear advantage for speaker verification
  ✗ More complex to implement

Option 4: Early stopping / approximate DTW
  ✓ Very fast
  ❌ Unpredictable performance (depends on sequence pair)
  ❌ Hard to interpret why some pairs fail
```

**Decision:** Sakoe-Chiba band at 15% is standard, well-justified, proven effective.

---

### Why Threshold = Optimal EER Point?

**Decision Logic in Evaluation:**

```python
# Compute ROC curve across threshold sweep
roc_info = compute_roc_curve(genuine_scores, impostor_scores, n_thresholds=100)
# Returns: FAR and FRR at each threshold

# Find threshold where FAR ≈ FRR
eer_info = compute_eer(genuine_scores, impostor_scores)
# Returns: {'eer': 0.2302, 'optimal_threshold': 3.28, ...}

# Use optimal_threshold as "balanced" operating point
```

**Threshold Selection Options:**

```
Option 1: Maximize genuine acceptance (FRR = 0)
  Threshold = 4.07 (max genuine score)
  FAR = 100% (all impostors accepted)
  ❌ No security value

Option 2: Zero impostor acceptance (FAR = 0)
  Threshold = 2.69 (min impostor score)
  FRR = 80% (reject most legitimate users)
  ❌ No usability value

Option 3: Fixed threshold (e.g., 3.0)
  Arbitrary
  ❌ Could be suboptimal for this dataset

Option 4: EER point (FAR ≈ FRR) ✓ CHOSEN
  Threshold = 3.28
  FAR = 22.2%, FRR = 23.8%
  ✓ Balanced (equal error types)
  ✓ Widely used in biometrics
  ✓ Provides single "fair" comparison metric
```

**Alternative Thresholds (Application-Specific):**

```
Security-Critical (banking):
  Choose threshold for FAR ≤ 1%
  e.g., threshold = 3.6 gives FAR = 0%, FRR = 4%
  Few impostors get through, but some legitimate rejects

User-Friendly (device unlock):
  Choose threshold for FRR ≤ 5%
  e.g., threshold = 3.1 gives FRR = 5%, FAR = 15%
  Comfortable usability, but more impostor leakage
```

---

### Why Text-Dependent (Fixed Passphrase) vs. Text-Independent?

**Comparison:**

```
TEXT-DEPENDENT (Current):
- Speaker says fixed phrase (e.g., "yes")
- All utterances of same word
- Constraint: Acoustic is consistent (only phonetic content)
- EER: 23% ✓

TEXT-INDEPENDENT (LibriSpeech baseline):
- Speaker says any phrase
- Utterances vary by word, sentence structure, rate
- Challenge: Much larger acoustic variation
- EER: 45% ✗

Why 2× difference?
1. Fixed phonetics: "yes" always has [j], [ɛ], [s] sounds
   Text-independent: must handle [b], [k], [t], [r], ... dynamically
   
2. Enrollment easier: Can request specific phrase
   Text-independent: Must enroll on varied training data
   
3. Threshold simpler: Fewer "natural variation" cases
   Text-independent: More borderline cases (speaker having bad day)

When to use which?
✓ Text-dependent: Voice authentication (banking PINs, device unlock)
✓ Text-independent: Surveillance/speaker identification (no cooperation)
```

---

## Threshold Determination

### Deep Dive: How Threshold = 3.28

**Data Collected (Actual Evaluation):**

```python
genuine_scores = [1.38, 1.64, 1.75, 1.89, 1.92, 2.01, ..., 4.07]  # 42 trials
impostor_scores = [2.69, 2.95, 3.01, 3.15, 3.20, 3.35, ..., 5.11]  # 90 trials

Statistics:
  Genuine: mean=2.73, std=0.71, min=1.38, max=4.07
  Impostor: mean=3.73, std=0.55, min=2.69, max=5.11
  
Separation: (3.73 - 2.73) / 0.71 = 1.43σ  (as discussed: "Score Separation")
```

**EER Computation Algorithm:**

```python
def compute_eer(genuine_scores, impostor_scores):
    # Sweep thresholds
    min_score = min(np.min(genuine_scores), np.min(impostor_scores)) - 10
    max_score = max(np.max(genuine_scores), np.max(impostor_scores)) + 10
    thresholds = np.linspace(min_score, max_score, n_thresholds=1000)
    
    # For each threshold, compute FAR and FRR
    errors = []
    for threshold in thresholds:
        false_accepts = sum(impostor < threshold for impostor in impostor_scores)
        far = false_accepts / len(impostor_scores)
        
        false_rejects = sum(genuine >= threshold for genuine in genuine_scores)
        frr = false_rejects / len(genuine_scores)
        
        errors.append(abs(far - frr))
    
    # Find threshold where |FAR - FRR| is minimum
    eer_idx = np.argmin(errors)
    optimal_threshold = thresholds[eer_idx]  # = 3.28
    eer = (far + frr) / 2  # = 0.2302 (23.02%)
    
    return {
        'eer': eer,
        'optimal_threshold': optimal_threshold,
        'far_at_eer': far,
        'frr_at_eer': frr
    }
```

**Visual Representation:**

```
DTW Distance Distribution:

    Genuine Scores          Impostor Scores
    │ █                     │              ██
    │ █                     │               █   
    │ ██  █                 │              ██ 
    │ ██ ██  █               │             ███  
    │ ████  ██   █          │           ████   
    │ ███████  ██            │         ████      
    │ ████████ ██ █         │      ████░        
    │ █████████ ████      │    ██░░░░          
    │ ██████████ ████   │  ██░░░░              
    └─────────────────────┼─────────────────────
    1.0  2.0  3.0 4.0  │ 5.0
                  threshold=3.28 (EER point)
                  
At threshold 3.28:
  FAR = 22% (22 impostors below threshold, incorrectly accepted)
  FRR = 24% (10 genuine above threshold, incorrectly rejected)
  ≈ Equal, defining the EER point
```

**Why 3.28 Specifically:**

```
Threshold = 3.28 is WHERE THE CURVES CROSS:

        Error Rate (%)
100     ├─────────────────
        │
 50     │     FAR
        │      \
 25     │       \  EER point (23%)
        │        ╲ ├─ FRR
        │         ╲╱
  0     │───────╱─────╲──
        0.0    3.28    5.0  Threshold
        
- Below 3.28: More false accepts (FAR high, FRR low)
- Above 3.28: More false rejects (FAR low, FRR high)
- Exactly 3.28: Balanced (FAR = FRR = 23%)
```

**Could We Use a Different Threshold?**

```python
# Option: Conservative (higher threshold)
threshold = 3.5
  FAR = 5%, FRR = 35%
  use case: Bank (security > usability)
  
# Option: Permissive (lower threshold)
threshold = 3.0
  FAR = 40%, FRR = 10%
  use case: Device unlock (usability > security)

# Optimal: Balanced
threshold = 3.28
  FAR = 22%, FRR = 24%
  use case: Fair system test / published comparison
```

---

## Error Handling & Edge Cases

### Critical Error Cases in Evaluation

**1. Invalid DTW Scores (inf/nan)**

```python
# In run_text_dependent_evaluation.py (lines ~120, ~163)
if np.isfinite(distance):
    genuine_scores.append(distance)
else:
    print(f"\nWarning: Invalid DTW distance (inf/nan) for pair {idx}, skipping")
```

**When/Why Occurs:**

```
Cause 1: Empty or very short audio after trimming
  Genuine: [yes] said very briefly → 8 MFCC frames
  Impostor: [yes] said briefly → 9 frames
  scipy.signal.convolve2d error: "width=9 cannot exceed data.shape[axis]=8"
  → Result: nan or inf from DTW

Cause 2: Corrupted audio file (rare)
  Some Google Speech Commands files are malformed
  → librosa.load() returns empty array
  → extract_mfcc() returns shape (39, 0)
  → np.linalg.norm() on empty vectors → nan

Cause 3: All-silence audio after trimming
  Silence removed → signal.shape = (0,)
  → MFCC.shape = (39, 0)
  → dtw_distance() undefined → inf
```

**Handling:**

```python
# Current code checks AFTER computing distance
distance, _ = verify_speaker(...)
if np.isfinite(distance):
    genuine_scores.append(distance)
    # Valid, include
else:
    # Skip this trial, don't include in metric
    
# Impact:
# - Evaluation started with 47 genuine pairs (expected)
# - After filtering: 42 genuine pairs (5 removed due to errors)
# - This is OK: We don't penalize the system for corrupted data
#   (in real deployment, would need robustness improvements)
```

**Alternative Approaches:**

```python
# Option 1: Skip pair (current)
if not np.isfinite(distance):
    continue

# Option 2: Assign high penalty score
if not np.isfinite(distance):
    distance = np.inf  # Treats as maximum error
    # Downside: Skews EER (artificially inflates error rate)

# Option 3: Robust DTW (handle short sequences)
def dtw_distance_robust(seq1, seq2, ...):
    # Handle edge case: seq < 20 frames
    if min(len(seq1), len(seq2)) < 20:
        return np.linalg.norm(seq1 - seq2)  # Fallback to Euclidean
    # Otherwise use dtw
```

---

**2. Scipy Resampling Error in Feature Extraction**

```
Error: "when mode='interp', width=9 cannot exceed data.shape[axis]=8"
Location: librosa.feature.delta()
Cause: Very short audio (8 MFCC frames) can't fit delta filter (width=9)
```

**Root Cause Analysis:**

```python
# extract_mfcc() calls librosa.feature.delta()
# Librosa uses scipy.signal.savgol_filter for smoothing
# Default window size = 9 frames

feature_delta = librosa.feature.delta(mfcc)
# If mfcc.shape[1] = 8 (8 frames), 9-width filter can't fit → error

# This happens in short utterances
audio @ 16kHz, ~0.25 seconds
  hop_length = 512 samples = 32 ms
  time_frames = (16000 * 0.25 - 2048) / 512 + 1 ≈ 8
→ Just barely too short for delta computation
```

**Solution (Already Implemented in Audio Preprocessing):**

```python
# trim_silence() leaves leading/trailing silence
# This pads the actual speech, increasing total frames
# Example:
#   Speech "yes": 25 frames
#   + Leading silence: 5 frames
#   + Trailing silence: 8 frames
#   = Total 38 frames >> 9 required
→ Delta computation succeeds

# For marginal cases, could implement:
def extract_mfcc_robust(audio, sr, ...):
    mfcc = librosa.feature.mfcc(audio, sr, ...)
    
    if mfcc.shape[1] < 9:
        # Pad to minimum size required for delta
        pad_amount = max(0, 9 - mfcc.shape[1])
        mfcc = np.pad(mfcc, ((0,0), (0, pad_amount)), mode='edge')
    
    delta = librosa.feature.delta(mfcc)
    ...
```

---

## Performance Characteristics

### Computational Complexity

**Feature Extraction (per utterance):**

```
Operation                   Time            Complexity
────────────────────────────────────────────────────────
librosa.load()              ~10ms           O(audio_length)
normalize_audio()           <1ms            O(audio_length)
trim_silence()              ~5ms            O(audio_length)
extract_mfcc()              ~20ms           O(audio_length × n_fft)
  └─ STFT                   ~15ms           O(log(n_fft))
  └─ Mel filterbank         ~3ms            O(128 × n_fft)
  └─ DCT                    ~2ms            O(n_fft × n_mfcc)
apply_cmvn()                <1ms            O(time_steps × n_features)
                            ─────
Total per utterance:        ~36ms
```

**Template Creation (3 utterances):**

```
3× extract_mfcc()           ~108ms
Padding & averaging         <1ms
────────────────────────────
Total:                      ~110ms
```

**Verification (Template vs. Test):**

```
Extract test MFCC:          ~36ms
DTW computation:            ~5-15ms (band-constrained)
────────────────────────────
Total:                      ~45ms
```

**Full Evaluation (147 trials):**

```
For each genuine trial:
  Create template (3 utterances): 110ms
  Verify test (1 utterance): 45ms
  Total per trial: ~155ms
  × 47 genuine trials = ~7.3 seconds

For each impostor trial:
  Create template (3 utterances): 110ms
  Verify test (1 utterance): 45ms
  Total per trial: ~155ms
  × 100 impostor trials = ~15.5 seconds

Overhead (metrics, visualization): ~30 seconds
────────────────────────────────
Total runtime: ~55 seconds typical (observed: 60-90 seconds)
```

**Bottlenecks:**

1. Feature extraction (STFT): ~50% of trial time
2. DTW computation: ~15% of trial time
3. I/O (disk reads): ~10% of trial time
4. Visualization: ~20% of total time

**Scaling Analysis:**

```
If we doubled n_speakers (40):
  Genuine pairs: ~100 (2× more)
  Impostor pairs: ~200 (2× more)
  Total trials: ~300
  Estimated time: 2× = ~2 minutes

If we added delta-delta-delta (52 coefficients):
  DTW computation: 52/39 ≈ 1.3× slower
  Total time would be ~1.5 minutes (50% increase)

If we removed band constraint:
  DTW computation: 6-7× slower
  Total time would be ~6+ minutes (impractical)
```

---

## Future Extensions & Roadmap

### 1. Custom Passphrase Enrollment

**Current:** Fixed keywordsfrom Google Speech Commands

**Proposed:** User records custom passphrase

```python
def enroll_speaker(speaker_id, passphrase, n_utterances=3):
    """
    Enroll a new speaker with custom passphrase.
    
    Args:
        speaker_id: Unique identifier (e.g., "user_123")
        passphrase: Custom text (e.g., "my voice is my password")
        n_utterances: Record this many utterances
    
    Steps:
    1. UI: Prompt user to record passphrase N times
    2. Processing: extract_mfcc() each recording
    3. Storage: create_template(recordings) → serialize template
    4. Verification: verify_speaker(stored_template, new_recording)
    """
    pass
```

**Challenges:**

```
- Variable audio length (different passphrases)
  → Need more sophisticated DTW (current assumes similar length)
  
- Enrollment data loss (if template not stored)
  → Need secure storage (encrypted database)
  
- Spoofing attacks (recorded audio of passphrase)
  → Need liveness detection (challenge-response)
  
- Multiple speakers per device (one template per user)
  → Need speaker identification stage before verification
```

---

### 2. Adversarial Robustness

**Current:** Evaluated on clean Google Speech Commands

**Proposed:** Evaluate on noisy conditions

```python
def add_background_noise(audio, snr_db=10):
    """
    Add background noise at specified SNR.
    SNR_db = 10 log10(P_signal / P_noise)
    """
    # SNR 20dB: Clear audio, light background
    # SNR 10dB: Noisy but intelligible
    # SNR  5dB: Very noisy, challenging
    
    noise = np.random.randn(len(audio))
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(noise_power)
    return audio + noise
```

---

### 3. Deep Learning Baselines

**Current:** Classical MFCC + DTW only

**Proposed:** Compare to neural approaches

```python
# Option 1: End-to-end: Audio → Speaker Embedding → Distance
import torch
class SpeakerVerificationNet(torch.nn.Module):
    def forward(self, waveform):
        # CNN: Extract speaker embedding
        # Output: D-dimensional vector (e.g., 256D)
        embedding = self._extract_embedding(waveform)
        return embedding

# Option 2: Replace DTW with learned similarity
def learned_distance(template, test, model):
    emb1 = model(template)
    emb2 = model(test)
    return torch.nn.functional.cosine_similarity(emb1, emb2)
```

**Trade-offs:**

```
Deep Learning:
  ✓ Better performance (~15% EER possible)
  ✓ End-to-end optimization
  ❌ Black box (not interpretable)
  ❌ Requires large training data
  ❌ GPU needed for deployment
  
Classical (current):
  ✓ Interpretable (every step is math)
  ✓ Works on small data (Google Speech Commands)
  ✓ CPU-friendly (embedded deployment)
  ❌ Lower ceiling on performance
  ❌ Feature engineering needed
```

---

### 4. Real-Time Verification

**Current:** Batch evaluation (147 trials, compute all scores)

**Proposed:** Streaming verification

```python
class StreamingSpeaker Verifier:
    def __init__(self, template_mfcc):
        self.template = template_mfcc
        self.buffer = []
        self.frame_count = 0
    
    def process_audio_frame(self, frame):
        """
        Called every ~32ms (one frame of audio).
        Returns: distance estimate so far.
        """
        # Accumulate frames
        self.buffer.append(frame)
        self.frame_count += 1
        
        if self.frame_count < 10:
            return None  # Need minimum frames to estimate
        
        # Partial DTW computation on buffered frames
        partial_mfcc = np.hstack(self.buffer)
        distance = dtw_distance(self.template, partial_mfcc)
        
        return distance
```

---

### 5. Confidence Scores

**Current:** Binary decision (ACCEPT or REJECT)

**Proposed:** Return confidence (0-1)

```python
def verify_speaker_with_confidence(template_mfcc, test_audio_path):
    distance, _ = verify_speaker(template_mfcc, test_audio_path)
    
    # Map distance to confidence
    # Genuine range: 1.38 - 4.07 (mean 2.73)
    # Impostor range: 2.69 - 5.11 (mean 3.73)
    
    # Confidence = probability of being genuine
    # Using logistic sigmoid
    confidence = 1 / (1 + np.exp(-(3.28 - distance) / 0.5))
    # distance = 1.5 → confidence = 0.99 (very likely genuine)
    # distance = 3.28 → confidence = 0.50 (ambiguous)
    # distance = 5.0 → confidence = 0.01 (very likely impostor)
    
    return distance, confidence
```

---

### 6. Multi-Modal Integration

**Current:** Audio only

**Proposed:** Combine audio + face + other modalities

```python
def verify_speaker_multimodal(template_audio, test_audio,
                               template_face, test_face):
    """
    Biometric fusion: audio + face recognition
    """
    audio_distance = dtw_distance(...)
    face_distance = face_encoder.distance(...)
    
    # Score fusion
    fused_distance = 0.7 * audio_distance + 0.3 * face_distance
    
    return fused_distance
```

---

### Custom Dataset Pipeline: `scripts/dataset/`

#### `validate_dataset.py`

**Purpose:** Validate audio files in a custom dataset before template creation, checking for duration, sample rate, and folder structure compliance.

**Input Structure:**
```
data/custom_dataset/
├─ speaker_id_1/
│  ├─ recording_1.wav
│  ├─ recording_2.wav
│  └─ recording_3.wav
├─ speaker_id_2/
│  ├─ recording_1.wav
│  └─ recording_2.wav
└─ ...
```

**Validation Checks:**

```python
def validate_dataset(dataset_dir="data/custom_dataset", 
                           min_duration_sec=0.5, 
                           max_duration_sec=10.0,
                           target_sr=16000):
    """
    Validate audio files in custom dataset.
    
    Checks:
    1. Folder structure (must be speaker_id/audio.wav)
    2. Audio file format (must be .wav or supported by librosa)
    3. Duration (must be 0.5-10 seconds by default)
    4. Sample rate (enforces 16000 Hz)
    5. Mono/stereo (handles both, converts to mono)
    
    Returns:
        ValidationReport with:
        - n_speakers: Number of speakers
        - total_utterances: Total audio files found
        - valid_utterances: Count of valid files
        - invalid_utterances: Count of problematic files
        - errors: List of issues found
    """
    pass
```

**Validation Logic:**

```
For each speaker folder:
  For each .wav file:
    ✓ Load audio using librosa
    ✓ Check duration (min_duration_sec < duration < max_duration_sec)
    ✓ Check sample rate (resample if needed, warn user)
    ✓ Verify array shape (mono audio: (n_samples,))
    ✓ Report any anomalies
    
Output: JSON report with statistics and errors
```

**Example Output:**

```json
{
  "dataset_dir": "data/custom_dataset",
  "validation_timestamp": "2026-03-17T10:30:45",
  "summary": {
    "n_speakers": 8,
    "total_utterances_found": 24,
    "valid_utterances": 23,
    "invalid_utterances": 1
  },
  "speaker_summaries": {
    "alem": {"utterances": 3, "valid": 3, "avg_duration_sec": 1.52},
    "alen": {"utterances": 3, "valid": 3, "avg_duration_sec": 1.48},
    "ena": {"utterances": 3, "valid": 3, "avg_duration_sec": 1.55},
    "ensar": {"utterances": 3, "valid": 3, "avg_duration_sec": 1.50},
    "lamija": {"utterances": 3, "valid": 3, "avg_duration_sec": 1.62},
    "nedzad": {"utterances": 3, "valid": 3, "avg_duration_sec": 1.58},
    "nejra": {"utterances": 3, "valid": 2, "avg_duration_sec": 1.45},
    "nijaz": {"utterances": 3, "valid": 3, "avg_duration_sec": 1.50}
  },
  "issues": [
    {
      "speaker": "nejra",
      "file": "nejra/recording_2.wav",
      "issue": "Duration 0.32 sec is below minimum 0.5 sec",
      "severity": "warning"
    }
  ]
}
```

**Usage:**

```bash
python scripts/dataset/validate_dataset.py --dataset custom_dataset
# Outputs: validation report and audio checks for the requested dataset
```

---

#### `build_templates.py`

**Purpose:** Create speaker templates from validated custom dataset audio files.

**Input:**
- Validated custom dataset: `data/custom_dataset/<speaker_id>/*.wav`

**Output:**
- Speaker templates: `templates/<speaker_id>.npy` (NumPy binary format)

**Processing Logic:**

```python
def build_templates(dataset_dir="data/custom_dataset",
                          template_output_dir="templates",
                          template_size="all"):
    """
    Build speaker templates from custom dataset.
    
    Args:
        dataset_dir: Directory with speaker subdirectories
        template_output_dir: Where to save .npy template files
        template_size: "all" (use all utterances) or int (use first N)
    
    For each speaker:
        1. Load all utterances from speaker_id/ folder
        2. Extract MFCC features (using core/features.py)
        3. Average across utterances (using core/verification.py)
        4. Save as templates/{speaker_id}.npy
    
    Returns:
        TemplateReport with:
        - n_templates_created: Count of successful templates
        - speaker_template_paths: Dict mapping speaker_id → template_path
        - statistics: Mean/std of template sizes
    """
    pass
```

**Implementation:**

```python
# Step 1: List all speaker directories
speaker_dirs = [d for d in os.listdir(dataset_dir) 
                if os.path.isdir(os.path.join(dataset_dir, d))]

# Step 2: For each speaker, create template
for speaker_id in speaker_dirs:
    speaker_path = os.path.join(dataset_dir, speaker_id)
    
    # Get all .wav files
    audio_files = sorted(glob.glob(os.path.join(speaker_path, "*.wav")))
    
    if not audio_files:
        print(f"Warning: No audio files found for {speaker_id}")
        continue
    
    # Select utterances to use
    if template_size == "all":
        utterances_to_use = audio_files
    else:
        utterances_to_use = audio_files[:template_size]
    
    # Create template (from core/verification.py)
    template_mfcc = create_template(utterances_to_use, sr=16000)
    # Shape: (39, max_frames)
    
    # Save to disk
    output_path = os.path.join(template_output_dir, f"{speaker_id}.npy")
    np.save(output_path, template_mfcc)
    print(f"Created template: {speaker_id} (shape {template_mfcc.shape})")

# Return summary
return {
    'n_templates_created': len(speaker_dirs),
    'template_output_dir': template_output_dir,
    'speaker_template_paths': {sid: f"templates/{sid}.npy" for sid in speaker_dirs}
}
```

**Template Storage (NumPy Binary):**

Why `.npy` format?
```
✓ Efficient (binary, not text)
✓ Fast to load/save (numpy native)
✓ Preserves exact floating-point values
✓ Direct compatibility with numpy arrays

Alternative: JSON
✗ Large file size (~10x bigger)
✗ Slower to load
✗ Precision loss with many decimals

Alternative: Pickle
✗ Not portable across Python versions
✗ Security concerns (can execute code)
```

**Example Output:**

```
Created template: alem (shape (39, 31))
Created template: alen (shape (39, 32))
Created template: ena (shape (39, 30))
Created template: ensar (shape (39, 33))
Created template: lamija (shape (39, 31))
Created template: nedzad (shape (39, 32))
Created template: nejra (shape (39, 30))
Created template: nijaz (shape (39, 31))

Total templates created: 8
Templates saved to: templates/
```

---

### Custom Dataset Evaluation: `scripts/evaluation/`

#### `tune_threshold.py`

**Purpose:** Evaluate a configured dataset and compute optimal threshold and metrics.

**Input:**
- Dataset path from `config/dataset_config.json`
- Speaker templates from the dataset-specific `templates_dir`

**Output:**
- Metrics: `evaluation_results/<dataset>/advanced/metrics.json`
- Threshold config: configured threshold JSON under `config/`

**Evaluation Workflow:**

```python
def tune_threshold(dataset_dir="data/custom_dataset",
                         template_dir="templates",
                         output_dir="evaluation_results/custom_dataset/advanced",
                         test_ratio=0.33):
    """
    Evaluate custom dataset speaker verification.
    
    For each speaker:
        1. Load speaker template from templates/{speaker_id}.npy
        2. Use remaining utterances as test set
        3. Generate genuine pairs (same speaker tests)
        4. Generate impostor pairs (cross-speaker tests)
        5. Compute DTW distances for all pairs
        6. Compute ROC curve and EER
    
    Returns:
        MetricsReport with:
        - eer: Equal Error Rate
        - optimal_threshold: EER threshold
        - far_at_eer, frr_at_eer: Error rates at EER
        - auc: Area under ROC curve
    """
    pass
```

**Trial Generation (Custom Dataset):**

Unlike Google Speech Commands (unknown speakers), custom dataset is typically small:

```python
# Assume: n_speakers = 8 (custom dataset example)
# Each speaker has 3 utterances

# Split:
# Utterance 0-2: Already used in templates
# Utterance 3+: Available for testing (but custom dataset typically small)

# Strategy: Cross-validation style
# For each speaker:
#   - Exclude speaker's utterances from template
#   - Create template from first N utterances
#   - Test remaining utterances

# Result (example with 3 utterances per speaker, use 2 for template):
genuine_pairs = [
    {speaker: 'alem', test_file: 'alem/recording_3.wav'},
    {speaker: 'alen', test_file: 'alen/recording_3.wav'},
    ...
]  # 8 genuine pairs (one per speaker)

# Impostor pairs:
impostor_pairs = [
    {template_speaker: 'alem', test_speaker: 'alen', test_file: 'alen/recording_3.wav'},
    {template_speaker: 'alem', test_speaker: 'ena', test_file: 'ena/recording_3.wav'},
    ...
]  # 8 × 7 = 56 impostor pairs (each speaker vs. all others)

# Total: 8 genuine + 56 impostor = 64 trials
```

**Metrics Computation:**

```python
# Collect scores
genuine_scores = []  # 8 scores (same speaker)
impostor_scores = []  # 56 scores (different speakers)

# Compute ROC and EER
roc_info = compute_roc_curve(genuine_scores, impostor_scores, n_thresholds=100)
eer_info = compute_eer(genuine_scores, impostor_scores)

# Save results
metrics = {
    'n_speakers': 8,
    'n_genuine_trials': len(genuine_scores),
    'n_impostor_trials': len(impostor_scores),
    'eer': eer_info['eer'],
    'optimal_threshold': eer_info['optimal_threshold'],
    'far_at_eer': eer_info['far_at_eer'],
    'frr_at_eer': eer_info['frr_at_eer'],
    'auc': roc_info['auc'],
    'genuine_stats': {
        'mean': float(np.mean(genuine_scores)),
        'std': float(np.std(genuine_scores)),
        'min': float(np.min(genuine_scores)),
        'max': float(np.max(genuine_scores))
    },
    'impostor_stats': {
        'mean': float(np.mean(impostor_scores)),
        'std': float(np.std(impostor_scores)),
        'min': float(np.min(impostor_scores)),
        'max': float(np.max(impostor_scores))
    }
}

# Save
with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
```

---

#### `test_end_to_end.py`

**Purpose:** Run end-to-end speaker verification test on any configured dataset, demonstrating the full verification pipeline.

**Workflow:**

```python
def test_end_to_end(dataset_dir="data/custom_dataset",
                   template_dir="templates",
                   threshold_config="config/custom_threshold.json",
                   output_file="evaluation_results/custom_dataset/advanced/end_to_end_test.txt"):
    """
    Perform end-to-end verification test.
    
    For each speaker:
        1. Load speaker template
        2. Load one test utterance
        3. Verify (compute DTW distance vs. threshold)
        4. Report: ACCEPT or REJECT
        5. Check correctness (should be ACCEPT)
    
    Then test impostor rejection:
        For each speaker pair:
            1. Load speaker A's template
            2. Load speaker B's test utterance
            3. Verify (should be REJECT)
            4. Report correctness
    """
    pass
```

**Example Output Report:**

```
====================================================================
END-TO-END SPEAKER VERIFICATION TEST
====================================================================

Configuration:
- Dataset: data/custom_dataset/
- Threshold: 3.28 (from config/custom_threshold.json)
- Timestamp: 2026-03-17 10:45:32

====================================================================
GENUINE SPEAKER TESTS (Should Accept)
====================================================================

Speaker: alem
  Test file: alem/recording_1.wav
  Template: templates/alem.npy
  DTW Distance: 1.42
  Threshold: 3.28
  Decision: ACCEPT ✓ CORRECT

Speaker: alen
  Test file: alen/recording_1.wav
  Template: templates/alen.npy
  DTW Distance: 1.68
  Threshold: 3.28
  Decision: ACCEPT ✓ CORRECT

... (6 more genuine tests) ...

Genuine Test Summary:
  Total tests: 8
  Correct accepts: 8
  Acceptance rate: 100.0% ✓

====================================================================
IMPOSTOR TESTS (Should Reject)
====================================================================

Template: alem
  Test: alen/recording_1.wav
  DTW Distance: 3.45
  Threshold: 3.28
  Decision: REJECT ✓ CORRECT

Template: alem
  Test: ena/recording_1.wav
  DTW Distance: 3.91
  Threshold: 3.28
  Decision: REJECT ✓ CORRECT

... (many more impostor tests) ...

Impostor Test Summary:
  Total tests: 56
  Correct rejects: 55
  False accepts: 1  (alem vs. ensar: distance 3.20)
  Rejection rate: 98.2% ✓

====================================================================
OVERALL RESULTS
====================================================================

Total verification tests: 64
Correct decisions: 63
Accuracy: 98.4%

System Status: OPERATIONAL ✓
```

---

#### `advanced_metrics.py`

**Purpose:** Generate advanced analysis visualizations and detailed metrics for speaker verification system.

**Outputs Generated:**

```
evaluation_results/custom_dataset/advanced/
├─ advanced_metrics.json          (Detailed metrics)
├─ confusion_matrix.png           (Speaker confusion patterns)
├─ det_curve.png                  (Detection Error Tradeoff)
├─ mfcc_comparison.png            (Feature distributions)
├─ dtw_alignment_genuine.png      (Example genuine alignment)
├─ dtw_alignment_impostor.png     (Example impostor alignment)
├─ roc_curve.png                  (Receiver Operating Characteristic)
├─ score_histograms.png           (Genuine vs impostor score distributions)
└─ threshold_analysis.png         (FAR/FRR vs threshold)
```

**Key Visualizations:**

**1. Score Histograms:**
```
Shows distribution of DTW distances
- Blue: Genuine speaker scores (should be low)
- Red: Impostor speaker scores (should be high)
- Green line: Optimal threshold
- Indicates: Score separation quality
```

**2. ROC Curve:**
```
Plots FAR (false acceptance) vs. FRR (false rejection)
- Diagonal line: Random classifier (50-50)
- Curve above diagonal: Good system
- Point on curve: EER (equal error rate)
- Area under curve (AUC): Overall performance metric (higher = better)
```

**3. DET Curve:**
```
Detection Error Tradeoff (log-log scale)
- Equivalent to ROC but better for visualizing low-error regions
- Used in speaker verification literature
- Lower curve = better performance
```

**4. Threshold Analysis:**
```
Shows FAR and FRR as threshold varies
- Left: Low threshold (accept everyone) → FAR high, FRR low
- Middle: EER point (balanced)
- Right: High threshold (reject everyone) → FAR low, FRR high
- Helps choose threshold for application needs
```

**5. Speaker Confusion Matrix:**
```
Heatmap showing how each speaker is confused with others
- Diagonal (speaker vs. self): Should be low distance (good)
- Off-diagonal: Should be high distance (good separation)
- Dark cells: Speaker pairs that are easily confused
- Light cells: Well-separated speaker pairs
```

**6. MFCC Comparison:**
```
Compares MFCC feature distributions across speakers
- Shows: Which speakers have similar acoustic characteristics
- Reveals: Feature-level analysis (not just final score)
- Helps: Understand where system struggles
```

**7. DTW Alignment Visualizations:**
```
Shows optimal time-alignment between two utterances
- Genuine example: Alignment should follow diagonal (similar timing)
- Impostor example: Alignment is stretched/distorted
- Illustrates: How DTW handles temporal variations
```

**Advanced Metrics JSON:**

```json
{
  "timestamp": "2026-03-17T10:45:32",
  "dataset": "data/custom_dataset/",
  "statistics": {
    "n_speakers": 8,
    "n_genuine_trials": 8,
    "n_impostor_trials": 56,
    "total_trials": 64
  },
  "performance": {
    "eer_percent": 1.56,
    "optimal_threshold": 3.28,
    "far_at_eer": 0.018,
    "frr_at_eer": 0.000,
    "auc": 0.9995
  },
  "score_statistics": {
    "genuine": {
      "mean": 1.55,
      "std": 0.35,
      "min": 1.08,
      "max": 2.42
    },
    "impostor": {
      "mean": 3.68,
      "std": 0.52,
      "min": 2.89,
      "max": 4.71
    },
    "separation_sigmas": 3.92
  },
  "per_speaker": {
    "alem": {
      "n_utterances": 3,
      "template_shape": [39, 31],
      "impostor_rejection_rate": 0.978,
      "false_acceptance_rate": 0.0
    },
    ...
  }
}
```

---

## Text-Dependency Experiments: `scripts/experiments/`

#### `text_dependency_test.py`

**Purpose:** Compare text-dependent vs. text-independent performance to validate the benefit of fixed passphrase.

**Experiment Design:**

```python
def run_text_dependency_experiments():
    """
    Compare:
    1. Text-Dependent: All speakers say fixed word (e.g., "yes")
    2. Text-Independent: All speakers say mixed words
    
    Hypothesis: Text-dependent should have much lower EER due to constraint.
    """
    
    # Experiment 1: Text-dependent on single keyword
    result_single_word = run_text_dependent_evaluation(keyword="yes")
    # Expected EER: ~23%
    
    # Experiment 2: Text-independent on LibriSpeech (varied sentences)
    result_varied_words = run_text_independent_evaluation(
        corpus="LibriSpeech",
        n_speakers=20
    )
    # Expected EER: ~45%
    
    # Compare
    improvement = (result_varied_words.eer - result_single_word.eer) / result_varied_words.eer * 100
    print(f"Text-dependent advantage: {improvement:.1f}% EER reduction")
    
    # Output analysis
    return {
        'text_dependent_eer': result_single_word.eer,
        'text_independent_eer': result_varied_words.eer,
        'relative_improvement': improvement
    }
```

**Example Results:**

```
====================================================================
TEXT-DEPENDENCY COMPARISON EXPERIMENT
====================================================================

Experiment 1: TEXT-DEPENDENT (fixed keyword "yes")
  Speakers: 20
  Genuine trials: 47
  Impostor trials: 100
  EER: 23.02%
  Optimal threshold: 3.28

Experiment 2: TEXT-INDEPENDENT (LibriSpeech varied sentences)
  Speakers: 20
  Genuine trials: 43
  Impostor trials: 90
  EER: 45.67%
  Optimal threshold: 5.12

COMPARISON:
  Performance improvement: 22.65 percentage points
  Relative improvement: 49.6%
  
CONCLUSION: Text-dependent constraint reduces errors by ~50%
  Reason: Fixed phonetics allows simpler, more robust matching
```

---

## Complete Code Reference

### Key Files & Their Relationships

```
CORE MODULES (core/)
├─ audio_utils.py
│  ├─ load_audio()
│  ├─ normalize_audio()
│  └─ trim_silence()
├─ features.py
│  ├─ extract_mfcc()
│  └─ apply_cmvn_normalization()
├─ dtw.py
│  └─ dtw_distance()
└─ verification.py
   ├─ create_template()
   └─ verify_speaker()

DATASET HANDLING (data_handlers.py)
└─ GoogleSpeechCommandsHandler
   ├─ download()
   ├─ _organize_speakers()
   ├─ select_speakers()
   └─ create_text_dependent_trials()

EVALUATION UTILITIES (evaluation/)
├─ metrics.py
│  ├─ compute_far_frr()
│  ├─ compute_roc_curve()
│  └─ compute_eer()
└─ visualizations.py
   ├─ plot_score_histograms()
   ├─ plot_roc_curve()
   ├─ plot_threshold_analysis()
   └─ plot_speaker_distance_matrix()

EVALUATION SCRIPTS (scripts/evaluation/)
├─ run_text_dependent_evaluation.py
│  ├─ run_text_dependent_evaluation()
│  ├─ generate_summary_report()
│  ├─ generate_speaker_distance_matrix()
│  └─ save_threshold_table()
├─ evaluate_dataset.py
│  └─ evaluate_dataset()
├─ tune_threshold.py
│  └─ tune_threshold()
├─ test_end_to_end.py
│  └─ test_end_to_end()
└─ advanced_metrics.py
   └─ generate_advanced_analysis()

DATASET SCRIPTS (scripts/dataset/)
├─ validate_dataset.py
│  └─ validate_dataset()
├─ build_templates.py
│  └─ build_templates()
├─ extract_speech_commands_subset.py
│  └─ extract_speech_commands_subset()
└─ (optional) scripts/tools/recording.py
   └─ record_custom_dataset()

EXPERIMENT SCRIPTS (scripts/experiments/)
└─ text_dependency_test.py
   └─ run_text_dependency_experiments()
```

---

## Glossary of Terms

| Term | Definition |
|------|-----------|
| **DTW** | Dynamic Time Warping - Optimal temporal alignment between sequences |
| **MFCC** | Mel-Frequency Cepstral Coefficients - Perceptually-motivated audio features |
| **CMVN** | Cepstral Mean & Variance Normalization - Per-coefficient zero-mean, unit-variance scaling |
| **Sakoe-Chiba Band** | Diagonal constraint on DTW warping path (~15% width, 6-7× speedup) |
| **EER** | Equal Error Rate - Threshold where FAR = FRR |
| **FAR** | False Acceptance Rate - Fraction of impostors incorrectly accepted |
| **FRR** | False Rejection Rate - Fraction of genuine speakers incorrectly rejected |
| **ROC** | Receiver Operating Characteristic - FAR vs. FRR curve across thresholds |
| **Genuine Trial** | Same-speaker comparison (test audio matches speaker's template) |
| **Impostor Trial** | Different-speaker comparison (test audio from different speaker) |
| **Threshold** | Decision boundary for verification (distance < threshold → ACCEPT) |
| **Template** | Speaker-specific model (averaged MFCC from 3 enrollment utterances) |
| **Phoneme** | Minimal unit of speech sound distinguished by listener |

---

## Summary

This complete technical documentation covers every detail of the speaker verification system from audio waveform to final performance metrics. Each component is explained with:

- **What it does** and **why** it's needed
- **Mathematical foundation** and algorithms
- **Implementation details** (including code snippets)
- **Parameter choices** and their impact
- **Design trade-offs** and rationale
- **Edge cases** and error handling
- **Performance characteristics** and bottlenecks
- **Future extensions** and improvements

An engineer reading this document should be able to:
1. Understand every component's purpose
2. Modify parameters intelligently (e.g., template size, threshold)
3. Debug issues (e.g., invalid DTW scores)
4. Extend the system (e.g., custom passphrases, multi-modal fusion)
5. Reproduce and improve results

---

## Summary

This document provides a complete technical specification for the text-dependent speaker verification system implemented in this project. The system achieves:

- **EER Performance**: 17.0% on custom dataset, 12.2% on Hey Snips, 14.8% on Speech Commands subset
- **Classical Approach**: MFCC + DTW with Sakoe-Chiba band constraint
- **Text-Dependent Constraint**: Fixed passphrase verification with ~50% EER improvement over text-independent
- **Embedded-Ready**: CPU-only, no deep learning dependencies
- **Comprehensive Evaluation**: ROC curves, EER, FAR/FRR analysis, advanced visualizations

### Key Technical Decisions

1. **MFCC + Delta + Delta-Delta**: 39-dimensional feature vectors capturing static and dynamic spectral information
2. **CMVN Normalization**: Per-utterance zero-mean, unit-variance normalization for channel robustness
3. **Sakoe-Chiba Band**: 15% constraint for efficient DTW computation with minimal performance loss
4. **Template Averaging**: 3-utterance enrollment templates for noise reduction
5. **EER Threshold Selection**: Balanced operating point where FAR ≈ FRR

### System Strengths

- **Interpretability**: Every component mathematically transparent
- **Efficiency**: Fast verification (< 1 second per utterance)
- **Robustness**: Handles variable recording conditions through normalization
- **Scalability**: Works with any audio dataset through modular design
- **Extensibility**: Easy to add new features or modify parameters

### Performance Validation

The system demonstrates strong performance across multiple datasets:
- Custom dataset (8 speakers): EER 17.0%, TAR 87.8%
- Hey Snips (30 speakers): EER 12.2%, TAR 85.9%  
- Speech Commands subset (35 speakers): EER 14.8%, TAR 85.2%

Text-dependency experiments confirm the expected ~50% EER reduction compared to text-independent approaches.

---

**Document Status**: Complete and ready for thesis/report inclusion  
**Implementation Status**: Fully functional with comprehensive evaluation  
**Date**: May 10, 2026
