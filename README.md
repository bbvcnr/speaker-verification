# Text-Dependent Speaker Verification System

A classical DSP-based speaker verification system optimized for text-dependent audio. The project separates reusable audio and verification logic into `core/` and dataset/evaluation orchestration into `scripts/`.

## Overview

This repository implements a **text-dependent speaker verification pipeline** using:
- **MFCC features** with delta and delta-delta coefficients
- **Cepstral Mean and Variance Normalization (CMVN)**
- **Dynamic Time Warping (DTW)** for time-sequence similarity
- **Threshold-based verification** with ROC curves, EER, TAR, FAR, and advanced diagnostics

## Supported datasets

The system supports three datasets configured in `config/dataset_config.json`:

- `custom_dataset`: open custom dataset under `data/custom_dataset/`
- `heysnips`: Hey Snips dataset under `data/heysnips/`
- `speech_commands_subset`: Speech Commands subset under `data/speech_commands_subset/`

## Project structure

- `core/`: reusable audio preprocessing and verification modules
- `scripts/dataset/`: dataset preparation and template-building scripts
- `scripts/evaluation/`: evaluation, threshold tuning, and metrics scripts
- `scripts/tools/`: auxiliary utilities such as recording helpers
- `evaluation/`: metric definitions and plotting helpers
- `config/`: dataset configuration and threshold files
- `evaluation_results/`: generated evaluation outputs

## Layout

```
speaker-verification/
├── core/
│   ├── __init__.py
│   ├── audio_utils.py
│   ├── dtw.py
│   ├── features.py
│   └── verification.py
├── scripts/
│   ├── dataset/
│   │   ├── build_templates.py
│   │   ├── extract_speech_commands_subset.py
│   │   ├── validate_dataset.py
│   ├── evaluation/
│   │   ├── advanced_metrics.py
│   │   ├── evaluate_dataset.py
│   │   ├── test_end_to_end.py
│   │   ├── tune_threshold.py
│   ├── tools/
│   │   └── recording.py
│   └── experiments/
│       └── text_dependency_test.py
├── evaluation/
│   ├── metrics.py
│   └── visualizations.py
├── config/
│   ├── dataset_config.json
│   ├── custom_threshold.json
│   └── speech_commands_subset_threshold.json
├── evaluation_results/
│   ├── custom_dataset/
│   ├── heysnips/
│   └── speech_commands_subset/
└── README.md
```

## Installation

### Requirements
- Python 3.8+
- `librosa`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `pandas`, `soundfile`

### Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1
pip install librosa numpy scipy matplotlib seaborn pandas soundfile
```

### Quick Start

Validate and evaluate the custom dataset:

```bash
python scripts/dataset/validate_dataset.py --dataset custom_dataset
python scripts/dataset/build_templates.py --dataset custom_dataset  
python scripts/evaluation/tune_threshold.py --dataset custom_dataset
python scripts/evaluation/test_end_to_end.py --dataset custom_dataset
python scripts/evaluation/advanced_metrics.py --dataset custom_dataset
```

Or run the complete unified pipeline:

```bash
python scripts/evaluation/evaluate_dataset.py --dataset custom_dataset
```

## Generic dataset workflow

All dataset scripts support the `--dataset` argument and read parameters from `config/dataset_config.json`.

### Validate dataset structure

```bash
python scripts/dataset/validate_dataset.py --dataset custom_dataset
```

### Build enrollment templates

```bash
python scripts/dataset/build_templates.py --dataset custom_dataset
```

### Tune dataset threshold

```bash
python scripts/evaluation/tune_threshold.py --dataset custom_dataset
```

### Run end-to-end verification

```bash
python scripts/evaluation/test_end_to_end.py --dataset custom_dataset
```

### Generate advanced metrics

```bash
python scripts/evaluation/advanced_metrics.py --dataset custom_dataset
```

For the speech commands subset, this also generates the speaker distance matrix and report-ready confusion matrix:

```bash
python scripts/evaluation/advanced_metrics.py --dataset speech_commands_subset
```

### Run the full unified pipeline

```bash
python scripts/evaluation/evaluate_dataset.py --dataset custom_dataset
```

Optional skips:

```bash
python scripts/evaluation/evaluate_dataset.py --dataset custom_dataset --skip-build --skip-tune
```

## Datasets

### `custom_dataset`
- Data path: `data/custom_dataset/`
- Template dir: `templates/custom_dataset/`
- Threshold config: `config/custom_threshold.json`

### `heysnips`
- Data path: `data/heysnips/`
- Template dir: `templates/heysnips/`
- Threshold config: `config/heysnips_threshold.json`

### `speech_commands_subset`
- Data path: `data/speech_commands_subset/`
- Template dir: `templates/speech_commands_subset/`
- Threshold config: `config/speech_commands_subset_threshold.json`
- Extracted with `scripts/dataset/extract_speech_commands_subset.py`

## Results comparison

| Dataset | Speakers | EER | TAR | FAR | Fisher DR | Bhattacharyya |
|--------|----------|-----|-----|-----|-----------|---------------|
| custom_dataset | 8 | 17.0% | 87.8% | 21.4% | 1.93 | 0.49 |
| Hey Snips | 30 | 12.2% | 85.9% | 5.7% | 2.35 | 0.61 |
| Speech Commands subset | 35 | 14.8% | 85.2% | 14.2% | 1.82 | 0.47 |

### Performance Analysis

- **Custom Dataset**: Smallest dataset (8 speakers) with controlled recording conditions. Demonstrates baseline performance on user-recorded audio with consistent microphone/environment.

- **Hey Snips**: Crowdsourced dataset (30 speakers) with moderate variation in recording quality and speaker demographics. Better separation (lower Fisher DR variation) suggests more diverse speaker characteristics.

- **Speech Commands Subset**: Largest dataset (35 speakers) with most variation in recording conditions and speaker types. Slightly higher EER reflects real-world deployment challenges while maintaining acceptable performance.

### Key Insights

1. **Text-Dependent Constraint Works**: All datasets show EER 12-17%, demonstrating fixed-passphrase verification is feasible with classical methods
2. **Scalability**: Performance remains consistent across 8-35 speakers, indicating robust generalization  
3. **Dataset Impact**: Quality and diversity of training data affects performance (custom < Hey Snips ≈ Speech Commands)
4. **Fisher Discriminant Ratio**: Ranges 1.8-2.4σ, showing good class separation and threshold stability

## Notes

- Use `--dataset` consistently for all generic dataset scripts.
- `dataset_config.json` centralizes dataset paths, template directories, and split settings.
- `scripts/evaluation/evaluate_dataset.py` runs the full pipeline for any configured dataset.

## Core modules

### `core/audio_utils.py`
- `load_audio(file_path, sr=16000)`
- `normalize_audio(audio)`
- `trim_silence(audio, sr, top_db=20)`

### `core/features.py`
- `extract_mfcc(audio, sr, n_mfcc=13, include_deltas=True, apply_cmvn=True)`

### `core/dtw.py`
- `dtw_distance(seq1, seq2, normalize=False, use_band=True, band_width=None)` - DTW distance with Sakoe-Chiba band constraint

### `core/verification.py`
- `create_template(audio_paths, sr=16000)` - Create speaker template from multiple utterances
- `verify_speaker(template_mfcc, test_audio_path, sr=16000, threshold=1000, normalize_dtw=False)` - Verify speaker identity

## Literature & Technical Background

The system implements well-established techniques in speaker verification:

- **MFCC Features** (Davis & Mermelstein, 1980): Standard in speech processing
- **Dynamic Time Warping** (Sakoe & Chiba, 1978): Robust sequence comparison  
- **Cepstral Mean & Variance Normalization**: Channel robustness technique
- **Text-Dependent Verification**: Improved performance through vocabulary constraint

## Project Context

This is an implementation of a classical signal-processing approach to speaker verification, suitable for:
- Educational purposes (understanding DSP-based biometrics)
- Embedded systems (CPU-only, no GPU required)  
- Baseline comparisons (against deep learning approaches)
- Controlled experimental environments

The system deliberately avoids deep learning to maintain interpretability and demonstrate that classical approaches remain competitive for constrained problems like text-dependent verification.
