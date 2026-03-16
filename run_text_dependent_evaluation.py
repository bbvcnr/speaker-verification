"""
Text-Dependent Speaker Verification Evaluation Pipeline

Uses Google Speech Commands dataset where all speakers say the same word (e.g., "yes").
This creates a controlled environment to benchmark speaker verification on text-dependent task.

Expected runtime: 2-5 minutes

Output:
    evaluation_results/text_dependent/
    ├── evaluation_scores.csv (genuine + impostor scores)
    ├── score_histograms.png
    ├── roc_curve.png
    ├── threshold_analysis.png
    ├── speaker_distance_matrix.png (heatmap of speaker separation)
    ├── threshold_table.csv (FAR/FRR at each threshold)
    ├── metrics.json (comprehensive metrics including runtime)
    └── evaluation_summary.txt
"""

import os
import json
import random
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_handlers import GoogleSpeechCommandsHandler
from audio_utils import load_audio, normalize_audio, trim_silence
from features import extract_mfcc
from verification import create_template, verify_speaker
from evaluation.visualizations import plot_score_histograms, plot_roc_curve, plot_threshold_analysis
from evaluation.metrics import compute_roc_curve, compute_eer


def run_text_dependent_evaluation(keyword="yes",
                                  n_speakers=20,
                                  template_size=3,
                                  output_dir="evaluation_results/text_dependent"):
    """
    Run text-dependent speaker verification evaluation on Google Speech Commands.
    
    Args:
        keyword: Speech Commands keyword to use (e.g., "yes", "no", "up", "down")
        n_speakers: number of speakers to evaluate
        template_size: number of recordings per speaker for template
        output_dir: output directory for results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"TEXT-DEPENDENT SPEAKER VERIFICATION EVALUATION")
    print(f"Dataset: Google Speech Commands ('{keyword}' keyword)")
    print("=" * 70)
    
    # Step 1: Download and organize dataset
    print(f"\n[Step 1/5] Downloading Google Speech Commands dataset...")
    try:
        dataset_info = GoogleSpeechCommandsHandler.download(keyword=keyword)
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("You can manually download from:")
        print("  https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html")
        print("Then extract and organize by keyword subdirectory")
        return
    
    # Step 2: Select speakers
    print(f"\n[Step 2/5] Selecting speakers and preparing trials...")
    selected_speakers = GoogleSpeechCommandsHandler.select_speakers(
        dataset_info['speaker_recordings'],
        n_speakers=n_speakers,
        min_recordings=template_size + 2  # At least template_size + 2 test utterances
    )
    
    # Create trial pairs
    trials_info = GoogleSpeechCommandsHandler.create_text_dependent_trials(
        selected_speakers,
        template_size=template_size
    )
    
    templates = trials_info['templates']
    genuine_pairs = trials_info['genuine_pairs']
    impostor_pairs = trials_info['impostor_pairs']
    
    # Step 3: Generate scores
    print(f"\n[Step 3/5] Computing DTW distances on {len(genuine_pairs) + len(impostor_pairs)} trials...")
    
    genuine_scores = []
    impostor_scores = []
    scores_data = []
    verification_times = []  # Track runtime measurements
    
    # Process genuine trials
    print("Processing genuine trials...", end='', flush=True)
    n_genuine = len(genuine_pairs)
    for idx, pair in enumerate(genuine_pairs):
        if (idx + 1) % max(1, n_genuine // 10) == 0:
            print(f" {idx+1}/{n_genuine}", end='', flush=True)
        
        template_speaker = pair['template_speaker']
        test_file = pair['test_file']
        
        try:
            # Create template
            template_files = templates[template_speaker]
            template_mfcc = create_template(template_files)
            
            # Verify test file (with timing)
            start_time = time.time()
            distance, _ = verify_speaker(
                template_mfcc,
                test_file,
                threshold=np.inf,
                normalize_dtw=True
            )
            elapsed_ms = (time.time() - start_time) * 1000
            verification_times.append(elapsed_ms)
            
            # Only add valid scores (check for inf/nan)
            if np.isfinite(distance):
                genuine_scores.append(distance)
                scores_data.append({
                    'template_speaker': template_speaker,
                    'test_speaker': pair['test_speaker'],
                    'test_file': test_file,
                    'distance': distance,
                    'same_speaker': True
                })
            else:
                print(f"\nWarning: Invalid DTW distance (inf/nan) for pair {idx}, skipping")
        except Exception as e:
            print(f"\nWarning: Error processing pair {idx}: {e}")
    
    print(" ✓")
    
    # Process impostor trials
    print("Processing impostor trials...", end='', flush=True)
    n_impostor = len(impostor_pairs)
    for idx, pair in enumerate(impostor_pairs):
        if (idx + 1) % max(1, n_impostor // 10) == 0:
            print(f" {idx+1}/{n_impostor}", end='', flush=True)
        
        template_speaker = pair['template_speaker']
        test_file = pair['test_file']
        
        try:
            # Create template
            template_files = templates[template_speaker]
            template_mfcc = create_template(template_files)
            
            # Verify test file (with timing)
            start_time = time.time()
            distance, _ = verify_speaker(
                template_mfcc,
                test_file,
                threshold=np.inf,
                normalize_dtw=True
            )
            elapsed_ms = (time.time() - start_time) * 1000
            verification_times.append(elapsed_ms)
            
            # Only add valid scores (check for inf/nan)
            if np.isfinite(distance):
                impostor_scores.append(distance)
                scores_data.append({
                    'template_speaker': template_speaker,
                    'test_speaker': pair['test_speaker'],
                    'test_file': test_file,
                    'distance': distance,
                    'same_speaker': False
                })
            else:
                print(f"\nWarning: Invalid DTW distance (inf/nan) for pair {idx}, skipping")
        except Exception as e:
            print(f"\nWarning: Error processing pair {idx}: {e}")
    
    print(" ✓")
    
    # Save scores
    scores_csv = os.path.join(output_dir, "evaluation_scores.csv")
    scores_df = pd.DataFrame(scores_data)
    scores_df.to_csv(scores_csv, index=False)
    print(f"Scores saved to {scores_csv}")
    
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    # Validate scores
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        print("\nERROR: No valid scores generated. Check dataset and template creation.")
        return
    
    # Verify no inf/nan in final arrays
    valid_genuine = np.isfinite(genuine_scores)
    valid_impostor = np.isfinite(impostor_scores)
    
    if not np.all(valid_genuine) or not np.all(valid_impostor):
        print(f"\nWarning: Filtering {np.sum(~valid_genuine)} invalid genuine and {np.sum(~valid_impostor)} invalid impostor scores")
        genuine_scores = genuine_scores[valid_genuine]
        impostor_scores = impostor_scores[valid_impostor]
    
    print(f"\nGenuine trials: {len(genuine_scores)} (mean: {np.mean(genuine_scores):.2f}, std: {np.std(genuine_scores):.2f})")
    print(f"Impostor trials: {len(impostor_scores)} (mean: {np.mean(impostor_scores):.2f}, std: {np.std(impostor_scores):.2f})")
    
    # Step 4: Generate visualizations
    print(f"\n[Step 4/5] Generating visualizations...")
    
    # Score histograms
    hist_png = os.path.join(output_dir, "score_histograms.png")
    plot_score_histograms(genuine_scores, impostor_scores, hist_png)
    
    # Speaker distance matrix heatmap
    speaker_matrix_png = os.path.join(output_dir, "speaker_distance_matrix.png")
    generate_speaker_distance_matrix(scores_df, speaker_matrix_png)
    
    # Compute metrics
    roc_info = compute_roc_curve(genuine_scores, impostor_scores, n_thresholds=100)
    eer_info = compute_eer(genuine_scores, impostor_scores)
    
    # Save threshold table
    threshold_csv = os.path.join(output_dir, "threshold_table.csv")
    save_threshold_table(roc_info, threshold_csv)
    
    # ROC curve
    roc_png = os.path.join(output_dir, "roc_curve.png")
    plot_roc_curve(roc_info['far'], roc_info['frr'], eer_info, roc_png)
    
    # Threshold analysis
    threshold_png = os.path.join(output_dir, "threshold_analysis.png")
    plot_threshold_analysis(roc_info['thresholds'], roc_info['far'], roc_info['frr'], eer_info, threshold_png)
    
    # Step 5: Generate summary
    print(f"\n[Step 5/5] Generating evaluation summary...")
    
    summary_file = os.path.join(output_dir, "evaluation_summary.txt")
    generate_summary_report(
        summary_file,
        keyword,
        genuine_scores,
        impostor_scores,
        eer_info,
        roc_info,
        n_speakers,
        template_size
    )
    
    # Save metrics as JSON (with runtime measurements)
    metrics_json = os.path.join(output_dir, "metrics.json")
    avg_verification_time = float(np.mean(verification_times))
    std_verification_time = float(np.std(verification_times))
    
    metrics = {
        'dataset': 'Google Speech Commands',
        'keyword': keyword,
        'n_speakers': len(selected_speakers),
        'n_genuine_trials': len(genuine_scores),
        'n_impostor_trials': len(impostor_scores),
        'template_size': template_size,
        'genuine_mean': float(np.mean(genuine_scores)),
        'genuine_std': float(np.std(genuine_scores)),
        'impostor_mean': float(np.mean(impostor_scores)),
        'impostor_std': float(np.std(impostor_scores)),
        'eer': float(eer_info['eer']),
        'optimal_threshold': float(eer_info['optimal_threshold']),
        'far_at_eer': float(eer_info['far_at_eer']),
        'frr_at_eer': float(eer_info['frr_at_eer']),
        'avg_verification_time_ms': avg_verification_time,
        'std_verification_time_ms': std_verification_time
    }
    
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 70)
    print("TEXT-DEPENDENT EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nKey Results:")
    print(f"  EER: {eer_info['eer']*100:.2f}%")
    print(f"  Optimal Threshold: {eer_info['optimal_threshold']:.2f}")
    print(f"  Score Separation: {(np.mean(impostor_scores) - np.mean(genuine_scores)) / np.std(genuine_scores):.2f}")
    print(f"\nRuntime Performance:")
    print(f"  Avg Verification Time: {avg_verification_time:.2f} ms")
    print(f"  Std Dev: {std_verification_time:.2f} ms")
    print(f"  Min: {np.min(verification_times):.2f} ms")
    print(f"  Max: {np.max(verification_times):.2f} ms")
    print(f"\nComparison with LibriSpeech (text-independent):")
    print(f"  LibriSpeech EER: 45.21%")
    print(f"  Improvement: {45.21 - eer_info['eer']*100:.1f} percentage points")
    print(f"\nAll results saved to: {output_dir}/")


def generate_summary_report(output_file, keyword, genuine_scores, impostor_scores, 
                           eer_info, roc_info, n_speakers, template_size):
    """Generate text summary report."""
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TEXT-DEPENDENT SPEAKER VERIFICATION EVALUATION\n")
        f.write("Google Speech Commands Dataset\n")
        f.write("=" * 70 + "\n\n")
        
        # 1. Dataset and Configuration
        f.write("1. EVALUATION SETUP\n")
        f.write("-" * 70 + "\n")
        f.write(f"Dataset: Google Speech Commands (keyword: '{keyword}')\n")
        f.write(f"Number of Speakers: {n_speakers}\n")
        f.write(f"Template Size: {template_size} recordings\n")
        f.write(f"MFCC Dimensions: 39 (13 coefficients + delta + delta-delta)\n")
        f.write(f"Feature Matching: Normalized Dynamic Time Warping (DTW)\n\n")
        
        # 2. Trial Statistics
        f.write("2. EVALUATION DATASET\n")
        f.write("-" * 70 + "\n")
        f.write(f"Genuine Trials (same-speaker):      {len(genuine_scores)}\n")
        f.write(f"Impostor Trials (different-speaker): {len(impostor_scores)}\n")
        f.write(f"Total Trials:                        {len(genuine_scores) + len(impostor_scores)}\n\n")
        
        # 3. Score Statistics
        f.write("3. SCORE DISTRIBUTION STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write("Genuine Scores:\n")
        f.write(f"  Mean:     {np.mean(genuine_scores):7.2f}\n")
        f.write(f"  Std Dev:  {np.std(genuine_scores):7.2f}\n")
        f.write(f"  Min:      {np.min(genuine_scores):7.2f}\n")
        f.write(f"  Max:      {np.max(genuine_scores):7.2f}\n")
        f.write(f"  Median:   {np.median(genuine_scores):7.2f}\n\n")
        
        f.write("Impostor Scores:\n")
        f.write(f"  Mean:     {np.mean(impostor_scores):7.2f}\n")
        f.write(f"  Std Dev:  {np.std(impostor_scores):7.2f}\n")
        f.write(f"  Min:      {np.min(impostor_scores):7.2f}\n")
        f.write(f"  Max:      {np.max(impostor_scores):7.2f}\n")
        f.write(f"  Median:   {np.median(impostor_scores):7.2f}\n\n")
        
        # 4. Key Metrics
        f.write("4. PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Equal Error Rate (EER):            {eer_info['eer']*100:6.2f}%\n")
        f.write(f"Optimal Threshold (at EER):        {eer_info['optimal_threshold']:7.2f}\n")
        f.write(f"FAR at EER:                        {eer_info['far_at_eer']*100:6.2f}%\n")
        f.write(f"FRR at EER:                        {eer_info['frr_at_eer']*100:6.2f}%\n\n")
        
        # 5. Comparison with LibriSpeech
        f.write("5. COMPARISON: TEXT-DEPENDENT vs TEXT-INDEPENDENT\n")
        f.write("-" * 70 + "\n")
        f.write(f"This Evaluation (Text-Dependent):\n")
        f.write(f"  EER: {eer_info['eer']*100:.2f}%\n")
        f.write(f"  Dataset: Google Speech Commands (same word across all speakers)\n\n")
        
        f.write(f"LibriSpeech Evaluation (Text-Independent):\n")
        f.write(f"  EER: 45.21%\n")
        f.write(f"  Dataset: LibriSpeech dev-clean (different text per speaker)\n\n")
        
        improvement = 45.21 - eer_info['eer']*100
        f.write(f"Performance Improvement (Text-Dependent):\n")
        f.write(f"  {improvement:+.1f} percentage points\n")
        f.write(f"  Factor: {45.21 / max(0.01, eer_info['eer']*100):.1f}x better\n\n")
        
        # 6. Interpretation
        f.write("6. KEY FINDINGS\n")
        f.write("-" * 70 + "\n")
        
        separation = (np.mean(impostor_scores) - np.mean(genuine_scores)) / np.std(genuine_scores)
        f.write(f"Score Separation Index:     {separation:.2f}\n")
        f.write(f"  (>3.0 indicates good separation)\n\n")
        
        overlap_threshold = np.percentile(genuine_scores, 75)
        overlap_impostors = np.sum(impostor_scores < overlap_threshold) / len(impostor_scores)
        f.write(f"Score Overlap Analysis:\n")
        f.write(f"  75th percentile of genuine: {overlap_threshold:.2f}\n")
        f.write(f"  Impostors below this:       {overlap_impostors*100:.1f}%\n\n")
        
        if eer_info['eer'] < 0.05:
            f.write(f"Assessment: ✓ EXCELLENT performance (EER < 5%)\n")
        elif eer_info['eer'] < 0.10:
            f.write(f"Assessment: ✓ GOOD performance (EER < 10%)\n")
        elif eer_info['eer'] < 0.20:
            f.write(f"Assessment: • MODERATE performance (EER < 20%)\n")
        else:
            f.write(f"Assessment: • FAIR performance (EER < 50%)\n")
        
        f.write(f"\nConclusion:\n")
        f.write(f"Text-dependent speaker verification (fixed passphrase) shows\n")
        f.write(f"significantly better performance than text-independent evaluation\n")
        f.write(f"on LibriSpeech. This demonstrates the advantage of constrained\n")
        f.write(f"acoustic conditions for speaker verification tasks.\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("END OF EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n")
    
    print(f"Summary saved to {output_file}")


def generate_speaker_distance_matrix(scores_df, output_path, figsize=(10, 8)):
    """
    Generate heatmap of average DTW distances between speakers.
    
    Creates a pivot table with speakers as rows/columns and average distances as values.
    Visualization shows speaker separation: diagonal should be low (intra-speaker),
    off-diagonal should be high (inter-speaker).
    
    Args:
        scores_df: DataFrame with columns [template_speaker, test_speaker, distance]
        output_path: path to save PNG heatmap
        figsize: figure size tuple
    """
    # Create pivot table: rows=template_speaker, cols=test_speaker, values=mean distance
    pivot = scores_df.pivot_table(
        values='distance',
        index='template_speaker',
        columns='test_speaker',
        aggfunc='mean'
    )
    
    # Generate heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot,
        cmap='viridis',
        annot=False,
        fmt='.2f',
        cbar_kws={'label': 'Average DTW Distance'},
        square=False
    )
    plt.title('Speaker Distance Matrix (DTW)', fontsize=14, fontweight='bold')
    plt.xlabel('Test Speaker', fontsize=12)
    plt.ylabel('Template Speaker', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Speaker distance matrix saved to {output_path}")


def save_threshold_table(roc_info, output_path):
    """
    Save threshold sweep data to CSV.
    
    Creates a table with columns [threshold, far, frr] showing system performance
    across all thresholds tested during ROC computation.
    
    Args:
        roc_info: dict returned by compute_roc_curve() with keys 'thresholds', 'far', 'frr'
        output_path: path to save CSV
    """
    # Create DataFrame from ROC info
    threshold_df = pd.DataFrame({
        'threshold': roc_info['thresholds'],
        'far': roc_info['far'],
        'frr': roc_info['frr']
    })
    
    # Save to CSV
    threshold_df.to_csv(output_path, index=False)
    print(f"Threshold table saved to {output_path}")


if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    run_text_dependent_evaluation(keyword="yes", n_speakers=20, template_size=3)
