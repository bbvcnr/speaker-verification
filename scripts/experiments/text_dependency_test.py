import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path so we can import core modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.audio_utils import load_audio, normalize_audio, trim_silence
from core.features import extract_mfcc
from core.dtw import dtw_distance

def text_dependency_test():
    speaker = "nejra"
    enrollment_phrase = "open"
    wrong_phrase = "down"
    correct_path = f"data/custom_dataset/{speaker}"
    wrong_path = f"data/experiments/{speaker}_wrong"
    template_path = f"templates/{speaker}.npy"
    config_path = "config/custom_threshold.json"
    results_dir = "evaluation_results/experiments"
    output_file = os.path.join(results_dir, "text_dependency_test.txt")

    os.makedirs(results_dir, exist_ok=True)

    # Load template
    if not os.path.exists(template_path):
        print(f"Template {template_path} not found.")
        return
    template = np.load(template_path)

    # Load threshold
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        threshold = config['threshold']
    else:
        threshold = 3.28  # fallback

    # Get correct phrase probes: files not used in template (assuming first 3 for template)
    correct_files = sorted([f for f in os.listdir(correct_path) if f.endswith('.wav')])
    probe_files = correct_files[3:]  # 4th and 5th if exist

    correct_results = []
    correct_distances = []

    for file in probe_files:
        path = os.path.join(correct_path, file)
        audio, _ = load_audio(path, sr=16000)
        audio = normalize_audio(audio)
        audio, _ = trim_silence(audio, sr=16000)
        mfcc = extract_mfcc(audio, sr=16000)
        dist = dtw_distance(template.T, mfcc.T)
        decision = 'ACCEPT' if dist < threshold else 'REJECT'
        correct_results.append((file, dist, decision))
        correct_distances.append(dist)

    # Wrong phrase probes
    if not os.path.exists(wrong_path):
        print(f"Wrong phrase path {wrong_path} not found.")
        return
    wrong_files = sorted([f for f in os.listdir(wrong_path) if f.endswith('.wav')])

    wrong_results = []
    wrong_distances = []

    for file in wrong_files:
        path = os.path.join(wrong_path, file)
        audio, _ = load_audio(path, sr=16000)
        audio = normalize_audio(audio)
        audio, _ = trim_silence(audio, sr=16000)
        mfcc = extract_mfcc(audio, sr=16000)
        dist = dtw_distance(template.T, mfcc.T)
        decision = 'ACCEPT' if dist < threshold else 'REJECT'
        wrong_results.append((file, dist, decision))
        wrong_distances.append(dist)

    # Compute stats
    correct_accepted = sum(1 for _, _, d in correct_results if d == 'ACCEPT')
    correct_total = len(correct_results)
    wrong_rejected = sum(1 for _, _, d in wrong_results if d == 'REJECT')
    wrong_total = len(wrong_results)

    avg_correct = np.mean(correct_distances) if correct_distances else 0
    avg_wrong = np.mean(wrong_distances) if wrong_distances else 0
    increase = ((avg_wrong - avg_correct) / avg_correct * 100) if avg_correct > 0 else 0

    is_text_dependent = (correct_accepted == correct_total) and (wrong_rejected == wrong_total)

    # Print results
    output = []
    output.append("TEXT DEPENDENCY EXPERIMENT")
    output.append("=" * 50)
    output.append(f"Speaker: {speaker} | Enrollment phrase: \"{enrollment_phrase}\"")
    output.append(f"Threshold: {threshold:.2f}")
    output.append("")
    output.append(f"CORRECT PHRASE (\"{enrollment_phrase}\") probes:")
    for file, dist, decision in correct_results:
        output.append(f"  {speaker}/{file} → distance: {dist:.2f} → {decision} {'✓' if decision == 'ACCEPT' else '✗'}")
    output.append("")
    output.append(f"WRONG PHRASE (\"{wrong_phrase}\") probes:")
    for file, dist, decision in wrong_results:
        output.append(f"  {speaker}_wrong/{file} → distance: {dist:.2f} → {decision} {'✓' if decision == 'REJECT' else '✗'}")
    output.append("")
    output.append("SUMMARY:")
    output.append(f"Correct phrase accepted: {correct_accepted}/{correct_total} (expected: all)")
    output.append(f"Wrong phrase rejected:   {wrong_rejected}/{wrong_total} (expected: all — proof of text-dependency)")
    output.append("")
    output.append(f"Average distance — correct phrase: {avg_correct:.2f}")
    output.append(f"Average distance — wrong phrase:   {avg_wrong:.2f}")
    output.append(f"Distance increase for wrong phrase: +{increase:.0f}%")
    output.append("")
    output.append(f"Conclusion: The system {'IS' if is_text_dependent else 'IS NOT'} text-dependent for this speaker.")
    output.append("=" * 50)

    # Print to console
    for line in output:
        print(line)

    # Save to file
    with open(output_file, 'w') as f:
        for line in output:
            f.write(line + '\n')

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    text_dependency_test()