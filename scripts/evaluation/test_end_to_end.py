import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path so we can import core and evaluation modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.audio_utils import load_audio, normalize_audio, trim_silence
from data_handlers import load_dataset_config
from core.features import extract_mfcc
from core.dtw import dtw_distance
from core.verification import verify_speaker
from evaluation.metrics import compute_far_frr

def test_end_to_end(dataset_name='custom_dataset'):
    cfg = load_dataset_config(dataset_name)
    dataset_path = cfg.get('data_path', f'data/{dataset_name}')
    templates_dir = cfg.get('templates_dir', 'templates')
    config_path = cfg.get('threshold_config', f'config/{dataset_name}_threshold.json')
    results_dir = os.path.join(cfg.get('results_path', f'evaluation_results/{dataset_name}'), 'advanced')
    enrollment_count = cfg.get('enrollment_count', 3)
    output_file = os.path.join(results_dir, 'end_to_end_test.txt')

    os.makedirs(results_dir, exist_ok=True)

    # Load threshold
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        threshold = config['threshold']
    else:
        threshold = 116.30  # fallback

    # Scan persons and templates
    persons = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    templates = {}
    person_test_files = {}

    for person in persons:
        template_path = os.path.join(templates_dir, f"{person}.npy")
        if os.path.exists(template_path):
            templates[person] = np.load(template_path)
            person_path = os.path.join(dataset_path, person)
            files = sorted([f for f in os.listdir(person_path) if f.endswith('.wav')])
            test_files = files[enrollment_count:]
            if test_files:
                person_test_files[person] = test_files

    # TEST 1 — Genuine verification
    test1_results = []
    test1_passed = 0
    test1_total = 0

    for person in sorted(person_test_files.keys()):
        template = templates[person]
        test_files = person_test_files[person]
        for test_file in test_files:
            test_path = os.path.join(dataset_path, person, test_file)
            distance, decision = verify_speaker(template, test_path, threshold=threshold)
            expected = 'ACCEPT'
            passed = decision == expected
            test1_results.append((person, test_file, distance, decision, passed))
            if passed:
                test1_passed += 1
            test1_total += 1

    persons_without_test = [p for p in persons if p not in person_test_files]

    # TEST 2 — Impostor verification
    test2_results = []
    test2_passed = 0
    test2_total = 0

    for person_p in sorted(person_test_files.keys()):
        test_files = person_test_files[person_p]
        for test_file in test_files:
            test_path = os.path.join(dataset_path, person_p, test_file)
            for person_q in sorted(templates.keys()):
                if person_q == person_p:
                    continue
                template = templates[person_q]
                distance, decision = verify_speaker(template, test_path, threshold=threshold)
                expected = 'REJECT'
                passed = decision == expected
                test2_results.append((person_p, test_file, person_q, distance, decision, passed))
                if passed:
                    test2_passed += 1
                test2_total += 1

    # TEST 3 — Threshold sensitivity
    # Collect all genuine and impostor distances for FAR/FRR
    genuine_scores = []
    impostor_scores = []

    # Genuine: from TEST 1 results
    for _, _, dist, _, _ in test1_results:
        genuine_scores.append(dist)

    # Impostor: from TEST 2 results
    for _, _, _, dist, _, _ in test2_results:
        impostor_scores.append(dist)

    thresholds_to_test = [threshold * 0.8, threshold * 1.0, threshold * 1.2]
    test3_results = []

    for t in thresholds_to_test:
        metrics = compute_far_frr(genuine_scores, impostor_scores, t)
        far = metrics['far'] * 100
        frr = metrics['frr'] * 100
        test3_results.append((t, far, frr))

    # Generate report
    output = []
    output.append("=" * 50)
    output.append("END-TO-END TEST RESULTS")
    output.append("=" * 50)
    output.append(f"Threshold: {threshold:.2f}")
    output.append("")

    output.append("TEST 1 — Genuine verification:")
    output.append(f"Result: {test1_passed}/{test1_total} passed")
    for person, test_file, dist, decision, passed in sorted(test1_results):
        status = "✓ PASS" if passed else "✗ FAIL"
        output.append(f"  {person}/{test_file} → {dist:.2f} → {decision} {status}")
    if persons_without_test:
        output.append(f"⚠ without test samples: {', '.join(persons_without_test)}")
    output.append("")

    output.append("TEST 2 — Impostor verification:")
    output.append(f"Result: {test2_passed}/{test2_total} passed")
    impostor_summary = {}
    for person_p, test_file, person_q, dist, decision, passed in sorted(test2_results):
        key = (person_p, test_file)
        if key not in impostor_summary:
            impostor_summary[key] = []
        impostor_summary[key].append((person_q, dist, decision, passed))

    for (person_p, test_file), results in sorted(impostor_summary.items()):
        output.append(f"  {person_p}/{test_file} (tested against {len(results)} speakers):")
        for person_q, dist, decision, passed in results:
            status = "✓" if decision == 'REJECT' else "✗"
            output.append(f"    vs {person_q} → {dist:.2f} → {decision} {status}")
    output.append("")

    output.append("TEST 3 — Threshold sensitivity:")
    for t, far, frr in test3_results:
        multiplier = t / threshold
        output.append(f"  Threshold {threshold:.2f}*{multiplier:.1f} ({t:.2f}): FAR={far:.1f}%, FRR={frr:.1f}%")
    output.append("")

    # Overall summary
    overall_passed = 0
    overall_total = 3
    if test1_total > 0 and test1_passed == test1_total:
        overall_passed += 1
    if test2_total > 0 and test2_passed == test2_total:
        overall_passed += 1
    if test3_results:
        overall_passed += 1

    output.append(f"OVERALL: {overall_passed}/3 test categories passed")
    output.append("")

    # Recommendation
    if test1_passed == test1_total and test2_passed == test2_total:
        output.append("Recommendation: Threshold is acceptable")
    elif test1_passed < test1_total * 0.9:
        output.append("Recommendation: Consider lowering the threshold (too many rejections)")
    elif test2_passed < test2_total * 0.9:
        output.append("Recommendation: Consider raising the threshold (too many acceptances)")
    else:
        output.append("Recommendation: Threshold is acceptable but monitor FAR/FRR trade-off")

    output.append("=" * 50)

    # Print and save
    for line in output:
        print(line)

    with open(output_file, 'w') as f:
        for line in output:
            f.write(line + '\n')

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run end-to-end verification tests for a dataset')
    parser.add_argument('--dataset', default='custom_dataset', help='Dataset name from config/dataset_config.json')
    args = parser.parse_args()
    test_end_to_end(dataset_name=args.dataset)