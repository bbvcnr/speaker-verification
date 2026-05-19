import argparse
import os
import sys
from pathlib import Path

# Add project root to path so we can import core modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from data_handlers import load_dataset_config
from core.verification import create_template, verify_speaker

def build_templates(dataset_name='custom_dataset'):
    cfg = load_dataset_config(dataset_name)
    dataset_path = cfg.get('data_path', f'data/{dataset_name}')
    templates_dir = cfg.get('templates_dir', 'templates')
    enrollment_count = cfg.get('enrollment_count', 3)

    # Create templates directory if not exists
    os.makedirs(templates_dir, exist_ok=True)

    # Scan for persons
    if not os.path.exists(dataset_path):
        print("ERROR: Dataset path does not exist.")
        return

    persons = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    persons = sorted(persons)

    templates = {}
    persons_with_test = []
    persons_without_test = []

    # Create templates for all persons
    for person in persons:
        person_path = os.path.join(dataset_path, person)
        files = sorted([f for f in os.listdir(person_path) if f.endswith('.wav')])

        if len(files) < enrollment_count + 1:
            print(f"⚠ {person}: only {len(files)} samples — skipping (need at least {enrollment_count + 1} total recordings)")
            continue

        enrollment = files[:enrollment_count]
        test_files = files[enrollment_count:]

        enrollment_paths = [os.path.join(person_path, f) for f in enrollment]
        template = create_template(enrollment_paths)
        templates[person] = template
        np.save(os.path.join(templates_dir, f'{person}.npy'), template)

        if len(test_files) == 0:
            print(f"⚠ {person}: only {enrollment_count} samples — template created but no test samples for sanity check")
            persons_without_test.append(person)
        else:
            persons_with_test.append((person, test_files))

    # Sanity check for persons with test samples
    results = []
    genuine_scores = []
    min_impostor_scores = []

    for person, test_files in persons_with_test:
        template = templates[person]
        genuine_distances = []

        all_impostor_distances = []

        for test_file in test_files:
            test_path = os.path.join(dataset_path, person, test_file)
            genuine_dist, _ = verify_speaker(template, test_path)
            genuine_distances.append(genuine_dist)

            # Impostor distances
            for other_person, other_template in templates.items():
                if other_person != person:
                    impostor_dist, _ = verify_speaker(other_template, test_path)
                    all_impostor_distances.append(impostor_dist)

        avg_genuine = np.mean(genuine_distances)
        min_impostor = np.min(all_impostor_distances)
        max_impostor = np.max(all_impostor_distances)

        separable = avg_genuine < min_impostor
        status = "✓ SEPARABLE" if separable else "✗ OVERLAP"

        results.append({
            'person': person,
            'num_tests': len(test_files),
            'genuine': avg_genuine,
            'min_impostor': min_impostor,
            'max_impostor': max_impostor,
            'status': status
        })

        genuine_scores.append(avg_genuine)
        min_impostor_scores.append(min_impostor)

    # Print table
    if results:
        print("\nOsoba  | Test uzorci | Genuine Score | Min Impostor | Max Impostor | Status")
        print("-------|-------------|---------------|--------------|--------------|--------")
        for r in results:
            print(f"{r['person']:<6} | {r['num_tests']:<12} | {r['genuine']:<13.2f} | {r['min_impostor']:<12.2f} | {r['max_impostor']:<12.2f} | {r['status']}")

    if persons_without_test:
        print(f"\n⚠ without test samples: {', '.join(persons_without_test)}")

    # Summary
    total_enrolled = len(templates)
    with_test = len(persons_with_test)
    without_test = len(persons_without_test)
    separable_count = sum(1 for r in results if "SEPARABLE" in r['status'])

    print(f"\nTotal enrolled: {total_enrolled} persons ({with_test} with test samples, {without_test} without)")
    print(f"Separable: {separable_count}/{with_test}")

    if genuine_scores and min_impostor_scores:
        suggested_threshold = (np.mean(genuine_scores) + np.mean(min_impostor_scores)) / 2
        print(f"Suggested threshold: {suggested_threshold:.2f}")
        print("⚠ Warning: The dataset is small — this is a rough estimate. Run tune_threshold.py for a more accurate result.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build enrollment templates for a dataset')
    parser.add_argument('--dataset', default='custom_dataset', help='Dataset name from config/dataset_config.json')
    args = parser.parse_args()
    build_templates(dataset_name=args.dataset)