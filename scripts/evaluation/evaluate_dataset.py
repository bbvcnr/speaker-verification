import argparse
import os
import sys
from pathlib import Path

# Add project root, script folder, and dataset script path so imports work cleanly
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
dataset_script_path = project_root / 'scripts' / 'dataset'
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(dataset_script_path))

from build_templates import build_templates
from tune_threshold import tune_threshold
from test_end_to_end import test_end_to_end
import advanced_metrics


def evaluate_dataset(dataset_name='custom_dataset', run_build=True, run_tune=True, run_end_to_end=True, run_advanced=True):
    print(f"\nStarting unified dataset evaluation for '{dataset_name}'\n")

    if run_build:
        print('=== Step 1: Build templates ===')
        build_templates(dataset_name=dataset_name)
        print('=== Templates build complete ===\n')

    if run_tune:
        print('=== Step 2: Tune threshold ===')
        tune_threshold(dataset_name=dataset_name)
        print('=== Threshold tuning complete ===\n')

    if run_end_to_end:
        print('=== Step 3: Run end-to-end tests ===')
        test_end_to_end(dataset_name=dataset_name)
        print('=== End-to-end tests complete ===\n')

    if run_advanced:
        print('=== Step 4: Compute advanced metrics ===')
        advanced_metrics.main(dataset_name=dataset_name)
        print('=== Advanced metrics complete ===\n')

    print(f"Unified evaluation finished for '{dataset_name}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unified evaluation for a dataset')
    parser.add_argument('--dataset', default='custom_dataset', help='Dataset name from config/dataset_config.json')
    parser.add_argument('--skip-build', action='store_true', help='Skip template building')
    parser.add_argument('--skip-tune', action='store_true', help='Skip threshold tuning')
    parser.add_argument('--skip-end-to-end', action='store_true', help='Skip end-to-end tests')
    parser.add_argument('--skip-advanced', action='store_true', help='Skip advanced metrics computation')
    args = parser.parse_args()

    evaluate_dataset(
        dataset_name=args.dataset,
        run_build=not args.skip_build,
        run_tune=not args.skip_tune,
        run_end_to_end=not args.skip_end_to_end,
        run_advanced=not args.skip_advanced,
    )
