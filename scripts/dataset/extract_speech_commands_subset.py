import argparse
import os
import re
import shutil
from pathlib import Path


def extract_speech_commands_subset(
    source_dir='data/speech_commands/forward',
    output_dir='data/speech_commands_subset',
    min_samples=4,
    max_samples=5,
    num_speakers=35,
):
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_path}")

    pattern = re.compile(r'^(?P<speaker>[^_]+)_nohash_\d+\.wav$')
    speakers = {}

    for wav_file in sorted(source_path.glob('*.wav')):
        match = pattern.match(wav_file.name)
        if not match:
            continue
        speaker_id = match.group('speaker')
        speakers.setdefault(speaker_id, []).append(wav_file)

    filtered = {
        speaker: sorted(paths)
        for speaker, paths in speakers.items()
        if min_samples <= len(paths) <= max_samples
    }

    selected_speakers = sorted(filtered.keys())[:num_speakers]
    if len(selected_speakers) < num_speakers:
        print(f"Warning: only found {len(selected_speakers)} speakers with {min_samples}-{max_samples} samples.")

    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for speaker_id in selected_speakers:
        speaker_out_dir = output_path / speaker_id
        speaker_out_dir.mkdir(parents=True, exist_ok=True)
        for idx, source_file in enumerate(filtered[speaker_id], start=1):
            target_name = f"sample_{idx:02d}.wav"
            target_path = speaker_out_dir / target_name
            shutil.copy2(source_file, target_path)

    print(f"Extracted {len(selected_speakers)} speakers to {output_path}")
    print("Each speaker directory contains 4 or 5 files named sample_XX.wav.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract a balanced speech_commands subset by speaker.')
    parser.add_argument('--source', default='data/speech_commands/forward', help='Source directory containing speech_commands forward WAV files')
    parser.add_argument('--output', default='data/speech_commands_subset', help='Destination directory for the extracted subset')
    parser.add_argument('--num-speakers', default=35, type=int, help='Number of speakers to select')
    args = parser.parse_args()

    extract_speech_commands_subset(
        source_dir=args.source,
        output_dir=args.output,
        num_speakers=args.num_speakers,
    )
