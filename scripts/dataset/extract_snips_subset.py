"""
Hey Snips Dataset Extractor - Metadata-Aware Version
=====================================================
Uses JSON metadata (worker_id, is_hotword) to group samples
by real speakers and filter only genuine "Hey Snips" utterances.

Archive structure (hey_snips_fl_5.0.tar):
  hey_snips_fl_amt/train.json
  hey_snips_fl_amt/audio_files/<uuid>.wav

Place in: scripts/dataset/extract_snips_subset.py
Run with: python scripts/dataset/extract_snips_subset.py
          python scripts/dataset/extract_snips_subset.py --debug
"""

import tarfile
import json
import os
import sys
from pathlib import Path
from collections import defaultdict


# ── Config ────────────────────────────────────────────────────────────────────

TAR_PATH                = "hey_snips_fl_5.0.tar"
TARGET_BASE_DIR         = os.path.join("data", "heysnips")

# JSON splits to try, in priority order (train has the most samples)
SPLITS                  = ["train", "dev", "test"]

# Top-level directory name inside the tar archive
TAR_ROOT                = "hey_snips_fl_amt"

MIN_SAMPLES_PER_SPEAKER = 4   # Minimum hotword samples required to include a speaker
SAMPLES_PER_SPEAKER     = 6   # Samples to extract per speaker (3 enrollment + 3 test)
NUM_SPEAKERS            = 30  # Number of speakers to extract


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_metadata_from_tar(tar):
    """
    Loads JSON metadata from the known split files inside the tar archive.
    Returns a list of sample dicts, or [] if none found.
    Tries train first (largest split with most hotword samples).
    """
    candidates = [
        f"{TAR_ROOT}/train.json",        # try train first — largest, most hotwords
        f"{TAR_ROOT}/dev.json",
        f"{TAR_ROOT}/test.json",
        "train.json",
        "dev.json",
        "test.json",
        "hey_snips/train.json",
        "hey_snips_v2/train.json",
    ]

    for name in candidates:
        try:
            member = tar.getmember(name)
            f = tar.extractfile(member)
            if f:
                data = json.load(f)
                print(f"  Loaded metadata: '{name}' ({len(data)} samples)")
                return data
        except KeyError:
            continue
        except json.JSONDecodeError as e:
            print(f"  Warning: '{name}' is not valid JSON ({e})")

    return []


def list_all_json_members(tar):
    """Returns all .json paths in the archive."""
    return [m.name for m in tar.getmembers() if m.name.endswith(".json")]


def group_by_speaker(samples):
    """
    Groups samples by worker_id (the real speaker identity).
    Keeps only is_hotword == 1 samples (genuine 'Hey Snips' utterances).
    """
    grouped = defaultdict(list)
    skipped_non_hotword = 0

    for sample in samples:
        if not sample.get("is_hotword", 0):
            skipped_non_hotword += 1
            continue
        worker_id = sample.get("worker_id") or sample.get("speaker_id")
        if not worker_id:
            continue
        grouped[str(worker_id)].append(sample)

    print(f"  Skipped non-hotword samples : {skipped_non_hotword}")
    print(f"  Speakers with >= 1 hotword  : {len(grouped)}")
    return grouped


def select_speakers(grouped, min_samples, n_speakers, samples_each):
    """
    Selects speakers that have enough hotword samples and
    trims each speaker's list to samples_each entries.
    """
    eligible = {
        wid: samples
        for wid, samples in grouped.items()
        if len(samples) >= min_samples
    }
    print(f"  Speakers with >= {min_samples} hotword samples: {len(eligible)}")

    if len(eligible) < n_speakers:
        print(
            f"  Warning: requested {n_speakers} speakers but only "
            f"{len(eligible)} are eligible. Using all available."
        )
        n_speakers = len(eligible)

    selected_ids = sorted(eligible.keys())[:n_speakers]
    return {wid: eligible[wid][:samples_each] for wid in selected_ids}


def resolve_wav_path(tar, audio_file_path):
    """
    Resolves the tar member path for a given audio_file_path from JSON metadata.
    The JSON stores only the filename (uuid.wav); the actual tar path is
    hey_snips_fl_amt/audio_files/uuid.wav. Tries several variants.
    Returns the matching tar member path string, or None if not found.
    """
    fname = Path(audio_file_path).name  # strip any directory prefix, keep uuid.wav

    candidates = [
        f"{TAR_ROOT}/audio_files/{fname}",   # hey_snips_fl_amt/audio_files/uuid.wav  (actual)
        audio_file_path,                      # exactly as written in JSON
        f"audio_files/{fname}",
        f"hey_snips/audio_files/{fname}",
        fname,
    ]
    for candidate in candidates:
        try:
            tar.getmember(candidate)
            return candidate
        except KeyError:
            continue
    return None


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_snips():
    script_dir    = Path(__file__).resolve().parent.parent.parent  # project root
    tar_full_path = script_dir / TAR_PATH

    if not tar_full_path.exists():
        print(f"Error: Could not find {TAR_PATH} at {tar_full_path}")
        print("Make sure the tar file is in the project root directory.")
        return False

    print(f"Opening {TAR_PATH} ...")

    try:
        with tarfile.open(tar_full_path, "r") as tar:

            # 1. Load metadata ─────────────────────────────────────────────────
            print("\n[1/4] Searching for JSON metadata ...")
            samples = load_metadata_from_tar(tar)

            if not samples:
                print("\nCould not find JSON metadata automatically.")
                print("JSON files present in archive:")
                for j in list_all_json_members(tar)[:20]:
                    print(f"  {j}")
                print("\nUpdate TAR_ROOT or SPLITS in this script and try again.")
                return False

            # 2. Group by speaker, filter hotword-only ─────────────────────────
            print("\n[2/4] Grouping by speaker ...")
            grouped  = group_by_speaker(samples)
            selected = select_speakers(
                grouped,
                min_samples=MIN_SAMPLES_PER_SPEAKER,
                n_speakers=NUM_SPEAKERS,
                samples_each=SAMPLES_PER_SPEAKER,
            )
            print(f"\n  Selected {len(selected)} speakers x {SAMPLES_PER_SPEAKER} samples each")

            # 3. Extract WAV files ─────────────────────────────────────────────
            print("\n[3/4] Extracting audio files ...")
            total_extracted = 0
            total_missing   = 0
            output_base     = script_dir / TARGET_BASE_DIR

            for idx, (worker_id, speaker_samples) in enumerate(selected.items()):
                speaker_folder = output_base / f"snips_user_{idx:03d}"
                speaker_folder.mkdir(parents=True, exist_ok=True)

                speaker_ok = 0
                for i, sample in enumerate(speaker_samples):
                    audio_path = sample.get("audio_file_path", "")
                    if not audio_path:
                        print(f"  Warning: sample {sample.get('id')} has no audio_file_path")
                        total_missing += 1
                        continue

                    tar_member_path = resolve_wav_path(tar, audio_path)
                    if tar_member_path is None:
                        print(f"  Warning: WAV not found in tar for: {audio_path}")
                        total_missing += 1
                        continue

                    dest_path = speaker_folder / f"sample_{i:02d}.wav"
                    if dest_path.exists():
                        print(f"    Skipping existing: {dest_path.name}")
                        speaker_ok      += 1
                        total_extracted += 1
                        continue
                    try:
                        src = tar.extractfile(tar_member_path)
                        if src is None:
                            total_missing += 1
                            continue
                        with src, open(dest_path, "wb") as dst:
                            dst.write(src.read())
                        speaker_ok      += 1
                        total_extracted += 1
                    except Exception as e:
                        print(f"  Error extracting {tar_member_path}: {e}")
                        total_missing += 1

                print(
                    f"  snips_user_{idx:03d} (worker={worker_id}): "
                    f"{speaker_ok}/{SAMPLES_PER_SPEAKER} samples"
                )

            # 4. Summary ───────────────────────────────────────────────────────
            print(f"\n[4/4] Done!")
            print(f"  Extracted : {total_extracted} files")
            print(f"  Missing   : {total_missing} files")
            print(f"  Output dir: {output_base}")

            if total_missing > 0:
                print(
                    "\n  Note: if total_missing is large, run --debug to check\n"
                    "  that audio_file_path values in JSON match tar member paths."
                )

            return total_extracted > 0

    except tarfile.TarError as e:
        print(f"Error opening tar file: {e}")
        return False


# ── Debug mode ────────────────────────────────────────────────────────────────
# Lists archive structure and peeks at JSON fields — no files are written.
# Usage: python scripts/dataset/extract_snips_subset.py --debug

if __name__ == "__main__":
    if "--debug" in sys.argv:
        tar_path = Path(__file__).resolve().parent.parent.parent / TAR_PATH
        print(f"DEBUG: Listing contents of {TAR_PATH}\n")
        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()
            print(f"Total members in archive: {len(members)}")

            print("\nFirst 30 paths:")
            for m in members[:30]:
                print(f"  {m.name}")

            print("\nAll JSON files:")
            for m in members:
                if m.name.endswith(".json"):
                    print(f"  {m.name}")

            print("\nFirst 10 WAV files:")
            wavs = [m.name for m in members if m.name.endswith(".wav")]
            for w in wavs[:10]:
                print(f"  {w}")

            # Peek at first JSON to reveal field names and audio_file_path format
            json_members = [m for m in members if m.name.endswith(".json")]
            if json_members:
                print(f"\nFirst 2 entries from '{json_members[0].name}':")
                f = tar.extractfile(json_members[0])
                if f:
                    data = json.load(f)
                    for entry in data[:2]:
                        print(json.dumps(entry, indent=4))
    else:
        extract_snips()