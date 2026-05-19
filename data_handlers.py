"""
Dataset handling utilities for text-dependent speaker verification.

Supports:
- Google Speech Commands (fixed keyword, single-word utterances)
"""

import os
import json
import urllib.request
import tarfile
import glob
import random
from pathlib import Path
from collections import defaultdict


class GoogleSpeechCommandsHandler:
    """Download and organize Google Speech Commands dataset."""
    
    DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    
    @staticmethod
    def download(target_dir="data/speech_commands", keyword="yes"):
        """
        Download Google Speech Commands dataset and extract keyword.
        
        Args:
            target_dir: directory to store dataset
            keyword: single keyword to extract (e.g., "yes", "no", "up", "down")
            
        Returns:
            dict with keys:
                - 'keyword_dir': path to keyword directory
                - 'speaker_recordings': dict mapping speaker_id -> list of file paths
                - 'n_speakers': number of speakers
                - 'n_recordings': total recordings
        """
        os.makedirs(target_dir, exist_ok=True)
        
        # Check if already downloaded
        keyword_dir = os.path.join(target_dir, keyword)
        if os.path.exists(keyword_dir):
            print(f"Dataset already exists at {keyword_dir}")
            return GoogleSpeechCommandsHandler._organize_speakers(keyword_dir)
        
        # Download
        print(f"Downloading Google Speech Commands dataset...")
        tar_path = os.path.join(target_dir, "speech_commands_v0.02.tar.gz")
        
        if not os.path.exists(tar_path):
            try:
                urllib.request.urlretrieve(
                    GoogleSpeechCommandsHandler.DATASET_URL,
                    tar_path,
                    reporthook=GoogleSpeechCommandsHandler._download_progress_hook
                )
                print(f"\nDownloaded to {tar_path}")
            except Exception as e:
                print(f"Error downloading: {e}")
                print("Alternative: Download manually from https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html")
                raise
        
        # Extract
        print(f"Extracting {keyword} keyword from dataset...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=target_dir)
        
        # Clean up tar file
        os.remove(tar_path)
        
        # Organize by speaker
        return GoogleSpeechCommandsHandler._organize_speakers(keyword_dir)
    
    @staticmethod
    def _organize_speakers(keyword_dir):
        """
        Organize keyword files by speaker ID.
        
        Google Speech Commands filenames: <speaker_id>_nohash_<index>.wav
        
        Args:
            keyword_dir: directory containing .wav files for keyword
            
        Returns:
            dict with speaker organization info
        """
        speaker_recordings = defaultdict(list)
        
        # Find all .wav files
        wav_files = glob.glob(os.path.join(keyword_dir, "*.wav"))
        
        # Parse speaker IDs from filenames
        for wav_file in wav_files:
            filename = os.path.basename(wav_file)
            # Format: <speaker_id>_nohash_<index>.wav
            parts = filename.replace('.wav', '').split('_nohash_')
            if len(parts) == 2:
                speaker_id = parts[0]
                speaker_recordings[speaker_id].append(wav_file)
        
        n_speakers = len(speaker_recordings)
        n_recordings = sum(len(files) for files in speaker_recordings.values())
        
        print(f"Found {n_speakers} speakers with {n_recordings} total recordings")
        
        return {
            'keyword_dir': keyword_dir,
            'speaker_recordings': dict(speaker_recordings),
            'n_speakers': n_speakers,
            'n_recordings': n_recordings
        }
    
    @staticmethod
    def select_speakers(speaker_recordings, n_speakers=20, min_recordings=5):
        """
        Select speakers with sufficient recordings.
        
        Args:
            speaker_recordings: dict mapping speaker_id -> list of file paths
            n_speakers: target number of speakers to select
            min_recordings: minimum recordings per speaker
            
        Returns:
            dict mapping selected speaker_id -> list of file paths
        """
        # Filter speakers with enough recordings
        filtered = {
            sid: files for sid, files in speaker_recordings.items()
            if len(files) >= min_recordings
        }
        
        print(f"Found {len(filtered)} speakers with >={min_recordings} recordings")
        
        # Randomly select n_speakers
        selected_ids = random.sample(list(filtered.keys()), min(n_speakers, len(filtered)))
        selected_speakers = {sid: filtered[sid] for sid in selected_ids}
        
        print(f"Selected {len(selected_speakers)} speakers for evaluation")
        
        return selected_speakers
    
    @staticmethod
    def create_text_dependent_trials(
        selected_speakers,
        template_size=3,
        impostor_speakers_per_test=5,
        random_seed=42,
    ):
        """
        Create templates and test trials for text-dependent verification.
        
        Args:
            selected_speakers: dict mapping speaker_id -> list of file paths
            template_size: number of recordings to use for template
            impostor_speakers_per_test: maximum impostor speakers sampled per test trial
            random_seed: seed used for deterministic trial generation
            
        Returns:
            dict with:
                - 'templates': dict mapping speaker_id -> list of template files
                - 'test_sets': dict mapping speaker_id -> list of test files
                - 'genuine_pairs': list of trial dicts for genuine tests
                - 'impostor_pairs': list of trial dicts for impostor tests
        """
        rng = random.Random(random_seed)

        templates = {}
        test_sets = {}
        
        # Split each speaker's recordings
        for speaker_id, recordings in selected_speakers.items():
            rng.shuffle(recordings)
            templates[speaker_id] = recordings[:template_size]
            test_sets[speaker_id] = recordings[template_size:]
        
        # Generate genuine test trials
        genuine_pairs = []
        for speaker_id in selected_speakers.keys():
            for test_file in test_sets[speaker_id]:
                genuine_pairs.append({
                    'template_speaker': speaker_id,
                    'test_speaker': speaker_id,
                    'test_file': test_file,
                    'label': True
                })
        
        # Generate impostor test trials using sampled impostor speakers
        impostor_pairs = []
        speaker_ids = list(selected_speakers.keys())
        
        for template_speaker in speaker_ids:
            other_speakers = [s for s in speaker_ids if s != template_speaker]
            n_impostors = min(impostor_speakers_per_test, len(other_speakers))
            impostor_speakers = rng.sample(other_speakers, n_impostors)
            
            for impostor_speaker in impostor_speakers:
                test_files = test_sets.get(impostor_speaker, [])
                if not test_files:
                    continue
                test_file = rng.choice(test_files)
                impostor_pairs.append({
                    'template_speaker': template_speaker,
                    'test_speaker': impostor_speaker,
                    'test_file': test_file,
                    'label': False
                })

        print(f"Generated {len(genuine_pairs)} genuine trials")
        print(f"Generated {len(impostor_pairs)} impostor trials")
        
        return {
            'templates': templates,
            'test_sets': test_sets,
            'genuine_pairs': genuine_pairs,
            'impostor_pairs': impostor_pairs
        }
    
    @staticmethod
    def _download_progress_hook(block_num, block_size, total_size):
        """Progress hook for urllib download."""
        downloaded = block_num * block_size
        percent = min(downloaded * 100 // total_size, 100)
        print(f"\rDownloading: {percent}%", end='', flush=True)


def load_dataset_config(dataset_name):
    config_path = Path(__file__).resolve().parent / 'config' / 'dataset_config.json'
    if not config_path.exists():
        return {}
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get(dataset_name, {})


class CustomDatasetHandler:
    """Handler for custom speaker verification datasets."""

    @staticmethod
    def load_speaker_recordings(dataset_path):
        recordings = {}
        if not os.path.exists(dataset_path):
            return recordings

        for speaker in sorted(os.listdir(dataset_path)):
            speaker_path = os.path.join(dataset_path, speaker)
            if not os.path.isdir(speaker_path):
                continue
            files = sorted(glob.glob(os.path.join(speaker_path, '*.wav')))
            if files:
                recordings[speaker] = files
        return recordings

    @staticmethod
    def select_speakers(speaker_recordings, n_speakers=None, min_recordings=None, random_seed=42):
        selected = {
            speaker: files
            for speaker, files in speaker_recordings.items()
            if min_recordings is None or len(files) >= min_recordings
        }

        speaker_ids = sorted(selected.keys())
        if n_speakers is not None and len(speaker_ids) > n_speakers:
            rng = random.Random(random_seed)
            speaker_ids = rng.sample(speaker_ids, n_speakers)

        return {speaker: selected[speaker] for speaker in speaker_ids}

    @staticmethod
    def split_enrollment_test(speaker_recordings, enrollment_count=3, test_count=2, random_seed=42):
        rng = random.Random(random_seed)
        templates = {}
        test_sets = {}

        for speaker, files in speaker_recordings.items():
            if len(files) < enrollment_count + test_count:
                continue

            files = list(files)
            rng.shuffle(files)
            templates[speaker] = files[:enrollment_count]
            test_sets[speaker] = files[enrollment_count:enrollment_count + test_count]

        return templates, test_sets

    @staticmethod
    def create_verification_trials(templates, test_sets, impostor_speakers_per_test=5, random_seed=42):
        rng = random.Random(random_seed)
        genuine_trials = []
        impostor_trials = []
        speaker_ids = list(templates.keys())

        for template_speaker in speaker_ids:
            for test_file in test_sets.get(template_speaker, []):
                genuine_trials.append({
                    'template_speaker': template_speaker,
                    'test_speaker': template_speaker,
                    'test_file': test_file,
                    'label': True
                })

                other_speakers = [s for s in speaker_ids if s != template_speaker]
                n_impostors = min(impostor_speakers_per_test, len(other_speakers))
                sampled_impostors = rng.sample(other_speakers, n_impostors)

                for impostor_speaker in sampled_impostors:
                    impostor_tests = test_sets.get(impostor_speaker, [])
                    if not impostor_tests:
                        continue
                    impostor_file = rng.choice(impostor_tests)
                    impostor_trials.append({
                        'template_speaker': template_speaker,
                        'test_speaker': impostor_speaker,
                        'test_file': impostor_file,
                        'label': False
                    })

        return genuine_trials, impostor_trials


class HeySNIPSHandler:
    """Handler for HeySNIPS speaker verification datasets."""

    @staticmethod
    def load_speaker_recordings(dataset_path):
        return CustomDatasetHandler.load_speaker_recordings(dataset_path)

    @staticmethod
    def select_speakers(speaker_recordings, n_speakers=None, min_recordings=None, random_seed=42):
        return CustomDatasetHandler.select_speakers(
            speaker_recordings,
            n_speakers=n_speakers,
            min_recordings=min_recordings,
            random_seed=random_seed,
        )

    @staticmethod
    def split_enrollment_test(speaker_recordings, enrollment_count=3, test_count=2, random_seed=42):
        return CustomDatasetHandler.split_enrollment_test(
            speaker_recordings,
            enrollment_count=enrollment_count,
            test_count=test_count,
            random_seed=random_seed,
        )

    @staticmethod
    def create_verification_trials(templates, test_sets, impostor_speakers_per_test=5, random_seed=42):
        return CustomDatasetHandler.create_verification_trials(
            templates,
            test_sets,
            impostor_speakers_per_test=impostor_speakers_per_test,
            random_seed=random_seed,
        )


def get_dataset_handler(dataset_name):
    if dataset_name == 'custom_dataset':
        return CustomDatasetHandler
    if dataset_name == 'heysnips':
        return HeySNIPSHandler
    if dataset_name == 'speech_commands':
        return GoogleSpeechCommandsHandler
    raise ValueError(f"Unknown dataset handler: {dataset_name}")
