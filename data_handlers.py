"""
Dataset handling utilities for text-dependent speaker verification.

Supports:
- Google Speech Commands (fixed keyword, single-word utterances)
"""

import os
import urllib.request
import tarfile
import glob
import random
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
    def create_text_dependent_trials(selected_speakers, template_size=3):
        """
        Create templates and test trials for text-dependent verification.
        
        Args:
            selected_speakers: dict mapping speaker_id -> list of file paths
            template_size: number of recordings to use for template
            
        Returns:
            dict with:
                - 'templates': dict mapping speaker_id -> list of template files
                - 'test_sets': dict mapping speaker_id -> list of test files
                - 'genuine_pairs': list of (template_files, test_file, label=True)
                - 'impostor_pairs': list of (template_files, test_file, label=False)
        """
        templates = {}
        test_sets = {}
        
        # Split each speaker's recordings
        for speaker_id, recordings in selected_speakers.items():
            random.shuffle(recordings)
            templates[speaker_id] = recordings[:template_size]
            test_sets[speaker_id] = recordings[template_size:]
        
        # Generate genuine pairs
        genuine_pairs = []
        for speaker_id in selected_speakers.keys():
            for test_file in test_sets[speaker_id]:
                genuine_pairs.append({
                    'template_speaker': speaker_id,
                    'test_speaker': speaker_id,
                    'test_file': test_file,
                    'label': True
                })
        
        # Generate impostor pairs (random sampling)
        impostor_pairs = []
        speaker_ids = list(selected_speakers.keys())
        
        for template_speaker in speaker_ids:
            # Sample 5 other speakers for impostor trials
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
                        'test_file': test_file,
                        'label': False
                    })
                    break  # Only take first test file per impostor speaker
        
        print(f"Generated {len(genuine_pairs)} genuine pairs")
        print(f"Generated {len(impostor_pairs)} impostor pairs")
        
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
