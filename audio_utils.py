import librosa
import numpy as np

def load_audio(file_path, sr=None):
    """
    Load audio file using librosa.
    - file_path: path to the audio file (.flac or .wav)
    - sr: target sample rate for resampling (None to keep original)
    Returns: audio array, sample rate
    """
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate

def normalize_audio(audio):
    """
    Normalize audio to the range [-1, 1].
    This helps in consistent processing across different audio levels.
    """
    return librosa.util.normalize(audio)

def trim_silence(audio, sr, top_db=20):
    """
    Trim leading and trailing silence from audio.
    - top_db: threshold in dB below which audio is considered silence
    Returns: trimmed audio, start and end indices
    """
    trimmed_audio, index = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio, index