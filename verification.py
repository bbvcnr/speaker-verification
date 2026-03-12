import numpy as np
from audio_utils import load_audio, normalize_audio, trim_silence
from features import extract_mfcc
from dtw import dtw_distance

def create_template(audio_paths, sr=16000):
    """
    Create a template from one or more audio recordings.
    For multiple recordings, MFCCs are extracted, padded to the same length,
    and averaged to create a robust speaker representation.

    - audio_paths: single path (str) or list of paths to audio files
    - sr: sample rate for processing

    Returns: averaged MFCC features of the template
    """
    if isinstance(audio_paths, str):
        # Single recording case
        audio_paths = [audio_paths]

    mfcc_list = []
    max_frames = 0

    # Extract MFCCs from each recording
    for path in audio_paths:
        audio, _ = load_audio(path, sr=sr)
        audio = normalize_audio(audio)
        audio, _ = trim_silence(audio, sr)
        mfcc = extract_mfcc(audio, sr)
        mfcc_list.append(mfcc)
        max_frames = max(max_frames, mfcc.shape[1])

    # Pad shorter sequences with zeros and average
    padded_mfccs = []
    for mfcc in mfcc_list:
        if mfcc.shape[1] < max_frames:
            # Pad with zeros on the right (time axis)
            padding = np.zeros((mfcc.shape[0], max_frames - mfcc.shape[1]))
            mfcc = np.hstack([mfcc, padding])
        padded_mfccs.append(mfcc)

    # Average across recordings
    template_mfcc = np.mean(padded_mfccs, axis=0)
    return template_mfcc

def verify_speaker(template_mfcc, test_audio_path, sr=16000, threshold=1000, normalize_dtw=False):
    """
    Verify if the test audio matches the template.
    - template_mfcc: MFCC features of the template
    - test_audio_path: path to the test audio file
    - sr: sample rate for processing
    - threshold: decision threshold for DTW distance
    - normalize_dtw: if True, normalize distance by warping path length

    Returns: (distance, decision) where decision is 'ACCEPT' or 'REJECT'
    """
    audio, _ = load_audio(test_audio_path, sr=sr)
    audio = normalize_audio(audio)
    audio, _ = trim_silence(audio, sr)
    test_mfcc = extract_mfcc(audio, sr)

    distance = dtw_distance(template_mfcc.T, test_mfcc.T, normalize=normalize_dtw)  # Transpose for (time, features)

    if distance < threshold:
        decision = 'ACCEPT'
    else:
        decision = 'REJECT'

    return distance, decision