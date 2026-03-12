import librosa
import numpy as np

def extract_mfcc(audio, sr, n_mfcc=13, hop_length=512, n_fft=2048, include_deltas=True):
    """
    Extract MFCC features from audio, optionally including delta and delta-delta coefficients.
    MFCC (Mel-Frequency Cepstral Coefficients) are widely used in speech processing
    because they represent the short-term power spectrum of sound in a way that
    mimics human auditory perception. Delta coefficients capture the rate of change
    (velocity) of MFCCs over time, while delta-delta coefficients capture acceleration,
    providing temporal dynamics that improve speaker discrimination.

    - audio: audio time series
    - sr: sample rate
    - n_mfcc: number of MFCC coefficients to return
    - hop_length: number of samples between successive frames
    - n_fft: length of the FFT window
    - include_deltas: if True, include delta and delta-delta features (triples feature dimension)

    Returns: MFCC matrix of shape (n_mfcc * (1 + 2*include_deltas), time_frames)
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                hop_length=hop_length, n_fft=n_fft)

    if include_deltas:
        # Compute delta (velocity) coefficients
        delta_mfcc = librosa.feature.delta(mfcc)
        # Compute delta-delta (acceleration) coefficients
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        # Concatenate: static + delta + delta-delta
        mfcc = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

    return mfcc