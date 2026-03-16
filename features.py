import librosa
import numpy as np

def extract_mfcc(audio, sr, n_mfcc=13, hop_length=512, n_fft=2048, include_deltas=True, apply_cmvn=True):
    """
    Extract MFCC features from audio with optional delta/delta-delta and CMVN normalization.
    
    MFCC (Mel-Frequency Cepstral Coefficients) are widely used in speech processing
    because they represent the short-term power spectrum of sound in a way that
    mimics human auditory perception. Delta coefficients capture the rate of change
    (velocity) of MFCCs over time, while delta-delta coefficients capture acceleration,
    providing temporal dynamics that improve speaker discrimination.
    
    CMVN (Cepstral Mean and Variance Normalization) normalizes each coefficient across
    time frames, reducing speaker-independent acoustic variation and improving robustness
    to channel and environmental differences.

    Args:
        audio: audio time series
        sr: sample rate
        n_mfcc: number of MFCC coefficients to return
        hop_length: number of samples between successive frames
        n_fft: length of the FFT window
        include_deltas: if True, include delta and delta-delta features
        apply_cmvn: if True, apply Cepstral Mean and Variance Normalization

    Returns: 
        MFCC matrix of shape (n_mfcc * (1 + 2*include_deltas), time_frames)
        If apply_cmvn=True, features are normalized to zero mean and unit variance per coefficient.
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

    if apply_cmvn:
        mfcc = apply_cmvn_normalization(mfcc)

    return mfcc


def apply_cmvn_normalization(features, epsilon=1e-8):
    """
    Apply Cepstral Mean and Variance Normalization (CMVN) to feature matrix.
    
    CMVN normalizes each coefficient (row) across all time frames (columns):
    - Compute mean and standard deviation per coefficient
    - Normalize to zero mean and unit variance
    - Prevents division by zero with small epsilon
    
    Args:
        features: feature matrix of shape (n_features, n_frames)
        epsilon: small value to prevent division by zero
        
    Returns:
        normalized features of same shape
    """
    # Compute mean and std per feature (across frames)
    feature_mean = np.mean(features, axis=1, keepdims=True)
    feature_std = np.std(features, axis=1, keepdims=True)
    
    # Normalize
    normalized = (features - feature_mean) / (feature_std + epsilon)
    
    return normalized