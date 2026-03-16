import numpy as np

def dtw_distance(seq1, seq2, normalize=False, use_band=True, band_width=None):
    """
    Compute Dynamic Time Warping (DTW) distance between two sequences.
    
    DTW is used for measuring similarity between two temporal sequences
    that may vary in speed or timing. It's particularly useful for speech
    because it can handle variations in speaking rate.
    
    Optionally applies Sakoe-Chiba band constraint to improve computational 
    efficiency by limiting the warping path to a diagonal band around the main diagonal.
    This is especially useful for speaker verification where sequences are typically
    similar in length.

    Args:
        seq1, seq2: sequences to compare (MFCC frames over time)
        normalize: if True, normalize distance by the length of the warping path
        use_band: if True, apply Sakoe-Chiba band constraint (auto-disabled for very short sequences)
        band_width: band width as absolute value; if None, computed as 15% of max sequence length

    Returns: 
        DTW distance (lower values indicate more similarity)
    """
    n, m = len(seq1), len(seq2)
    
    # For very short sequences, disable band constraint to avoid impossibly restrictive band
    if use_band and min(n, m) < 20:
        use_band = False
    
    # Compute band width if using band constraint
    if use_band and band_width is None:
        # Band width ~15% of sequence length (standard for speech)
        band_width = max(2, int(0.15 * max(n, m)))
    
    # Initialize DTW matrix
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0

    # Fill DTW matrix
    for i in range(1, n+1):
        # Determine column range based on band constraint
        if use_band:
            j_min = max(1, i - band_width)
            j_max = min(m + 1, i + band_width + 1)
        else:
            j_min = 1
            j_max = m + 1
        
        for j in range(j_min, j_max):
            if dtw_matrix[i-1, j] == np.inf and \
               dtw_matrix[i, j-1] == np.inf and \
               dtw_matrix[i-1, j-1] == np.inf:
                # Skip if all neighbors are unreachable (outside band)
                continue
                
            cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],     # insertion
                                          dtw_matrix[i, j-1],     # deletion
                                          dtw_matrix[i-1, j-1])   # match

    distance = dtw_matrix[n, m]
    
    if normalize:
        # Normalize by the length of the optimal warping path
        # The path length is approximately n + m for unconstrained DTW
        path_length = n + m
        distance /= path_length

    return distance