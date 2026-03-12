import numpy as np

def dtw_distance(seq1, seq2, normalize=False):
    """
    Compute Dynamic Time Warping (DTW) distance between two sequences.
    DTW is used for measuring similarity between two temporal sequences
    that may vary in speed or timing. It's particularly useful for speech
    because it can handle variations in speaking rate.

    - seq1, seq2: sequences to compare (MFCC frames over time)
    - normalize: if True, normalize distance by the length of the warping path

    Returns: DTW distance (lower values indicate more similarity)
    """
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix[0, :] = np.inf
    dtw_matrix[:, 0] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                          dtw_matrix[i, j-1],    # deletion
                                          dtw_matrix[i-1, j-1])  # match

    distance = dtw_matrix[n, m]
    if normalize:
        # Normalize by the length of the optimal warping path
        # The path length is approximately n + m for unconstrained DTW
        path_length = n + m
        distance /= path_length

    return distance