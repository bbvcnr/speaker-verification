"""
Generate window function comparison figure for thesis Section 2.3.1.

Produces figures/window_comparison.png showing Hamming, Hann, and Rectangular
windows in both time and frequency domains.

Usage:
    python scripts/tools/generate_window_plots.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    N = 400
    n = np.arange(N)

    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    hann = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    rectangular = np.ones(N)

    windows = [
        ('Hamming', hamming, 'tab:blue'),
        ('Hann', hann, 'tab:orange'),
        ('Rectangular', rectangular, 'tab:green'),
    ]

    NFFT = 4096
    # Frequency bins corresponding to positive frequencies (0 to sr/2)
    # We only plot the first 512 bins as the spec says
    n_plot_bins = 512

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # --- Panel 1: Time domain ---
    for label, win, color in windows:
        ax1.plot(n, win, label=label, color=color, linewidth=1.5)

    ax1.set_xlim(0, N - 1)
    ax1.set_ylim(0, 1.1)
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title('Window Functions (N=400, 25 ms @ 16 kHz)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Frequency domain (magnitude in dB, normalised to 0 dB peak) ---
    for label, win, color in windows:
        spectrum = np.fft.fft(win, n=NFFT)
        magnitude = 20 * np.log10(np.abs(spectrum[:n_plot_bins]) + 1e-12)
        magnitude -= magnitude.max()  # normalise peak to 0 dB
        ax2.plot(np.arange(n_plot_bins), magnitude, label=label, color=color, linewidth=1.5)

    ax2.set_xlim(0, n_plot_bins - 1)
    ax2.set_ylim(-80, 5)
    ax2.set_xlabel('Bin Index (relative frequency)', fontsize=12)
    ax2.set_ylabel('Magnitude (dB)', fontsize=12)
    ax2.set_title('Frequency Response (normalized)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    figures_dir = Path(__file__).parent.parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    out_path = figures_dir / 'window_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Window comparison plot saved to {out_path}")


if __name__ == '__main__':
    main()
