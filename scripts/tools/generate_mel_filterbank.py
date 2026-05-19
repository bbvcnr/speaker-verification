"""
Generate mel filter bank diagram for thesis Section 2.3.1.

Produces figures/mel_filterbank.png showing all 26 mel filters on a Hz axis
with a secondary Mel-scale axis on top.

Usage:
    python scripts/tools/generate_mel_filterbank.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import librosa

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def main():
    sr = 16000
    n_fft = 2048
    n_mels = 26

    filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)  # (26, 1025)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)  # (1025,) in Hz

    fig, ax = plt.subplots(figsize=(12, 5))

    colormap = plt.cm.viridis
    for i, filt in enumerate(filterbank):
        peak = filt.max()
        normalized = filt / peak if peak > 0 else filt
        color = colormap(i / (n_mels - 1))
        ax.plot(freqs, normalized, color=color, linewidth=1.2)

    ax.set_xlim(0, 8000)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Filter Amplitude', fontsize=12)
    ax.set_title('Mel Filter Bank (26 filters, sr=16 kHz, n_fft=2048)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Secondary x-axis showing Mel scale
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())

    # Place Mel ticks at round Hz values so labels are clean
    hz_ticks = np.array([0, 500, 1000, 2000, 3000, 4000, 6000, 8000])
    mel_labels = [f"{hz_to_mel(f):.0f}" for f in hz_ticks]
    ax_top.set_xticks(hz_ticks)
    ax_top.set_xticklabels(mel_labels, fontsize=9)
    ax_top.set_xlabel('Mel Scale', fontsize=12)

    plt.tight_layout()

    figures_dir = Path(__file__).parent.parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    out_path = figures_dir / 'mel_filterbank.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Mel filterbank plot saved to {out_path}")


if __name__ == '__main__':
    main()
