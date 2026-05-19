#!/bin/bash
# Run this script ONCE on the Raspberry Pi after the first SSH session.
#
#   chmod +x ~/speaker-verification/scripts/pi/setup_pi.sh
#   ~/speaker-verification/scripts/pi/setup_pi.sh
#
# What it does:
#   1. Installs system packages (portaudio, libsndfile)
#   2. Installs Python packages via pip3
#   3. Creates the required directory layout
#   4. Identifies available audio devices
#   5. Optionally sets the USB sound card as the ALSA default

set -e

echo "=== Step 1: system packages ==="
sudo apt install -y \
    python3-pip \
    python3-numpy \
    python3-scipy \
    portaudio19-dev \
    python3-dev \
    libsndfile1 \
    libsndfile1-dev \
    libasound2-dev

echo ""
echo "=== Step 2: Python packages ==="
# Newer Pi OS (Debian Trixie) enforces PEP 668 — use a venv instead of --user
# --system-site-packages lets the venv reuse numpy/scipy already installed via apt
python3 -m venv --system-site-packages ~/speaker-verification/venv
~/speaker-verification/venv/bin/pip install sounddevice soundfile librosa RPi.GPIO

echo ""
echo "=== Step 3: directory layout ==="
mkdir -p ~/speaker-verification/core
mkdir -p ~/speaker-verification/templates/custom_dataset
mkdir -p ~/speaker-verification/config
mkdir -p ~/speaker-verification/audio_responses
mkdir -p ~/speaker-verification/scripts/pi
echo "Directories ready."

echo ""
echo "=== Step 4: audio devices ==="
echo "Available playback devices:"
aplay -l 2>/dev/null || echo "  (aplay not installed)"
echo ""
echo "Available capture devices:"
arecord -l 2>/dev/null || echo "  (arecord not installed)"

echo ""
echo "=== Step 5: ALSA default (optional) ==="
# Find the USB sound card index from aplay output
USB_CARD=$(aplay -l 2>/dev/null | grep -i usb | grep -oP 'card \K[0-9]+' | head -1)

if [ -n "$USB_CARD" ]; then
    echo "USB sound card detected as card $USB_CARD."
    read -rp "Set it as the ALSA default? [y/N] " ANSWER
    if [[ "$ANSWER" =~ ^[Yy]$ ]]; then
        cat > ~/.asoundrc <<EOF
defaults.pcm.card $USB_CARD
defaults.ctl.card $USB_CARD
EOF
        echo "~/.asoundrc written — USB card $USB_CARD is now the ALSA default."
        echo "sounddevice will now use it without any extra configuration in main.py."
    else
        echo "Skipped.  If you see 'no default device' errors, set AUDIO_DEVICE in main.py"
        echo "to the integer index shown by:"
        echo "  python3 -c \"import sounddevice; print(sounddevice.query_devices())\""
    fi
else
    echo "No USB audio card detected yet.  Plug in the sound card and rerun, or"
    echo "set AUDIO_DEVICE manually in main.py."
fi

echo ""
echo "=== Setup complete ==="
echo "Next: run deploy.ps1 from your Windows machine to copy project files."
echo "Then start the system with:"
echo "  ~/speaker-verification/venv/bin/python3 ~/speaker-verification/scripts/pi/main.py"
