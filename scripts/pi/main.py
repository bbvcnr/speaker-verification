"""
Raspberry Pi speaker verification entry point.

Hardware:
  - Button: GPIO 17 -> button -> GND  (internal pull-up, active LOW)
  - USB sound card: card 1 on most Pi OS installs (arecord -l to confirm)

Directory layout expected on the Pi:
  ~/speaker-verification/
    core/                        (copied from dev machine)
    templates/custom_dataset/    (.npy files)
    config/custom_threshold.json
    audio_responses/             (.wav files recorded with record_responses.py)
    scripts/pi/main.py           (this file)

Run:
  python3 ~/speaker-verification/scripts/pi/main.py
"""

import os
import sys
import json
import time
import tempfile
import logging

import numpy as np
import sounddevice as sd
import soundfile as sf

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser("~/speaker-verification")  # ~ resolves to /home/malina on the Pi
sys.path.insert(0, PROJECT_ROOT)

from core.verification import verify_speaker

# ── configuration ─────────────────────────────────────────────────────────────
BUTTON_PIN      = 17
SAMPLE_RATE     = 16000   # processing rate (librosa resamples to this)
RECORD_RATE     = 44100   # USB mic native rate
RECORD_SECONDS  = 4
TEMPLATES_DIR   = os.path.join(PROJECT_ROOT, "templates", "custom_dataset")
THRESHOLD_FILE  = os.path.join(PROJECT_ROOT, "config", "custom_threshold.json")
RESPONSES_DIR   = os.path.join(PROJECT_ROOT, "audio_responses")

PLAYBACK_DEVICE = 0   # Pi built-in 3.5mm jack (bcm2835)
RECORD_DEVICE   = 1   # USB microphone

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_templates(directory):
    templates = {}
    for fname in os.listdir(directory):
        if fname.endswith(".npy"):
            speaker = fname[:-4]
            templates[speaker] = np.load(os.path.join(directory, fname))
    if not templates:
        raise RuntimeError(f"No .npy templates found in {directory}")
    log.info("Loaded templates: %s", list(templates.keys()))
    return templates


def load_threshold(path):
    with open(path) as f:
        data = json.load(f)
    threshold = data["threshold"]
    log.info("Threshold: %.4f", threshold)
    return threshold


def play_wav(path):
    if not os.path.exists(path):
        log.warning("Audio file not found: %s", path)
        return
    data, sr = sf.read(path, dtype="float32")
    sd.play(data, samplerate=sr, device=PLAYBACK_DEVICE)
    sd.wait()


def record_audio(duration):
    log.info("Recording %.1fs ...", duration)
    audio = sd.rec(int(duration * RECORD_RATE), samplerate=RECORD_RATE,
                   channels=1, dtype="float32", device=RECORD_DEVICE)
    sd.wait()
    return audio.flatten()


def identify_speaker(audio, templates, threshold):
    """Return (speaker_name, distance) or (None, distance) if rejected."""
    tmp = os.path.join(tempfile.gettempdir(), "pi_input.wav")
    sf.write(tmp, audio, RECORD_RATE)

    best_speaker  = None
    best_distance = float("inf")

    for speaker, template_mfcc in templates.items():
        dist, _ = verify_speaker(template_mfcc, tmp, sr=SAMPLE_RATE, threshold=float("inf"))
        log.debug("  %s: %.2f", speaker, dist)
        if dist < best_distance:
            best_distance = dist
            best_speaker  = speaker

    if best_distance < threshold:
        return best_speaker, best_distance
    return None, best_distance


def on_button_press(templates, threshold):
    log.info("Button pressed — starting verification")

    play_wav(os.path.join(RESPONSES_DIR, "prompt.wav"))
    time.sleep(0.3)

    audio = record_audio(RECORD_SECONDS)

    speaker, distance = identify_speaker(audio, templates, threshold)
    log.info("Best match: %s  distance: %.2f  threshold: %.2f",
             speaker or "NONE", distance, threshold)

    if speaker:
        response = os.path.join(RESPONSES_DIR, f"{speaker}.wav")
        log.info("ACCEPT → %s", speaker)
        play_wav(response)
    else:
        log.info("REJECT")
        play_wav(os.path.join(RESPONSES_DIR, "access_denied.wav"))


# ── main loop ─────────────────────────────────────────────────────────────────

def main():
    try:
        import RPi.GPIO as GPIO
    except ImportError:
        raise SystemExit("RPi.GPIO not available. Run this on the Raspberry Pi.")

    templates = load_templates(TEMPLATES_DIR)
    threshold = load_threshold(THRESHOLD_FILE)

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    log.info("Ready — press the button (GPIO %d)", BUTTON_PIN)

    try:
        while True:
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                on_button_press(templates, threshold)
                # Wait for release before re-arming
                while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                    time.sleep(0.05)
                time.sleep(0.2)
            time.sleep(0.05)
    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
