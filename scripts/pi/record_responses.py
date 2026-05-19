"""
Record the response audio files on your development machine.

Run this script BEFORE deploying to the Pi.  It will record each file one
by one and save them to audio_responses/ in the project root.  The Pi
main.py expects exactly these filenames.

Files recorded:
  prompt.wav         — played when the button is pressed ("please say passphrase")
  access_denied.wav  — played on rejection
  alem.wav           — played when Alem is verified
  alen.wav
  ena.wav
  ensar.wav
  lamija.wav
  nedzad.wav
  nejra.wav
  nijaz.wav

Usage:
  python scripts/pi/record_responses.py

Dependencies (install in your dev venv):
  pip install sounddevice soundfile
"""

import os
import sys
import time
import sounddevice as sd
import soundfile as sf
import numpy as np

SAMPLE_RATE    = 16000
RECORD_SECONDS = 4
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "audio_responses")

RECORDINGS = [
    (
        "prompt",
        'Please state your passphrase.',
        "Say the prompt that will play when the button is pressed.\n"
        '  Example: "Please state your passphrase."',
    ),
    (
        "access_denied",
        'Unknown speaker or phrase.  Access denied.',
        "Say the rejection message.\n"
        '  Example: "Unknown speaker or phrase.  Access denied."',
    ),
    ("alem",    "This is Alem.  Access granted.",    'Say: "This is Alem.  Access granted."'),
    ("alen",    "This is Alen.  Access granted.",    'Say: "This is Alen.  Access granted."'),
    ("ena",     "This is Ena.  Access granted.",     'Say: "This is Ena.  Access granted."'),
    ("ensar",   "This is Ensar.  Access granted.",   'Say: "This is Ensar.  Access granted."'),
    ("lamija",  "This is Lamija.  Access granted.",  'Say: "This is Lamija.  Access granted."'),
    ("nedzad",  "This is Nedzad.  Access granted.",  'Say: "This is Nedzad.  Access granted."'),
    ("nejra",   "This is Nejra.  Access granted.",   'Say: "This is Nejra.  Access granted."'),
    ("nijaz",   "This is Nijaz.  Access granted.",   'Say: "This is Nijaz.  Access granted."'),
]


def record_one(duration: int, sr: int) -> np.ndarray:
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    for remaining in range(duration, 0, -1):
        print(f"  {remaining}s ...", end="\r", flush=True)
        time.sleep(1)
    sd.wait()
    print()
    return audio.flatten()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nAudio device info:")
    print(sd.query_devices())
    print()

    total = len(RECORDINGS)
    for idx, (name, _example, instruction) in enumerate(RECORDINGS, 1):
        out_path = os.path.join(OUTPUT_DIR, f"{name}.wav")

        print(f"\n[{idx}/{total}]  {name}.wav")
        print(f"  {instruction}")

        while True:
            input("  Press Enter to start recording (Ctrl-C to skip) ...")
            print(f"  Recording {RECORD_SECONDS}s — speak now!")
            audio = record_one(RECORD_SECONDS, SAMPLE_RATE)

            sf.write(out_path, audio, SAMPLE_RATE)
            print(f"  Saved → {out_path}")

            print("  Playing back ...")
            sd.play(audio, SAMPLE_RATE)
            sd.wait()

            answer = input("  Keep this recording? [Y/n] ").strip().lower()
            if answer in ("", "y", "yes"):
                break
            print("  Re-recording ...")

    print(f"\nAll {total} files saved to:  {os.path.abspath(OUTPUT_DIR)}")
    print("Next: run scripts/pi/deploy.ps1 to copy them to the Pi.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)
