import sounddevice as sd
from scipy.io import wavfile
import os
import sys
import time
from pathlib import Path

# Add project root to path so we can import core modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def record_custom_dataset():
    fs = 16000
    duration = 1.5
    samples_per_person = 5
    output_dir = "data/custom_dataset"
    phrase = input("Enter the phrase to say: ").strip()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
        name = input("Enter colleague name (or 'q' to quit): ").strip().lower()
        if name == 'q':
            break

        person_dir = os.path.join(output_dir, name)
        os.makedirs(person_dir, exist_ok=True)

        print(f"Correct phrase to say: '{phrase}'")
        input(f"Ready to record {samples_per_person} correct samples for {name.upper()}. Press ENTER...")

        for i in range(1, samples_per_person + 1):
            print(f"Say: '{phrase}'")
            print(f"[{i}/{samples_per_person}] Recording in 3... ", end="", flush=True)
            time.sleep(1)
            print("2... ", end="", flush=True)
            time.sleep(1)
            print("1... ", end="", flush=True)
            time.sleep(1)
            print("START!")

            audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()

            file_path = os.path.join(person_dir, f"sample_{i}.wav")
            wavfile.write(file_path, fs, audio_data)
            print(f"Saved: {file_path}")

        wrong_folder = input("Record wrong-phrase samples for this person? (y/n): ").strip().lower()
        if wrong_folder == 'y':
            wrong_phrase = input("Enter the wrong phrase to say: ").strip()
            wrong_dir = os.path.join(output_dir, f"{name}_wrong")
            os.makedirs(wrong_dir, exist_ok=True)
            input(f"Ready to record {samples_per_person} wrong-phrase samples for {name.upper()}. Press ENTER...")

            for i in range(1, samples_per_person + 1):
                print(f"Say: '{wrong_phrase}'")
                print(f"[{i}/{samples_per_person}] Recording in 3... ", end="", flush=True)
                time.sleep(1)
                print("2... ", end="", flush=True)
                time.sleep(1)
                print("1... ", end="", flush=True)
                time.sleep(1)
                print("START!")

                audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
                sd.wait()

                file_path = os.path.join(wrong_dir, f"sample_{i}.wav")
                wavfile.write(file_path, fs, audio_data)
                print(f"Saved: {file_path}")

        print(f"Done with {name}!")

if __name__ == "__main__":
    record_custom_dataset()