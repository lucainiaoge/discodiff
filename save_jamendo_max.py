import os
import soundfile as sf
from datasets import load_dataset

def read_jamendo_max():
    dataset = load_dataset("amaai-lab/JamendoMaxCaps", data_dir="data")
    return dataset

def save_mp3_files(dataset, output_dir="data/jamendomaxcaps_audio_files"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, example in enumerate(dataset['train']):
        audio = example['audio']['array']
        sample_rate = example['audio']['sampling_rate']
        print(audio.shape, sample_rate)
        path = example['audio']['path']
        print(f"Processing file: {path}")
        output_path = os.path.join(output_dir, path)
        # sf.write(output_path, audio, sample_rate, format='MP3')
        print(f"Saved file: {output_path}")

if __name__ == "__main__":
    dataset = read_jamendo_max()
    save_mp3_files(dataset)
    print(f"MP3 files saved to {os.path.abspath('mp3_files')}")
