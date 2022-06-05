import os
import librosa
import numpy as np

def get_audio_files(audio_files_directory):
    audio_files = []
    for filename in os.listdir(audio_files_directory):
        path = os.path.join(audio_files_directory, filename)
        if os.path.isfile(path):
            audio_files.append(path)

    return np.array(audio_files)

def load_audio(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    return audio, sample_rate




