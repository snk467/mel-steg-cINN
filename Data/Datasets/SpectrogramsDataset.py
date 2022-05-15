import torch
from torch.utils.data import Dataset
import librosa
import random
import numpy as np
import os

def load_audio(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    return audio, sample_rate

def get_audio_files(audio_files_directory):
    audio_files = np.array()
    for filename in os.listdir():
        path = os.path.join(audio_files_directory, filename)
        if os.path.isfile():
            audio_files.append(path)

    return audio_files

class SpectrogramDataset(Dataset):

    def __init__(self, audo_files_directory, segment_length, sample_rate, stft_parameters):
        self.audio_files = get_audio_files(audo_files_directory)
        self.stft_parameters = stft_parameters
        self.segment_length = segment_length
        self.sample_rate = sample_rate

        # Initialize RNG
        random.seed(1234)
        random.shuffle(self.audio_files)

    def __getitem__(self, index):
        # Read audio
        file_path = self.audio_files[index]
        audio, sample_rate = load_audio(file_path)

        #TODO: check sample_rate

        # Take segment - pad if too small
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            audio = np.pad(audio, (0, self.segment_length - audio.size(0)), 'constant')

        # Calculate spectrogram
        spectrogram = np.abs(librosa.stft(audio, hop_length=512))
        mel_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

        # Transform form numpy array to tensor
        mel_spectrogram_tensor = torch.from_numpy(mel_spectrogram)

        return mel_spectrogram_tensor

    def __len__(self):
        return len(self.audio_files)