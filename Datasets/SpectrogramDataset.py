import random
from Noise import GaussianNoise
from torch.utils.data import Dataset
from Utilities import *
from Normalization import *

class SpectrogramDataset(Dataset):

    def __init__(self, audo_files_directory, segment_length, sample_rate, stft_parameters):
        self.audio_files = get_audio_files(audo_files_directory)
        self.stft_parameters = stft_parameters
        self.segment_length = segment_length
        self.sample_rate = sample_rate

        # Initialize RNG
        random.seed(1234)
        random.shuffle(self.audio_files)

        # Augmentation
        self.augment = GaussianNoise([0], [0.01])

    def __getitem__(self, index):
        raise NotImplementedError


    def __len__(self):
        return len(self.audio_files)


