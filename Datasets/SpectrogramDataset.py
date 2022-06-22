import random
from torch.utils.data import Dataset
from Utilities import *
from Normalization import *
import torch

class SpectrogramDataset(Dataset):

    def __init__(self, dataset_directory, augmentor=None):
        files = get_files(dataset_directory)

        # Filter spectrograms from files
        self.spectrograms = list(filter(lambda k: 'spec_color_lab' in k, files))  

        # Initialize RNG
        random.seed(1234)
        random.shuffle(self.spectrograms)

        # Augmentation
        self.augment = augmentor

    def __getitem__(self, index):

        # Load file as tensor
        spectrogram_tensor = torch.load(self.spectrograms[index])

        # Augment the tensor
        if self.augmentor is not None:
            spectrogram_tensor = self.augmentor(spectrogram_tensor)

        # Return the tensor
        return spectrogram_tensor

    def __len__(self):
        return len(self.spectrograms)


