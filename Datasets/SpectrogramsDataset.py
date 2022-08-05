import random
from torch.utils.data import Dataset
from Utilities import *
from Normalization import *
import torch
from PIL import Image
import torchvision.transforms as transforms
        

class SpectrogramsDataset(Dataset):

    def __init__(self, dataset_directory, train, colormap="parula_norm_lab", augmentor=None):
        files = get_files(dataset_directory)

        # Filter spectrograms from files
        self.spectrograms = list(filter(lambda k: f"spectrogram_color_{colormap}" in k, files))  
        self.colormap = Colormap.from_colormap(colormap)

        if train:
            dataset_range = range(len(self.spectrograms) // 10, len(self.spectrograms))
        else:
            dataset_range = range(len(self.spectrograms) // 10)

        self.spectrograms = self.spectrograms[dataset_range]

        # Augmentation
        self.augmentor = augmentor

    def __getitem__(self, index):

        # Load spectrogram from .npy
        with open(self.spectrograms[index], 'rb') as file:
            spectrogram_data = np.load(file)      

        L_tensor = torch.from_numpy(spectrogram_data[:,:,0])
        
        color_mapping_tensor = torch.from_numpy(self.colormap.get_indexes_from_colors(spectrogram_data))
        
        # Augment the tensor
        if self.augmentor is not None:
            L_tensor = self.augmentor(L_tensor)

        # Return the tensor
        return L_tensor, color_mapping_tensor

    def __len__(self):
        return len(self.spectrograms)


