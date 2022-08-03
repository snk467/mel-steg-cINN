import random
from torch.utils.data import Dataset
from Utilities import *
from Normalization import *
import torch
from PIL import Image
import torchvision.transforms as transforms
        

class SpectrogramsDataset(Dataset):

    def __init__(self, dataset_directory, train, augmentor=None):
        files = get_files(dataset_directory)

        # Filter spectrograms from files
        self.spectrograms = list(filter(lambda k: 'spectrogram_color_rgb' in k, files))  

        if train:
            dataset_range = range(len(self.spectrograms) // 10, len(self.spectrograms))
        else:
            dataset_range = range(len(self.spectrograms) // 10)

        self.spectrograms = self.spectrograms[dataset_range]

        # Augmentation
        self.augmentor = augmentor

    def __getitem__(self, index):

        # Load file as image
        image = Image.open(self.spectrograms[index])
        
        # Define a transform to convert PIL 
        # image to a Torch tensor
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        
        # Convert the PIL image to Torch tensor
        spectrogram_tensor = transform(image)

        # Augment the tensor
        if self.augmentor is not None:
            spectrogram_tensor = self.augmentor(spectrogram_tensor)

        # Return the tensor
        return spectrogram_tensor

    def __len__(self):
        return len(self.spectrograms)


