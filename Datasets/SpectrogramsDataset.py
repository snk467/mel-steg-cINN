import random
from torch.utils.data import Dataset
from Utilities import *
from Normalization import *
import torch
from PIL import Image
import torchvision.transforms as transforms
from zipfile import ZipFile
import zipfile
        

class SpectrogramsDataset(Dataset):

    def __init__(self, dataset_directory, train, device="cpu", colormap="parula_norm_lab", augmentor=None):
        files = get_files(dataset_directory)
        self.dataset_directory = dataset_directory
        self.device = device

        # Filter spectrograms from files
        self.spectrograms_L_channel = list(filter(lambda k: f"spectrogram_L_channel_{colormap}" in k, files))  
        self.labels = list(filter(lambda k: f"labels_{colormap}" in k, files))

        if train:
            dataset_range = slice(len(self.spectrograms_L_channel) // 10, len(self.spectrograms_L_channel))
        else:
            dataset_range = slice(len(self.spectrograms_L_channel) // 10)

        self.spectrograms_L_channel = self.spectrograms_L_channel[dataset_range]

        # Augmentation
        self.augmentor = augmentor

    def __getitem__(self, index):  

        # Load tensors
        L_channel_tensor = self.__load_tensor(self.spectrograms_L_channel[index])
        labels_tensor = self.__load_tensor(self.labels[index])
        
        # Augment tensor
        if self.augmentor is not None:
            L_channel_tensor = self.augmentor(L_channel_tensor)

        # Return tensors
        return L_channel_tensor, labels_tensor

    def __load_tensor(self, zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_directory)

        tensor_path = zip_path.replace(".zip", ".pt")
        return torch.load(tensor_path, map_location=self.device)

    def __len__(self):
        return len(self.spectrograms_L_channel)


