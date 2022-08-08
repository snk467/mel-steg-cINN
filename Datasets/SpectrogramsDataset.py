import random
from torch.utils.data import Dataset
from Utilities import *
from Normalization import *
import torch
from PIL import Image
import torchvision.transforms as transforms
        

class SpectrogramsDataset(Dataset):

    def __init__(self, dataset_directory, train, colormap="parula_norm_lab", augmentor=None):
        # self.info_printed = False

        files = get_files(dataset_directory)

        #print(dataset_directory)

        # Filter spectrograms from files
        self.spectrograms = list(filter(lambda k: f"spectrogram_color_{colormap}" in k, files))  
        self.colormap = Colormap.from_colormap(colormap)

        #print(len(self.spectrograms))

        if train:
            dataset_range = slice(len(self.spectrograms) // 10, len(self.spectrograms))
        else:
            dataset_range = slice(len(self.spectrograms) // 10)

        self.spectrograms = self.spectrograms[dataset_range]

        # Augmentation
        self.augmentor = augmentor

    def __transform_color_mapping(self, color_mapping):
        transformed = torch.zeros((self.colormap.get_colors_length(), color_mapping.shape[0], color_mapping.shape[1]))
        for x in range(color_mapping.shape[0]):
            for y in range(color_mapping.shape[1]):
                transformed[color_mapping[x, y], x, y] = 1.0

        return transformed

    def __getitem__(self, index):

        # Load spectrogram from .npy
        with open(self.spectrograms[index], 'rb') as file:
            spectrogram_data = np.load(file)      

        L_tensor = torch.from_numpy(spectrogram_data[:,:,0])
        L_tensor = torch.reshape(L_tensor, (1, L_tensor.shape[0], L_tensor.shape[1]))
        
        
        color_mapping_tensor = torch.from_numpy(self.colormap.get_indexes_from_colors(spectrogram_data))
        color_mapping_tensor = self.__transform_color_mapping(color_mapping_tensor)

        # if self.info_printed:
        #     print("L_tensor.shape:", L_tensor.shape)
        #     print("color_mapping_tensor.shape:", color_mapping_tensor.shape)
        #     self.info_printed = True

        # Augment the tensor
        if self.augmentor is not None:
            L_tensor = self.augmentor(L_tensor)

        # Return the tensor
        return L_tensor, color_mapping_tensor

    def __len__(self):
        return len(self.spectrograms)


