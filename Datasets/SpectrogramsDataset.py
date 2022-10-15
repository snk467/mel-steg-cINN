import random
from torch.utils.data import Dataset
from Utilities import *
from Normalization import *
import torch
import gzip
import Logger
from zipfile import ZipFile
import zipfile
import h5py
        
# Get logger
logger = Logger.get_logger(__name__)


class SpectrogramsDataset(Dataset):

    def __init__(self, dataset_location, train, colormap="parula_norm_lab", augmentor=None, size=None):

        self.__dataset_file = h5py.File(dataset_location, 'r')
        self.dataset = self.__dataset_file["melspectrograms"]
        
        if size is not None and size < len(self.dataset):
            self.indexes = list(range(size))
        else:
            self.indexes = list(range(len(self.dataset)))

        print(len(self.indexes))
            
        if train:
            dataset_range = slice(len(self.indexes) // 10, len(self.indexes))
        else:
            dataset_range = slice(len(self.indexes) // 10)        

        self.indexes = self.indexes[dataset_range]
        
        if train:            
            logger.info(f"Train dataset size: {self.__len__()}")            
        else:
            logger.info(f"Validation dataset size: {self.__len__()}")            

        # Augmentation
        self.augmentor = augmentor

    def __getitem__(self, index):  

        # Load tensors
        input = torch.from_numpy(self.dataset[self.indexes[index], :, :, 0])
        target = torch.from_numpy(self.dataset[self.indexes[index], :, :, [1,2]])

        # Adjust axies 
        input = torch.reshape(input, (1, input.shape[0], input.shape[1]))
        target = torch.permute(target, (2, 0, 1))
        
        # Augment tensor
        clear_input = input
        if self.augmentor is not None:
            input = self.augmentor(input)

        # Prepare label
        label = f"melspectrogram_{str(index).zfill(5)}",

        # Return tensors(L_noise, ab, string_name, L_clear)
        return input, target, label, clear_input        

    def __len__(self):
        return len(self.indexes)
    
    def release_resources(self):
        self.__dataset_file.close()
