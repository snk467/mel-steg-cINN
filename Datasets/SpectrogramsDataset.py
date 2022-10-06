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

        dataset_file = h5py.File(dataset_location, 'r')
        self.dataset = dataset_file["melspectrograms"]
        
        if size is not None and size < len(self.inputs):
            self.inputs = self.dataset[:size, :, :, 0]
            self.targets = self.dataset[:size, :, :, [1,2]]
        else:
            self.inputs = self.dataset[:, :, :, 0]
            self.targets = self.dataset[:, :, :, [1,2]]

        print(self.inputs.shape)
        print(self.targets.shape)
            
        if train:
            dataset_range = slice(len(self.inputs) // 10, len(self.inputs))
        else:
            dataset_range = slice(len(self.inputs) // 10)        

        self.inputs = self.inputs[dataset_range]
        self.targets = self.targets[dataset_range]
        
        if train:            
            logger.info(f"Train dataset size: {self.__len__()}")            
        else:
            logger.info(f"Validation dataset size: {self.__len__()}")            

        # Augmentation
        self.augmentor = augmentor

    def __getitem__(self, index):  

        # Load tensors
        input = torch.from_numpy(self.inputs[index])
        target = torch.from_numpy(self.targets[index])

        # Adjust axies 
        input = torch.reshape(input, (1, input.shape[0], input.shape[1]))
        target = torch.permute(target, (2, 0, 1))
        
        # Augment tensor
        clear_input = input
        if self.augmentor is not None:
            input = self.augmentor(input)

        # Prepare label
        label = f"melspectrogram_{str(index).zfill(5)}",

        # Return tensors
        return input, target, label, clear_input        

    def __len__(self):
        return len(self.inputs)
