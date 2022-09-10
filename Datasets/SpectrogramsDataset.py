import random
from torch.utils.data import Dataset
from Utilities import *
from Normalization import *
import torch
from PIL import Image
import torchvision.transforms as transforms
import gzip
import Logger
from zipfile import ZipFile
import zipfile
        
# Get logger
logger = Logger.get_logger(__name__)


class SpectrogramsDataset(Dataset):

    def __init__(self, dataset_directory, train, colormap="parula_norm_lab", augmentor=None, size=None):
        files = get_files(dataset_directory)
        self.dataset_directory = dataset_directory

        # Filter spectrograms from files
        self.inputs = sorted(list(filter(lambda k: f"spectrogram_L_channel_{colormap}" in k and k.endswith(".gz"), files)), key = lambda filename: int(filename.replace(".", "_").split("_")[-3]))
        self.targets = sorted(list(filter(lambda k: f"spectrogram_ab_channels_{colormap}" in k and k.endswith(".gz"), files)), key = lambda filename: int(filename.replace(".", "_").split("_")[-3]))         
         
        if size is not None and size < len(self.inputs):
            self.inputs = self.inputs[:size]
            self.targets = self.targets[:size]
            
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
        input = self.__load_tensor(os.path.join(self.dataset_directory , self.inputs[index]))
        target = self.__load_tensor(os.path.join(self.dataset_directory , self.targets[index]))
        
        # Augment tensor
        clear_input = input
        if self.augmentor is not None:
            input = self.augmentor(input)

        # Return tensors
        return input, target, os.path.basename(self.inputs[index]), clear_input

    def __load_tensor(self, compressed_tensor_path):

        with gzip.open(compressed_tensor_path, 'rb') as file:
            tensor = torch.load(file)        
                                 
        return tensor
        

    def __len__(self):
        return len(self.inputs)
