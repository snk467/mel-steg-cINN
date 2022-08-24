import random
from torch.utils.data import Dataset
from Utilities import *
from Normalization import *
import torch
from PIL import Image
import torchvision.transforms as transforms
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
        self.spectrograms_L_channel = sorted(list(filter(lambda k: f"spectrogram_L_channel_{colormap}" in k and k.endswith(".zip"), files)), key = lambda filename: int(filename.replace(".", "_").split("_")[-2]))
        self.labels = sorted(list(filter(lambda k: f"labels_{colormap}" in k and k.endswith(".zip"), files)), key = lambda filename: int(filename.replace(".", "_").split("_")[-2]))         
         
        if size is not None and size < len(self.spectrograms_L_channel):
            self.spectrograms_L_channel = self.spectrograms_L_channel[:size]
            self.labels = self.labels[:size]
            
        if train:
            dataset_range = slice(len(self.spectrograms_L_channel) // 10, len(self.spectrograms_L_channel))
            labels_range = slice(len(self.labels) // 10, len(self.labels))
        else:
            dataset_range = slice(len(self.spectrograms_L_channel) // 10)
            labels_range = slice(len(self.labels) // 10)            

        self.spectrograms_L_channel = self.spectrograms_L_channel[dataset_range]
        self.labels = self.labels[dataset_range]
        
        if train:            
            logger.info(f"Train dataset size: {self.__len__()}")            
        else:
            logger.info(f"Validation dataset size: {self.__len__()}")            

        # Augmentation
        self.augmentor = augmentor

    def __getitem__(self, index):  

        # Load tensors
        L_channel_tensor = self.__load_tensor(os.path.join(self.dataset_directory , self.spectrograms_L_channel[index]))
        labels_tensor = self.__load_tensor(os.path.join(self.dataset_directory , self.labels[index]))
        
        # Augment tensor
        if self.augmentor is not None:
            L_channel_tensor = self.augmentor(L_channel_tensor)

        # Return tensors
        return L_channel_tensor, labels_tensor, os.path.basename(self.spectrograms_L_channel[index])

    def __load_tensor(self, zip_path):
        tensor_path = zip_path.replace(".zip", ".pt")
        if os.path.exists(tensor_path):
            os.remove(tensor_path)
                             
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_directory)

        tensor = torch.load(tensor_path)
                             
        os.remove(tensor_path)       
        return tensor
        

    def __len__(self):
        return len(self.spectrograms_L_channel)
