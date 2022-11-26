from torch.utils.data import Dataset
from helpers.utilities import *
from helpers.normalization import *
import torch
import helpers.logger
import h5py
import torch.nn.functional as F
        
# Get logger
logger = helpers.logger.get_logger(__name__)


class SpectrogramsDataset(Dataset):

    def __init__(self, dataset_location, train, colormap="parula_norm_lab", augmentor=None, size=None, output_dim=None):

        self.__dataset_file = h5py.File(dataset_location, 'r')
        self.dataset = self.__dataset_file["melspectrograms"]
        self.output_dim = output_dim
        
        if size is not None and size < len(self.dataset):
            self.indexes = list(range(size))
        else:
            self.indexes = list(range(len(self.dataset)))
            
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
        L = torch.from_numpy(self.dataset[self.indexes[index], :, :, 0])
        ab = torch.from_numpy(self.dataset[self.indexes[index], :, :, [1,2]])

        # Adjust axies 
        L = torch.reshape(L, (1, L.shape[0], L.shape[1]))
        ab = torch.permute(ab, (2, 0, 1))
        
        if self.output_dim is not None:
            L = F.interpolate(L[None, :], size=self.output_dim)[0]
            ab = F.interpolate(ab[None, :], size=self.output_dim)[0]
        
        # Augment tensor
        L_clear = L
        if self.augmentor is not None:
            L = self.augmentor(L)

        # Prepare label
        label = f"melspectrogram_{str(index).zfill(5)}",

        # Return tensors(L_noise, ab, string_name, L_clear)
        return L, ab, label, L_clear        

    def __len__(self):
        return len(self.indexes)