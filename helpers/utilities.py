import os
import bitarray
import librosa
import random

import wandb
import helpers.logger
import helpers.normalization as normalization
import numpy as np
from exceptions import ArgumentError
from LUT import Colormap
import torch
from pynvml import *
from hurry.filesize import size
import gc
import cinn.cinn_model
import mel_steg_cinn_config

logger = helpers.logger.get_logger(__name__)

class Audio:

    def __init__(self, audio, config):
        # Take segment - pad if too small
        self.config = config
        segment_length = self.config.segment_length

        if audio.shape[0] >= segment_length:
            max_audio_start = audio.shape[0] - segment_length
            audio_start = random.randint(0, max_audio_start)
            self.audio = audio[audio_start:audio_start+segment_length]
        else:
            self.audio = np.pad(audio, (0, segment_length - audio.shape[0]), 'constant')

    def get_audio(self):
        return self.audio

    def get_mel_spectrogram(self, normalized=False, range=None): 
        """
        
        Calculate mel-spectorgram of audio waveform. 
        The spectrogram can be scaled to a given range and normalized if requested.

        """
        spectrogram = librosa.feature.melspectrogram(y=self.audio, 
                                                    n_mels=self.config.n_mels,
                                                    sr=self.config.sample_rate,
                                                    n_fft=self.config.window_length,
                                                    hop_length=self.config.hop_length)   

        mel_spectrogram = librosa.power_to_db(spectrogram)

        if normalized and (self.config.mean is not None) and (self.config.standard_deviation is not None):
            mel_spectrogram = normalization.normalize(mel_spectrogram, self.config.mean, self.config.standard_deviation)

        if range is not None and self.config.global_min is not None and self.config.global_max is not None:
            if len(range) != 2:
                logger.error("Invalid range shape!");
                raise ArgumentError
            mel_spectrogram = normalization.scale_global_minmax(mel_spectrogram, self.config.global_min, self.config.global_max, min(range), max(range))

        return MelSpectrogram.from_value(mel_spectrogram, normalized, range, self.config)

    def get_color_mel_spectrogram(self, normalized=False, colormap:Colormap=None):
        """
        
        Calculate image-represented mel-spectrogram of audio waveform. 
        The spectogram is scaled to [0;1] range and normalized if requested.
        Color representation and color map can be specified.
        
        """

        mel_spectrogram = self.get_mel_spectrogram(normalized, (0.0, 1.0))
        mel_spectrogram_data = mel_spectrogram.mel_spectrogram_data

        if colormap is not None:
            mel_spectrogram_data = colormap.get_colors_from_values(mel_spectrogram_data)

        return MelSpectrogram.from_color(mel_spectrogram_data, normalized, colormap, self.config)

class MelSpectrogram:
    def __init__(self, mel_spectrogram_data, normalized, config, range=None, colormap:Colormap=None):
        self.mel_spectrogram_data = mel_spectrogram_data
        self.normalized = normalized
        self.range = range 
        if colormap is not None and type(colormap) != Colormap:
            raise ArgumentError()
        self.colormap = colormap
        self.config = config
        self.audio = None        

    @classmethod
    def from_color(cls, mel_spectrogram_data, normalized, colormap, config):
        return cls(mel_spectrogram_data, normalized, config, range=(0.0,1.0), colormap=colormap)

    @classmethod
    def from_value(cls, mel_spectrogram_data, normalized, range, config):
        return cls(mel_spectrogram_data, normalized, config, range=range, colormap=None)

    def get_audio(self):
        
        if self.audio is not None:
            return self.audio

        mel_spectrogram_data = self.mel_spectrogram_data

        if mel_spectrogram_data is None:
            return None

        # Convert color back to values
        if self.colormap is not None:
            # colormap = Colormap.from_colormap(self.colormap)
            mel_spectrogram_data = self.colormap.get_values_from_colors(mel_spectrogram_data)

        # Inverse scaling
        if self.range is not None:
            mel_spectrogram_data = normalization.scale_global_minmax(mel_spectrogram_data, min(self.range), max(self.range), self.config.global_min, self.config.global_max)

        # Inverse normalization
        if self.normalized:
            mel_spectrogram_data = normalization.normalize(mel_spectrogram_data, self.config.mean, self.config.standard_deviation, inverse=True)

        # Inverse mel spectrogram
        mel_spectrogram_data = librosa.db_to_power(mel_spectrogram_data)
        audio = librosa.feature.inverse.mel_to_audio(M=mel_spectrogram_data, 
                                                    sr=self.config.sample_rate,
                                                    n_fft=self.config.window_length,
                                                    hop_length=self.config.hop_length,
                                                    n_iter=60)

        return Audio(audio, self.config)

def get_files(files_directory):
    files = []
    for filename in os.listdir(files_directory):
        path = os.path.join(files_directory, filename)
        if os.path.isfile(path):
            files.append(path)

    return np.array(files)

def load_audio(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    return audio, sample_rate

def test_CUDA(verbose=True):
    if torch.cuda.is_available():
        if verbose:
            logger.info("PyTorch is running on CUDA!")
            logger.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
            device_id = torch.cuda.current_device()
            logger.info(f"Device ID: {device_id}")
            logger.info(f"Device name: {torch.cuda.get_device_name(device_id)}")
        return True
    else:
        if verbose:
            logger.warning("PyTorch is not running on CUDA!")   
        return False

def get_device(verbose=True):
    is_cuda = test_CUDA(verbose)
    
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')   

    return device

def print_gpu_memory_usage():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    logger.info("Memory usage:")
    logger.info(f'total    : {size(info.total)}')
    logger.info(f'free     : {size(info.free)}')
    logger.info(f'used     : {size(info.used)}')
    
def wipe_memory(cinn_training_utilities, models):
    logger.info("Freeing memory")
    print_gpu_memory_usage()
    _optimizer_to(cinn_training_utilities.optimizer, torch.device('cpu'))    
    del cinn_training_utilities.optimizer
    for model in models:
        model.to(torch.device('cpu'))
        del model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Result:")
    print_gpu_memory_usage()

def _optimizer_to(optimizer, device):
    for param in optimizer.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)




class MelStegCinn:
    def __init__(self, config: mel_steg_cinn_config.Config):
        self.config = config
        self.cinn_utilities = None
        self.cinn_z_dimensions = None
        pass
    
    def load_cinn(self):
        cinn_builder = cinn.cinn_model.cINN_builder(self.config.cinnConfig)
        feature_net = cinn_builder.get_feature_net().double()
        fc_cond_net = cinn_builder.get_fc_cond_net().double()
        cinn_net, self.cinn_z_dimensions = cinn_builder.get_cinn()
        cinn_model = cinn.cinn_model.WrappedModel(feature_net, fc_cond_net, cinn_net.double())
        self.cinn_utilities = cinn.cinn_model.cINNTrainingUtilities(cinn_model, self.config.cinnConfig)
        self.cinn_utilities.load(self.config.model_path, device='cpu') 
        
        return self.cinn_utilities.model
        
    def load_audio(self, path: str):
        return Audio(load_audio(path)[0], self.config.audio_parameters)
    
    def encode(self, message: str):
        bin_message = bitarray.bitarray()
        bin_message.frombytes((message + self.config.end_of_message_string).encode('ascii'))
        
        desired_size =  sum([x for x in self.cinn_z_dimensions])
        
        z = []
        
        for bit in bin_message:
            sample = np.random.normal()
            if bit == 0:
                while not (sample < -np.abs(self.config.alpha)):
                    sample = np.random.normal() 
            else:
                while not (sample > np.abs(self.config.alpha)):
                    sample = np.random.normal()
                    
            z.append(sample) 
            
            if len(z) == desired_size:
                break
            
        if len(z) != desired_size:
            z.extend(np.random.normal(size=desired_size-len(z)))
            
        logger.debug(f"Z length: {len(z)}")
        
        # plt.hist(z, bins=100)
        # plt.show()
        
        z = torch.from_numpy(np.array(z))
        z = list(z.split(self.cinn_z_dimensions))
        
        logger.debug(f"Z length after split: {len(z)}")
        for i in range(len(z)):
            logger.debug(f"z[{i}].shape: {z[i].shape}")
            z[i] = z[i][None, :]
            logger.debug(f"z[{i}].shape(corrected): {z[i].shape}")
            
        logger.info(f"Encoded message: {message}")
        
        return z
    
    def decode(self, z):
        
        logger.debug(f"Z input length: {len(z)}")
        z = torch.cat(z, dim=1).squeeze().detach().numpy()
        
        # plt.hist(z, bins=100)
        # plt.show()
        
        logger.debug(f"Z concatenated shape: {z.shape}")
        
        bin_message = []
        
        for sample in z:
            if sample < -np.abs(self.config.alpha):
                bin_message.append(False)
            elif sample > np.abs(self.config.alpha):
                bin_message.append(True)
            
        message_with_noise = bitarray.bitarray(bin_message).tobytes().decode('ascii', errors='replace')
        message = message_with_noise.split(self.config.end_of_message_string)[0]
        logger.info(f"Decoded message: {message[:100]}")        
    
    def get_L_channel(self, melspectrogram: MelSpectrogram):
        # Load tensor
        L = torch.from_numpy(melspectrogram.mel_spectrogram_data[:, :, 0])
        # Adjust axies 
        L = torch.reshape(L, (1, 1, L.shape[0], L.shape[1]))
        return L
    
    def get_melspectrogram_tensor(self, melspectrogram: MelSpectrogram):
        # Load tensor
        spectrogram_data = torch.from_numpy(melspectrogram.mel_spectrogram_data)
        # Adjust axies 
        spectrogram_data = torch.permute(spectrogram_data, (2, 0, 1))[None, :]
        
        return spectrogram_data
        
    
    def get_cond(self, L_channel: torch.Tensor):
        with torch.no_grad():
            features, _ = self.cinn_utilities.model.feature_network.features(L_channel)
            return [*features]
       
def get_cinn_model(cinn_training_config, filename=None, run_path=None, device='cpu'):
    cinn_builder = cinn.cinn_model.cINN_builder(cinn_training_config)
    
    feature_net = cinn_builder.get_feature_net()
    fc_cond_net = cinn_builder.get_fc_cond_net()
    cinn_net, cinn_output_dimensions = cinn_builder.get_cinn()
            
    cinn_model = cinn.cinn_model.WrappedModel(feature_net, fc_cond_net, cinn_net).to(device)
    cinn_utilities = cinn.cinn_model.cINNTrainingUtilities(cinn_model.float(), cinn_training_config) 
    
    if run_path is not None:
        restored_model = wandb.restore(filename, run_path=run_path)
        cinn_utilities.load(restored_model.name, device=device) 
        
    return cinn_utilities, cinn_output_dimensions 
             
    
# def load_audio(path: str, audio_parameters):
#     return Audio(load_audio(path)[0], audio_parameters)

def get_L_channel(melspectrogram: MelSpectrogram):
    # Load tensor
    L = torch.from_numpy(melspectrogram.mel_spectrogram_data[:, :, 0])
    # Adjust axies 
    L = torch.reshape(L, (1, 1, L.shape[0], L.shape[1]))
    return L

def get_melspectrogram_tensor(melspectrogram: MelSpectrogram):
    # Load tensor
    spectrogram_data = torch.from_numpy(melspectrogram.mel_spectrogram_data)
    # Adjust axies 
    spectrogram_data = torch.permute(spectrogram_data, (2, 0, 1))[None, :]
    
    return spectrogram_data
    
def get_cond(L_channel: torch.Tensor, cinn_utilities):
    with torch.no_grad():
        features, _ = cinn_utilities.model.feature_network.features(L_channel)
        return [*features]