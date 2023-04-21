import random

import librosa
import numpy as np
import torch
from hurry.filesize import size
from pynvml import *

import utils.logger
import utils.normalization as normalization
from LUT import Colormap
from exceptions import ArgumentError

logger = utils.logger.get_logger(__name__)


class Audio:

    def __init__(self, audio, config):
        # Take segment - pad if too small
        self.config = config
        segment_length = self.config.segment_length

        if audio.shape[0] >= segment_length:
            max_audio_start = audio.shape[0] - segment_length
            audio_start = random.randint(0, max_audio_start)
            self.audio = audio[audio_start:audio_start + segment_length]
        else:
            self.audio = np.pad(audio, (0, segment_length - audio.shape[0]), 'constant')

    def get_audio(self):
        return self.audio

    def get_mel_spectrogram(self, normalized=False, values_range=None):
        """
        
        Calculate mel-spectrogram of audio waveform.
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

        if values_range is not None and self.config.global_min is not None and self.config.global_max is not None:
            if len(values_range) != 2:
                logger.error("Invalid range shape!")
                raise ArgumentError
            mel_spectrogram = normalization.scale_global_minmax(mel_spectrogram, self.config.global_min,
                                                                self.config.global_max, min(values_range),
                                                                max(values_range))

        return MelSpectrogram.from_value(mel_spectrogram, normalized, values_range, self.config)

    def get_color_mel_spectrogram(self, normalized=False, colormap: Colormap = None):
        """
        
        Calculate image-represented mel-spectrogram of audio waveform. 
        The spectrogram is scaled to [0;1] range and normalized if requested.
        Color representation and color map can be specified.
        
        """

        mel_spectrogram = self.get_mel_spectrogram(normalized, (0.0, 1.0))
        mel_spectrogram_data = mel_spectrogram.mel_spectrogram_data

        if colormap is not None:
            mel_spectrogram_data = colormap.get_colors_from_values(mel_spectrogram_data)

        return MelSpectrogram.from_color(mel_spectrogram_data, normalized, colormap, self.config)


class MelSpectrogram:
    def __init__(self, mel_spectrogram_data, normalized, config, range=None, colormap: Colormap = None):
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
        return cls(mel_spectrogram_data, normalized, config, range=(0.0, 1.0), colormap=colormap)

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
            mel_spectrogram_data = normalization.scale_global_minmax(mel_spectrogram_data, min(self.range),
                                                                     max(self.range), self.config.global_min,
                                                                     self.config.global_max)

        # Inverse normalization
        if self.normalized:
            mel_spectrogram_data = normalization.normalize(mel_spectrogram_data, self.config.mean,
                                                           self.config.standard_deviation, inverse=True)

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


def test_cuda(verbose=True):
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
    is_cuda = test_cuda(verbose)

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


def get_L_channel(melspectrogram: MelSpectrogram):
    # Load tensor
    L = torch.from_numpy(melspectrogram.mel_spectrogram_data[:, :, 0])
    # Adjust axes
    L = torch.reshape(L, (1, 1, L.shape[0], L.shape[1]))
    return L


def get_melspectrogram_tensor(melspectrogram: MelSpectrogram):
    # Load tensor
    spectrogram_data = torch.from_numpy(melspectrogram.mel_spectrogram_data)
    # Adjust axes
    spectrogram_data = torch.permute(spectrogram_data, (2, 0, 1))[None, :]

    return spectrogram_data


def get_cond(l_channel: torch.Tensor, cinn_utilities):
    with torch.no_grad():
        features, _ = cinn_utilities.model.feature_network.features(l_channel)
        return [*features]


def sample_z(out_shapes, batch_size, alpha=None, device=get_device(verbose=False)):
    def get_value_out_of_range():
        value = 0.0
        while np.abs(value) < alpha:
            value = np.random.normal(loc=0.0, scale=1.0)
        return value

    samples = []

    for out_shape in out_shapes:
        sample = torch.normal(mean=0.0, std=1.0, size=(batch_size, out_shape), device=device)

        if alpha is not None:
            sample = torch.where(torch.abs(sample) > torch.tensor(alpha), sample,
                                 torch.tensor(get_value_out_of_range(), device=device))

        samples.append(sample)

    return samples
