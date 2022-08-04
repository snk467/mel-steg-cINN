import os
import librosa
import random
import logging
import Normalization
import numpy as np
from Exceptions import ArgumentError
from LUT import Colormap
from skimage import color

logger = logging.getLogger(__name__)

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
            mel_spectrogram = Normalization.normalize(mel_spectrogram, self.config.mean, self.config.standard_deviation)

        if range is not None and self.config.global_min is not None and self.config.global_max is not None:
            if len(range) != 2:
                logger.error("Invalid range shape!");
                raise ArgumentError
            mel_spectrogram = Normalization.scale_global_minmax(mel_spectrogram, self.config.global_min, self.config.global_max, min(range), max(range))

        return MelSpectrogram.from_value(mel_spectrogram, normalized, range, self.config)

    def get_color_mel_spectrogram(self, normalized=False, colormap=None):
        """
        
        Calculate image-represented mel-spectrogram of audio waveform. 
        The spectogram is scaled to [0;1] range and normalized if requested.
        Color representation and color map can be specified.
        
        """

        mel_spectrogram = self.get_mel_spectrogram(normalized, (0.0, 1.0))
        mel_spectrogram_data = mel_spectrogram.mel_spectrogram_data

        if colormap is not None:
            color_map = Colormap.from_colormap(colormap)
            mel_spectrogram_data = color_map.get_color_from_2D_array(mel_spectrogram_data)

        return MelSpectrogram.from_color(mel_spectrogram_data, normalized, colormap, self.config)

class MelSpectrogram:
    def __init__(self, mel_spectrogram_data, normalized, config, range=None, colormap=None):
        self.mel_spectrogram_data = mel_spectrogram_data
        self.normalized = normalized
        self.range = range 
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
            colormap = Colormap.from_colormap(self.colormap)
            mel_spectrogram_data = colormap.get_value_from_image(mel_spectrogram_data)

        # Inverse scaling
        if self.range is not None:
            mel_spectrogram_data = Normalization.scale_global_minmax(mel_spectrogram_data, min(self.range), max(self.range), self.config.global_min, self.config.global_max)

        # Inverse normalization
        if self.normalized:
            mel_spectrogram_data = Normalization.normalize(mel_spectrogram_data, self.config.mean, self.config.standard_deviation, inverse=True)

        # Inverse mel spectrogram
        mel_spectrogram_data = librosa.db_to_power(mel_spectrogram_data)
        audio = librosa.feature.inverse.mel_to_audio(M=mel_spectrogram_data, 
                                                    sr=self.config.sample_rate,
                                                    n_fft=self.config.window_length,
                                                    hop_length=self.config.hop_length)

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




