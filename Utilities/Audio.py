import numpy as np
import random
import librosa
import logging
from LUT import Colormap
logger = logging.getLogger(__name__)
from Exceptions import ArgumentError
import Normalization
from Utilities.MelSpectorgram import MelSpectrogram

class Audio:

    __supported_color_representations = ["rgb"]

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

    def calculate_mel_spectrogram(self, normalized=False, range=None) -> MelSpectrogram:
        """
        
        Calculate mel-spectorgram of audio waveform. 
        The spectrogram can be scaled to a given range and normalized if requested.

        """
        spectrogram = librosa.feature.melspectrogram(y=self.audio, 
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

    def calculate_color_mel_spectrogram(self, normalized=False, color="rgb", colormap="parula") -> MelSpectrogram:
        """
        
        Calculate image-represented mel-spectrogram of audio waveform. 
        The spectogram is scaled to [0;1] range and normalized if requested.
        Color representation and color map can be specified.
        
        """

        mel_spectrogram = self.calculate_mel_spectrogram(normalized, (0.0, 1.0))
        mel_spectrogram_data = mel_spectrogram.mel_spectrogram_data

        if colormap is not None:
            colormap = Colormap.from_colormap(colormap)
            mel_spectrogram_data = colormap.get_color_from_2D_array(mel_spectrogram_data)

        if color in Audio.__supported_color_representations:
            pass

        return MelSpectrogram.from_color(mel_spectrogram_data, normalized, color, colormap, self.config)