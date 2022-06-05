class MelSpectrogram:
    def __init__(self, mel_spectrogram_data, normalized, config, range=None, color=None, colormap=None):
        self.mel_spectrogram_data = mel_spectrogram_data
        self.normalized = normalized
        self.range = range
        self.color = color
        self.colormap = colormap
        self.config = config

    @classmethod
    def from_color(cls, mel_spectrogram_data, normalized, color, colormap, config):
        return cls(mel_spectrogram_data, normalized, config, range=(0.0,1.0), color=color, colormap=colormap)

    @classmethod
    def from_value(cls, mel_spectrogram_data, normalized, range, config):
        return cls(mel_spectrogram_data, normalized, config, range=range)

    def get_audio(self):
        raise NotImplementedError


