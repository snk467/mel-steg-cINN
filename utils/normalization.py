import numpy as np
from tqdm import tqdm

import utils.logger

logger = utils.logger.get_logger(__name__)


def scale_minmax(x, min_value=0.0, max_value=1.0):
    x_std = (x - x.min()) / (x.max() - x.min())
    x_scaled = x_std * (max_value - min_value) + min_value
    return x_scaled


def scale_global_minmax(x, global_min=0.0, global_max=1.0, min_value=0.0, max_value=1.0):
    x_std = (x - global_min) / (global_max - global_min)
    x_scaled = x_std * (max_value - min_value) + min_value

    # For numeric safety
    x_scaled = np.clip(x_scaled, a_min=min_value, a_max=max_value)
    return x_scaled


def calculate_statistics(melspectrograms):
    n = 0
    sum_1 = 0.0
    sum_2 = 0.0

    for melspectrogram in tqdm(melspectrograms, leave=False, desc="Calculating statistics"):
        melspectrogram = melspectrogram.mel_spectrogram_data

        n += melspectrogram.size
        sum_1 += np.sum(melspectrogram)
        sum_2 += np.sum(melspectrogram ** 2)

    return n, sum_1, sum_2


def calculate_minmax(melspectrograms):
    min_value = np.Inf
    max_value = -np.Inf

    for melspectrogram in tqdm(melspectrograms, leave=False, desc="Calculating statistics"):
        melspectrogram = melspectrogram.mel_spectrogram_data
        min_value = np.min([min_value, np.min(melspectrogram)])
        max_value = np.max([min_value, np.max(melspectrogram)])

    return min_value, max_value


def normalize_wave_form(sample):
    return (sample - np.mean(sample)) / np.std(sample)


def normalize(melspectrogram, mean, standard_deviation, inverse=False):
    if inverse:
        return melspectrogram * standard_deviation + mean
    else:
        assert standard_deviation != 0
        return (melspectrogram - mean) / standard_deviation
