import numpy as np
from tqdm import tqdm
import helpers.logger

logger = helpers.logger.get_logger(__name__)

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def scale_global_minmax(X, global_min=0.0, global_max=1.0, min=0.0, max=1.0):
    X_std = (X - global_min) / (global_max - global_min)
    X_scaled = X_std * (max - min) + min

    # For numeric safety
    X_scaled = np.clip(X_scaled, a_min=min, a_max=max)
    return X_scaled

def calculate_statistics(melspectrograms):
    n = 0
    sum = 0.0
    sum_2 = 0.0

    for melspectrogram in tqdm(melspectrograms, leave=False, desc="Calculating statistics"):
        melspectrogram = melspectrogram.mel_spectrogram_data
        
        n += melspectrogram.size
        sum += np.sum(melspectrogram)
        sum_2 += np.sum(melspectrogram**2)
         
    return n, sum, sum_2

def calculate_minmax(melspectrograms):
    min = np.Inf
    max = -np.Inf

    for melspectrogram in tqdm(melspectrograms, leave=False, desc="Calculating statistics"):
        melspectrogram = melspectrogram.mel_spectrogram_data
        min = np.min([min, np.min(melspectrogram)])
        max = np.max([min, np.max(melspectrogram)])
         
    return min, max

def normalize_wave_form(sample):
    return (sample - np.mean(sample)) / np.std(sample)

def normalize(melspectrogram, mean, standard_deviation, inverse=False):
    if inverse:
        return melspectrogram*standard_deviation + mean
    else:
        assert standard_deviation != 0
        return (melspectrogram-mean) / standard_deviation