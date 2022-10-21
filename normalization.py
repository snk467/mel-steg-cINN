import numpy as np
from tqdm import tqdm
import logger as logger_module

logger = logger_module.get_logger(__name__)

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
   
    means = []
    standard_deviations = []
    mins = []
    maxs = []

    for melspectrogram in tqdm(melspectrograms, leave=False, desc="Calculating statistics"):
        melspectrogram = melspectrogram.mel_spectrogram_data
        means.append(np.mean(melspectrogram))
        standard_deviations.append(np.std(melspectrogram))
        mins.append(np.min(melspectrogram))
        maxs.append(np.max(melspectrogram))   

    return means, standard_deviations, mins, maxs

def normalize_wave_form(sample):
    return (sample - np.mean(sample)) / np.std(sample)

def normalize(melspectrogram, mean, standard_deviation, inverse=False):
    if inverse:
        return melspectrogram*standard_deviation + mean
    else:
        return (melspectrogram-mean) / standard_deviation