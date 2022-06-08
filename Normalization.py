import imp
import numpy as np
from tqdm import tqdm
import Logger
from Utilities import MelSpectrogram

logger = Logger.get_logger(__name__)

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def scale_global_minmax(X, global_min=0.0, global_max=1.0, min=0.0, max=1.0):
    X_std = (X - global_min) / (global_max - global_min)
    X_scaled = X_std * (max - min) + min

    X_scaled = np.clip(X_scaled, a_min=min, a_max=max)
    return X_scaled

def calculate_statistics(melspectrograms: list[MelSpectrogram]):
    count = 0
    logger.info("Calculating statistics.")
    for melspectrogram in tqdm(melspectrograms, leave=False):
        melspectrogram = melspectrogram.mel_spectrogram_data
        if count == 0:
            mean = np.mean(melspectrogram)
            standard_deviation = np.std(melspectrogram)
            min = np.min(melspectrogram)
            max = np.max(melspectrogram)
        else:            
            current_mean = np.mean(melspectrogram)
            current_standard_deviation = np.std(melspectrogram)
            min = np.min([min, np.min(melspectrogram)])
            max = np.max([max, np.max(melspectrogram)])

            old_mean = mean
            m = count * 1.0
            n = 1.0
            mean = m/(m+n)*old_mean + n/(m+n)*current_mean
            standard_deviation  = m/(m+n)*standard_deviation**2 + n/(m+n)*current_standard_deviation**2 +\
                        m*n/(m+n)**2 * (old_mean - current_mean)**2
            standard_deviation = np.sqrt(standard_deviation)
        count += 1

    logger.info("Statistics calculated.")

    return mean, standard_deviation, min, max

def normalize_wave_form(sample):
    return (sample - np.mean(sample)) / np.std(sample)

def normalize(melspectrogram, mean, standard_deviation, inverse=False):
    if inverse:
        return melspectrogram*standard_deviation + mean
    else:
        return (melspectrogram-mean) / standard_deviation

