import argparse
import random
import soundfile
import os
import Exceptions
import Configuration
import Normalization 
import Logger
import matplotlib.pyplot as plt
from Utilities import *
from tqdm import tqdm
logger = Logger.get_logger(__name__)

class AudioDatasetProcessor:
    
    def __init__(self, audo_files_directory, config):
        self.config = config

        self.audio_files = get_audio_files(audo_files_directory)

        self.logger = Logger.get_logger(__name__)

        # Initialize RNG
        random.seed(1234)
        random.shuffle(self.audio_files)

    def load_audio_files(self, audio_files):
        loaded_audio = []

        self.logger.info("Loading audio files.")
        for audio_file in tqdm(audio_files, leave=False):
            audio, sample_rate = load_audio(audio_file)

            if sample_rate != self.config.sample_rate:
                raise Exceptions.SampleRateError

            audio = Audio(audio, self.config)

            loaded_audio.append(audio)

        self.logger.info("Audio files loaded.")
        return loaded_audio

    def get_color_mel_spectrograms(self, loaded_audio, normalized=True, color="rgb", colormap="parula"):
        mel_spectrograms = []
        self.logger.info("Calculating color mel spectrograms.")
        for audio in tqdm(loaded_audio, leave=False):
            mel_spectrograms.append(audio.get_color_mel_spectrogram(normalized=normalized, color=color, colormap=colormap))
        self.logger.info("Color mel spectrograms calculated.")
        return mel_spectrograms

    def get_mel_spectrograms(self, loaded_audio, normalized=True, range=(0.0,1.0)):
        mel_spectrograms = []
        self.logger.info("Calculating mel spectrograms.")
        for audio in tqdm(loaded_audio, leave=False):
            mel_spectrograms.append(audio.get_mel_spectrogram(normalized=normalized, range=range))
        self.logger.info("Mel spectrograms calculated.")
        return mel_spectrograms

    def restore_audio(self, mel_spectrograms):
        audio = []
        self.logger.info("Restoring audio.")
        for mel_spectrogram in tqdm(mel_spectrograms, leave=False):
            audio.append(mel_spectrogram.get_audio())
        self.logger.info("Audio restored.")
        return audio

    def get_statistics(self):
        self.logger.info("Processing audio.")

        loaded_audio = self.load_audio_files(self.audio_files)

        mel_spectrograms = self.get_mel_spectrograms(loaded_audio, normalized=False, range=None)

        mean, standard_deviation, _, _ = Normalization.calculate_statistics(mel_spectrograms)

        self.logger.info("Adjusting statistics.")
        def modify_stats(audio):
            audio.config.mean = mean
            audio.config.standard_deviation = standard_deviation      
            return audio      

        loaded_audio = map(modify_stats, loaded_audio)

        mel_spectrograms = self.get_mel_spectrograms(loaded_audio, normalized=True, range=None)

        _, _, min, max = Normalization.calculate_statistics(mel_spectrograms)

        self.logger.info("Audio processing done.")

        return mean, standard_deviation, min, max


    def process(self):

        self.logger.info("Processing audio.")

        loaded_audio = self.load_audio_files(self.audio_files)

        color_mel_spectrograms = self.get_color_mel_spectrograms(loaded_audio)

        restored_audio = self.restore_audio(color_mel_spectrograms)

        self.logger.info("Audio processing done.")
        return loaded_audio, color_mel_spectrograms, restored_audio

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_dir", type=str, required=True, help='Input directory.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Output directory.')
    parser.add_argument('-s', '--statistics', action="store_true", help='Calculate mel spectrograms statistics.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Get arguments
    args = get_args()

    # Get configuration
    config = Configuration.load()
    logger.info("Configuration loaded.")

    # Get audio processor
    audio_processor = AudioDatasetProcessor(args.input_dir, config.audio_parameters)
    logger.info("Audio processor initialized.")

    if args.statistics:
        mean, standard_deviation, min, max = audio_processor.get_statistics()
        file = open(os.path.join(args.output_dir, "dataset_stats.txt"), "w")
        file.write(f"Mean:{mean}\n")
        file.write(f"Standard deviation:{standard_deviation}\n")
        file.write(f"Min:{min}\n")
        file.write(f"Max:{max}\n")
        file.close()
        logger.info("Statistics saved.")
    else:
        data = audio_processor.process()

        id = 0
        digits = 5
        logger.info("Saving dataset.")
        for audio_data in  tqdm(list(zip(data[0],data[1],data[2])), leave=False):
        
            # Get items
            audio, color_spec, restored_audio = audio_data

            # Get ID
            id_string = str(id).zfill(digits)

            # Save audio
            soundfile.write(os.path.join(args.output_dir, f"audio_{id_string}.wav"), audio.get_audio(), config.audio_parameters.sample_rate)

            # Save color spectrogram
            plt.imsave(os.path.join(args.output_dir, f"spec_color_{id_string}.png"), color_spec.mel_spectrogram_data) 

            # Save audio
            soundfile.write(os.path.join(args.output_dir, f"audio_restored_{id_string}.wav"), restored_audio.get_audio(), config.audio_parameters.sample_rate)      

            id += 1

        logger.info("Dataset saved.")