import argparse
import random
import soundfile
import os

import torch
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

        self.audio_files = get_files(audo_files_directory)

        self.logger = Logger.get_logger(__name__)

        # Initialize RNG
        random.seed(1234)

    def load_audio_files(self, audio_files):
        loaded_audio = []

        for audio_file in tqdm(audio_files, leave=False, desc="Loading audio files"):
            audio, sample_rate = load_audio(audio_file)

            if sample_rate != self.config.sample_rate:
                raise Exceptions.SampleRateError

            audio = Audio(audio, self.config)

            loaded_audio.append(audio)

        return loaded_audio

    def get_color_mel_spectrograms(self, loaded_audio, normalized=True, color="rgb", colormap="parula"):
        mel_spectrograms = []
        for audio in tqdm(loaded_audio, leave=False, desc="Calculating color mel spectrograms"):
            mel_spectrograms.append(audio.get_color_mel_spectrogram(normalized=normalized, color=color, colormap=colormap))
        return mel_spectrograms

    def get_mel_spectrograms(self, loaded_audio, normalized=True, range=(0.0,1.0)):
        mel_spectrograms = []
        for audio in tqdm(loaded_audio, leave=False, desc="Calculating mel spectrograms"):
            mel_spectrograms.append(audio.get_mel_spectrogram(normalized=normalized, range=range))
        return mel_spectrograms

    def restore_audio(self, mel_spectrograms):
        audio = []
        for mel_spectrogram in tqdm(mel_spectrograms, leave=False, desc="Restoring audio"):
            audio.append(mel_spectrogram.get_audio())
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

    def process(self, args):
        self.logger.info("Processing audio.")
        initial_id = 0
        batch_size = 100
        
        for audio_files_batch in tqdm(np.array_split(self.audio_files, 1 if self.audio_files.size < batch_size else self.audio_files.size // batch_size), leave=False, desc="Precessing audio files batches"):
            loaded_audio = self.load_audio_files(audio_files_batch)

            color_mel_spectrograms = self.get_color_mel_spectrograms(loaded_audio, color=args.color)

            restored_audio = None
            if args.restore:
                restored_audio = self.restore_audio(color_mel_spectrograms)
            else:                
                loaded_audio, color_mel_spectrograms  

            self.save_data(args, initial_id, loaded_audio, color_mel_spectrograms, restored_audio)     
            initial_id += audio_files_batch.size

        self.logger.info("Audio processing done.")

    def save_data(self, args, initial_id, loaded_audio, color_mel_spectrograms, restored_audio=None):
        digits = 5
        id = initial_id
        if restored_audio is not None:
            data = list(zip(loaded_audio, color_mel_spectrograms, restored_audio))
        else:
            data = list(zip(loaded_audio, color_mel_spectrograms))

        for audio_data in  tqdm(data, leave=False, desc="Saving files"):
            # Get items
            if restored_audio is not None:
                audio, color_spec, restored_audio = audio_data
            else:
                audio, color_spec = audio_data

            # Get ID
            id_string = str(id).zfill(digits)

            # Save audio
            soundfile.write(os.path.join(args.output_dir, f"audio_original_{id_string}.wav"), audio.get_audio(), self.config.sample_rate)

            # Save color spectrogram
            file_name = f"spectrogram_color_{color_spec.color}_{id_string}"
            if color_spec.color == "rgb":
                plt.imsave(os.path.join(args.output_dir, f"{file_name}.png"), color_spec.mel_spectrogram_data)
            else:
                torch.save(torch.from_numpy(color_spec.mel_spectrogram_data), os.path.join(args.output_dir, f"{file_name}.pt"))

            # Save audio
            if args.restore:
                soundfile.write(os.path.join(args.output_dir, f"audio_restored_{id_string}.wav"), restored_audio.get_audio(), self.config.sample_rate)      

            id += 1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_dir", type=str, required=True, help='Input directory.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Output directory.')
    parser.add_argument('-s', '--statistics', action="store_true", help='Calculate mel spectrograms statistics.')
    parser.add_argument('-r', '--restore', action="store_true", help='Restore audio from generated tensors.')
    parser.add_argument('-c', '--color', type=str, help='Spectrograms color format.', default="rgb")
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
        audio_processor.process(args)