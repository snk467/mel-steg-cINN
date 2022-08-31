import argparse
import soundfile
import gzip
import shutil
import os
import random
import Exceptions
import Configuration
import Normalization 
import Logger
import torch
import matplotlib.pyplot as plt
from zipfile import ZipFile
import zipfile
from Utilities import *
from tqdm import tqdm
import torchvision.transforms as torch_trans
import PIL.Image as Image
logger = Logger.get_logger(__name__)

class AudioDatasetProcessor:
    
    def __init__(self, audo_files_directory, config):
        self.config = config

        self.audio_files = get_files(audo_files_directory)

        self.logger = Logger.get_logger(__name__)

    def load_audio_files(self, audio_files):
        loaded_audio = []

        for audio_file in tqdm(audio_files, leave=False, desc="Loading audio files"):
            audio, sample_rate = load_audio(audio_file)

            if sample_rate != self.config.sample_rate:
                raise Exceptions.SampleRateError

            audio = Audio(audio, self.config)

            loaded_audio.append(audio)

        return loaded_audio

    def get_color_mel_spectrograms(self, loaded_audio, normalized=True, colormap="parula_rgb"):
        mel_spectrograms = []
        for audio in tqdm(loaded_audio, leave=False, desc="Calculating color mel spectrograms"):
            mel_spectrograms.append(audio.get_color_mel_spectrogram(normalized=normalized, colormap=colormap))
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

            color_mel_spectrograms = self.get_color_mel_spectrograms(loaded_audio, colormap=args.colormap)

            restored_audio = None
            if args.restore:
                restored_audio = self.restore_audio(color_mel_spectrograms)

            self.save_data(args, initial_id, loaded_audio, color_mel_spectrograms, restored_audio)     
            initial_id += audio_files_batch.size

        self.logger.info("Audio processing done.")

    def __prepare_labels_tensor(self, color_mapping, colormap):
        labels = torch.zeros((colormap.get_colors_length(), color_mapping.shape[0], color_mapping.shape[1]))
        for x in range(color_mapping.shape[0]):
            for y in range(color_mapping.shape[1]):
                labels[color_mapping[x, y], x, y] = 1.0

        return labels

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
            colormap_string = str(color_spec.colormap).lower()
            spectrogram_data = color_spec.mel_spectrogram_data

            if args.debug and id == initial_id:
                img = Image.fromarray((spectrogram_data[:,:,0] * 255).astype(np.uint8), 'L')
                img.show(title="mel_spectrogram_data_L")

                img = Image.fromarray((spectrogram_data[:,:,1] * 255).astype(np.uint8), 'L')
                img.show(title="mel_spectrogram_data_a")

                img = Image.fromarray((spectrogram_data[:,:,2] * 255).astype(np.uint8), 'L')
                img.show(title="mel_spectrogram_data_b")

            # # Save labels if colormap is applied
            # if color_spec.colormap is not None:
            #     colormap = Colormap.from_colormap(colormap_string)
            #     labels_tensor = self.__prepare_labels_tensor(colormap.get_indexes_from_colors(spectrogram_data), colormap)
            #     labels_file_basename= f"labels_{colormap_string}_{id_string}"

            #     self.__save_compressed_tensor(args.output_dir, labels_tensor, labels_file_basename)

            

            # Save spectrogram data
            file_name = f"spectrogram_{colormap_string}_{id_string}"
            if "rgb" in str(color_spec.colormap):                
                plt.imsave(os.path.join(args.output_dir, f"{file_name}.png"), color_spec.mel_spectrogram_data)
            elif "lab" in str(color_spec.colormap):             
                L_tensor = torch.from_numpy(spectrogram_data[:,:,0])
                L_tensor = torch.reshape(L_tensor, (1, L_tensor.shape[0], L_tensor.shape[1]))
                L_channel_file_basename = f"spectrogram_L_channel_{colormap_string}_{id_string}"
                self.__save_compressed_tensor(args.output_dir, L_tensor, L_channel_file_basename)

                ab_tensor = torch.from_numpy(spectrogram_data[:,:,1:])
                ab_tensor = torch.permute(ab_tensor, (2, 0, 1))
                ab_channel_file_basename = f"spectrogram_ab_channels_{colormap_string}_{id_string}"
                self.__save_compressed_tensor(args.output_dir, ab_tensor, ab_channel_file_basename)

                if args.debug and id == initial_id:
                    toImage = torch_trans.ToPILImage()
                    img = toImage(L_tensor) 
                    img.show(title="L channel")        
                    
                    img = toImage(ab_tensor[0]) 
                    img.show(title="a channel")  
                    
                    img = toImage(ab_tensor[1]) 
                    img.show(title="b channel")  

            else:
                with open(os.path.join(args.output_dir, f"{file_name}.npy"), 'wb') as file:
                    np.save(file, color_spec.mel_spectrogram_data) 

            # Save audio
            if args.restore:
                soundfile.write(os.path.join(args.output_dir, f"audio_restored_{colormap_string}_{id_string}.wav"), restored_audio.get_audio(), self.config.sample_rate)      

            id += 1

    def __save_compressed_tensor(self, dest_dir, input_tensor, filename):
        tensor_path =  os.path.join(dest_dir, f"{filename}.pt")
        zip_filename = f"{os.path.splitext(os.path.basename(tensor_path))[0]}.zip"
        zip_path = os.path.join(dest_dir, zip_filename)

        torch.save(input_tensor, tensor_path)

        if os.path.exists(zip_path):
            os.remove(zip_path)

        with open(tensor_path, 'rb') as f_in:
            with gzip.open(f'{tensor_path}.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(tensor_path)
        logger.debug(f"Saved tensor of shape {input_tensor.shape} in {zip_path}.")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_dir", type=str, required=True, help='Input directory.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Output directory.')
    parser.add_argument('-s', '--statistics', action="store_true", help='Calculate mel spectrograms statistics.')
    parser.add_argument('-r', '--restore', action="store_true", help='Restore audio from generated tensors.')
    parser.add_argument('-d', '--debug', action="store_true", help='Display debug information.')
    parser.add_argument('-c', '--colormap', type=str, help='Spectrograms colormap.', default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":   
    # Initialize RNG
    random.seed(1234)

    # Get arguments
    args = get_args()

    if args.debug:
        Logger.enable_debug_mode()

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