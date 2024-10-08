import argparse
import os.path

import PIL.Image as Image
import filetype
import h5py
import soundfile
from tqdm import tqdm

import exceptions
import utils.logger
from config import config
from utils.utilities import *


class AudioDatasetProcessor:

    def __init__(self, audo_files_directory, config):
        self.config = config

        self.audio_files = get_files(audo_files_directory)

        self.logger = utils.logger.get_logger(__name__)

    def load_audio_files(self, audio_files):
        loaded_audio = []

        for audio_file in tqdm(audio_files, leave=False, desc="Loading audio files"):
            if not filetype.is_audio(audio_file):
                continue

            audio_full, sample_rate = load_audio(audio_file, self.config.sample_rate)

            if sample_rate != self.config.sample_rate:
                raise exceptions.SampleRateError

            audio = Audio(audio_full, self.config)

            loaded_audio.append(audio)

        return loaded_audio

    def get_color_mel_spectrograms(self, loaded_audio, colormap, normalized=True):
        mel_spectrograms = []
        for audio in tqdm(loaded_audio, leave=False, desc="Calculating color mel spectrograms"):
            mel_spectrograms.append(audio.get_color_mel_spectrogram(normalized=normalized, colormap=colormap))
        return mel_spectrograms

    def get_mel_spectrograms(self, loaded_audio, normalized=True, range=(0.0, 1.0)):
        mel_spectrograms = []
        for audio in tqdm(loaded_audio, leave=False, desc="Calculating mel spectrograms"):
            mel_spectrograms.append(audio.get_mel_spectrogram(normalized=normalized, range=range))
        return mel_spectrograms

    def restore_audio(self, mel_spectrograms):
        audio = []
        for mel_spectrogram in tqdm(mel_spectrograms, leave=False, desc="Restoring audio"):
            audio.append(mel_spectrogram.get_audio())
        return audio

    def __calculate_global_statistics(self, n, sum, sum_2):
        self.logger.info("Calculating mean and standard deviation.")
        return sum / n, np.sqrt(sum_2 / n - (sum / n) ** 2)

    def get_statistics(self):
        self.logger.info("Processing audio.")

        n = 0
        sum = 0.0
        sum_2 = 0.0
        batch_size = 100

        for audio_files_batch in tqdm(self.__get_batches(self.audio_files, batch_size=batch_size), leave=False,
                                      desc="Precessing audio files batches"):
            loaded_audio = self.load_audio_files(audio_files_batch)
            mel_spectrograms = self.get_mel_spectrograms(loaded_audio, normalized=False, range=None)

            n_batch, sum_batch, sum_2_batch = normalization.calculate_statistics(mel_spectrograms)

            n += n_batch
            sum += sum_batch
            sum_2 += sum_2_batch

        mean, standard_deviation = self.__calculate_global_statistics(n, sum, sum_2)

        self.logger.info("Adjusting statistics.")

        def modify_stats(audio):
            audio.config.mean = float(mean)
            audio.config.standard_deviation = float(standard_deviation)
            return audio

        min = np.Inf
        max = -np.Inf
        self.logger.info("Calculating min and max.")
        for audio_files_batch in tqdm(self.__get_batches(self.audio_files, batch_size=batch_size), leave=False,
                                      desc="Precessing audio files batches"):
            loaded_audio = self.load_audio_files(audio_files_batch)
            loaded_audio = map(modify_stats, loaded_audio)

            mel_spectrograms = self.get_mel_spectrograms(loaded_audio, normalized=True, range=None)

            min_batch, max_batch = normalization.calculate_minmax(mel_spectrograms)

            min = np.min([min, min_batch])
            max = np.max([max, max_batch])

        self.logger.info("Audio processing done.")

        return mean, standard_deviation, min, max

    def __get_batches(self, array, batch_size=100):
        return np.array_split(array, 1 if array.size < batch_size else array.size // batch_size)

    def process(self, args):
        self.logger.info("Processing audio.")
        initial_id = 0

        dataset_length = len(self.audio_files)
        os.makedirs(args.output_dir, exist_ok=True)
        dataset_file = h5py.File(
            os.path.join(args.output_dir, f"melspectrograms_{args.colormap}_{dataset_length}.hdf5"), "w")

        dataset_shape = (dataset_length, self.config.n_mels, self.config.n_mels)
        if args.colormap is not None:
            # RGB/Lab channels
            dataset_shape += (3,)

        melspectrograms_dataset = dataset_file.create_dataset("melspectrograms",
                                                              shape=dataset_shape,
                                                              compression="gzip",
                                                              chunks=(1,) + dataset_shape[1:])

        melspectrograms_dataset.attrs["colormap"] = str(args.colormap)

        for audio_files_batch in tqdm(self.__get_batches(self.audio_files), leave=False,
                                      desc="Precessing audio files batches"):
            loaded_audio = self.load_audio_files(audio_files_batch)

            color_mel_spectrograms = self.get_color_mel_spectrograms(loaded_audio,
                                                                     colormap=LUT.Colormap.from_colormap(args.colormap))

            restored_audio = None
            if args.restore:
                restored_audio = self.restore_audio(color_mel_spectrograms)

            self.save_data(args, initial_id, loaded_audio, color_mel_spectrograms, melspectrograms_dataset,
                           restored_audio)
            initial_id += audio_files_batch.size

        dataset_file.close()

        self.logger.info("Audio processing done.")

    def __prepare_labels_tensor(self, color_mapping, colormap):
        labels = torch.zeros((colormap.get_colors_length(), color_mapping.shape[0], color_mapping.shape[1]))
        for x in range(color_mapping.shape[0]):
            for y in range(color_mapping.shape[1]):
                labels[color_mapping[x, y], x, y] = 1.0

        return labels

    def save_data(self, args, initial_id, loaded_audio, color_mel_spectrograms, melspectrograms_dataset,
                  restored_audio=None):
        digits = 5
        id = initial_id
        if restored_audio is not None:
            data = list(zip(loaded_audio, color_mel_spectrograms, restored_audio))
        else:
            data = list(zip(loaded_audio, color_mel_spectrograms))

        for audio_data in tqdm(data, leave=False, desc="Saving files"):
            # Get items
            if restored_audio is not None:
                audio, color_spec, restored_audio = audio_data
            else:
                audio, color_spec = audio_data

            # Get ID
            id_string = str(id).zfill(digits)

            # Save audio
            soundfile.write(os.path.join(args.output_dir, f"audio_original_{id_string}.wav"), audio.get_audio(),
                            self.config.sample_rate)

            # Save spectrogram data
            colormap_string = str(color_spec.colormap).lower()
            melspectrograms_dataset[id] = color_spec.mel_spectrogram_data

            if args.debug and id == initial_id:
                spectrogram_data = color_spec.mel_spectrogram_data
                img = Image.fromarray((spectrogram_data[:, :, 0] * 255).astype(np.uint8), 'L')
                img.show(title="mel_spectrogram_data_L")

                img = Image.fromarray((spectrogram_data[:, :, 1] * 255).astype(np.uint8), 'L')
                img.show(title="mel_spectrogram_data_a")

                img = Image.fromarray((spectrogram_data[:, :, 2] * 255).astype(np.uint8), 'L')
                img.show(title="mel_spectrogram_data_b")

                # Save audio
            if args.restore:
                soundfile.write(os.path.join(args.output_dir, f"audio_restored_{colormap_string}_{id_string}.wav"),
                                restored_audio.get_audio(), self.config.sample_rate)

            id += 1


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
        utils.logger.enable_debug_mode()

    # Get audio processor
    audio_processor = AudioDatasetProcessor(args.input_dir, config.audio_parameters.resolution_512x512)
    logger.info("Audio processor initialized.")

    if args.statistics:
        mean, standard_deviation, min, max = audio_processor.get_statistics()
        file_location = os.path.join(args.output_dir, "dataset_stats.txt")
        file = open(file_location, "w")
        file.write(f"mean: {mean}\n")
        file.write(f"standard_deviation: {standard_deviation}\n")
        file.write(f"global_min: {min}\n")
        file.write(f"global_max: {max}\n")
        file.close()
        logger.info(f"Statistics saved to {file_location}")
    else:
        audio_processor.process(args)
