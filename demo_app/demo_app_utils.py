import bitarray
from PIL import Image
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append(os.path.join(parent, "demo_app"))

import demo_app_config
import models.cinn.cinn_model
import utils.utilities
from utils.utilities import *

END_OF_MESSAGE_SEQUENCE = "_EOM"


def load_cinn(compress: bool, config, device='cpu'):
    cinn_builder = models.cinn.cinn_model.cINN_builder(config.cinn)
    feature_net = cinn_builder.get_feature_net().float()
    fc_cond_net = cinn_builder.get_fc_cond_net().float()
    cinn_net, cinn_z_dimensions = cinn_builder.get_cinn()
    cinn_model = models.cinn.cinn_model.WrappedModel(feature_net, fc_cond_net, cinn_net.float())
    cinn_utilities = models.cinn.cinn_model.cINNTrainingUtilities(cinn_model, None)

    if compress:
        cinn_utilities.load(config.compression_cinn_model_path, device=device)
    else:
        cinn_utilities.load(config.cinn_model_path, device=device)

    return cinn_utilities.model, cinn_z_dimensions


def load_audio(path: str, config: demo_app_config.Config):
    return Audio(utils.utilities.load_audio(path)[0], config.audio)


def encode_bit(bit, initial_value, config: demo_app_config.Config):
    sample = initial_value
    if bit == 0:
        while not (sample < -np.abs(config.alpha)):
            sample = np.random.normal()
    else:
        while not (sample > np.abs(config.alpha)):
            sample = np.random.normal()

    return sample


def encode(bin_message: list, cinn_z_dimensions, config: demo_app_config.Config,
           desired_shape=(1024, 512)):
    desired_size = desired_shape[0] * desired_shape[1]
    if len(bin_message) > desired_size:
        raise ValueError(f"Binary message is too long: {bin_message.size} (max: {desired_size})")

    logger.info(f"Used capacity: {len(bin_message)}/{desired_size} [bits]")

    z = np.random.normal(size=desired_size)

    for index, bit in enumerate(bin_message):
        z[index] = encode_bit(bit, z[index], config)

    logger.debug(f"Z shape: {z.shape}")

    z = torch.from_numpy(z).float()
    z = list(z.split(cinn_z_dimensions))

    logger.debug(f"Z length after split: {len(z)}")
    for i in range(len(z)):
        logger.debug(f"z[{i}].shape: {z[i].shape}")
        z[i] = z[i][None, :]
        logger.debug(f"z[{i}].shape(corrected): {z[i].shape}")

    return z


def decode(z):
    logger.debug(f"Z input length: {len(z)}")
    z = torch.cat(z, dim=1).squeeze().detach().numpy()

    logger.debug(f"Z concatenated shape: {z.shape}")

    bin_message = []

    for sample in z:
        if sample < 0:
            bin_message.append(0)
        elif sample >= 0:
            bin_message.append(1)

    return bin_message


def get_l_channel(melspectrogram: MelSpectrogram):
    # Load tensor
    l_channel = torch.from_numpy(melspectrogram.mel_spectrogram_data[:, :, 0])
    # Adjust axes
    l_channel = torch.reshape(l_channel, (1, 1, l_channel.shape[0], l_channel.shape[1]))
    return l_channel.float()


def get_ab_channels(melspectrogram: MelSpectrogram):
    # Load tensor
    ab = torch.from_numpy(melspectrogram.mel_spectrogram_data[:, :, 1:3])
    # Adjust axes
    return ab.permute((2, 0, 1))[None, :].float()


def get_melspectrogram_tensor(melspectrogram: MelSpectrogram):
    # Load tensor
    spectrogram_data = torch.from_numpy(melspectrogram.mel_spectrogram_data)
    # Adjust axes
    spectrogram_data = torch.permute(spectrogram_data, (2, 0, 1))[None, :]

    return spectrogram_data


def get_cond(l_channel: torch.Tensor, cinn_model: models.cinn.cinn_model.WrappedModel):
    with torch.no_grad():
        features, _ = cinn_model.feature_network.features(l_channel)
        return [*features]


def image_to_bin(image: Image, out_shape=(1024, 512)):
    white = [255, 255, 255, 255]
    black = [0, 0, 0, 255]

    image_data = np.asarray(image)

    if np.any(image_data.shape[:2] > out_shape):
        raise ValueError(f"Image dimensions are too big: {image_data.shape[:2]} (max: {out_shape})")

    binary_values = np.zeros(image_data.shape[:2])

    for x in range(image_data.shape[0]):
        for y in range(image_data.shape[1]):
            color = image_data[x, y]
            if (color == white).all():
                binary_values[x, y] = 1
            elif (color == black).all():
                binary_values[x, y] = 0
            else:
                raise ValueError(f"The image contains unsupported color: {color}.")

    return binary_values.flatten().tolist(), binary_values.shape


def bin_to_image(binary_values: list, image_shape=(1024, 512)):
    white = [255, 255, 255, 255]
    black = [0, 0, 0, 255]

    image_data = np.zeros((*image_shape, 4), dtype=np.uint8)

    i = 0
    logger.debug(f"binary_values length: {len(binary_values)}")
    for x in range(image_data.shape[0]):
        for y in range(image_data.shape[1]):
            bit = binary_values[i]
            if bit == 1:
                image_data[x, y] = white
            else:
                image_data[x, y] = black

            i += 1

    return Image.fromarray(image_data)


def text_to_bin(text: str, max_size=1024 * 512):
    bits = bitarray.bitarray()
    bits.frombytes((text + END_OF_MESSAGE_SEQUENCE).encode('utf-8'))
    bits_list = bits.tolist()

    if len(bits_list) >= max_size:
        raise ValueError(f"Message binary size is too big: {len(bits_list)}(max: {max_size})")

    return bits_list, len(bits_list)


def bin_to_text(binary_values: list):
    return bitarray.bitarray(binary_values).tobytes().decode('utf-8', errors='replace').split(END_OF_MESSAGE_SEQUENCE,
                                                                                              maxsplit=1)[0]
