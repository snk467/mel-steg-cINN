import argparse
import os
import sys
import traceback

import filetype
import numpy as np
import soundfile as soundfile
import torch
from PIL import Image

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from bchcode import BCHCode
import demo_app_utils
from demo_app_config import config
import utils.logger
from utils import utilities
import utils.metrics as metrics
import utils.visualization as visualization
import LUT

MEL_SPECTROGRAM_WITH_MESSAGE_FILENAME = "melspectrogram_with_message.npz"
logger = utils.logger.get_logger(__name__)
data = {}


def main(args):
    if args.demo:
        logger.info("!!!======>DEMO MODE<======!!!")
        hide(args)
        reveal(args)
    else:
        if args.hide:
            hide(args)
        else:
            reveal(args)


def write_melspectrogram_data(melspectrogram_data: torch.Tensor, centroids, message_shape, bch_parameters):
    np.savez_compressed(
        os.path.join(args.output, MEL_SPECTROGRAM_WITH_MESSAGE_FILENAME),
        melspectrogram_data=melspectrogram_data,
        message_shape=message_shape,
        bch_parameters=bch_parameters,
        centroids=centroids)


def read_melspectrogram_data(audio: str):
    data = np.load(audio, allow_pickle=True)
    return data["melspectrogram_data"], data["message_shape"], data["centroids"], data["bch_parameters"]


def validate_args(args):
    if args.hide is not None:
        if args.image is not None and not filetype.is_image(args.image):
            raise ValueError(f"The file {args.image} is not an image!")

        if not (args.message is None or args.message.find(demo_app_utils.END_OF_MESSAGE_SEQUENCE) == -1):
            raise ValueError(f"Message cannot contain \'{demo_app_utils.END_OF_MESSAGE_SEQUENCE}\'")

        if args.bch is not None and (len(args.bch) != 0 and len(args.bch) != 2):
            raise ValueError("argument -b/--bch expects 0 or 2 arguments")

    if not os.path.isfile(args.container):
        raise ValueError(f"Container is not a file: {args.container}")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if os.path.isfile(args.output):
        raise ValueError(f"Container is not a directory: {args.output}")

    if not (args.container.endswith(".npz") or filetype.is_audio(args.container)):
        raise ValueError(f"The container has unsupported type: {args.container.split(os.path.extsep)[-1]}")


def print_statistics():
    message_shape = data["message_shape"]

    if type(message_shape) is tuple:
        out_shape = (1024, 512)
        message_only_accuracy = metrics.accuracy(
            data['z_result'].reshape(out_shape)[:message_shape[0], :message_shape[1]],
            data['z_target'].reshape(out_shape)[:message_shape[0], :message_shape[1]])
    else:
        message_only_accuracy = metrics.accuracy(data['z_result'][:message_shape], data['z_target'][:message_shape])

    logger.info(f"MSE_ab: {metrics.mse(data['ab_result'], data['ab_target'])}, "
                f"MAE_ab: {metrics.mae(data['ab_result'], data['ab_target'])}, "
                f"MSE_z: {metrics.mse(data['z_result'], data['z_target'])}, "
                f"MAE_z: {metrics.mae(data['z_result'], data['z_target'])}, "
                f"Accuracy_z: {1.0 - metrics.accuracy(data['z_result'], data['z_target'])}, "
                f"Accuracy_binary: {1.0 - metrics.accuracy(torch.Tensor(data['binary_data_result'][:len(data['binary_data_target'])]), torch.Tensor(data['binary_data_target']))}, "
                f"Accuracy_z (only message): {1.0 - message_only_accuracy}")


def save_images():
    visualization.get_img_simple(data['melspectrogram_target'][0][:1], data['melspectrogram_target'][0][1:],
                                 "melspectrogram_target").save(os.path.join(args.output, "melspectrogram_target.png"))
    visualization.get_img_simple(data['melspectrogram_result'][0][:1], data['melspectrogram_result'][0][1:],
                                 "melspectrogram_result").save(
        os.path.join(args.output, "melspectrogram_result.png"))


def hide(args):
    logger.info("Mode: hiding")

    cinn_model, z_dim = demo_app_utils.load_cinn(args.compress, config)
    cinn_model.eval()

    audio = demo_app_utils.load_audio(args.container, config)
    colormap = LUT.Colormap.from_colormap("parula_norm_lab")
    melspectrogram = audio.get_color_mel_spectrogram(True, colormap=colormap)

    if args.image:
        binary_data, message_shape = demo_app_utils.image_to_bin(Image.open(args.image))
        logger.info("Image converted to binary data.")
    else:
        binary_data, message_shape = demo_app_utils.text_to_bin(args.message)
        logger.info("Message converted to binary data.")

    if args.demo:
        data["binary_data_target"] = binary_data
        filepath = os.path.join(args.output, "audio_target.wav")
        soundfile.write(filepath, melspectrogram.get_audio().get_audio(), config.audio.sample_rate)
        logger.info("Saved audio_target.wav")

        if args.compress:
            melspectrogram_target_decompressed = utilities.decompress_melspectrogram(
                *utilities.compress_melspectrogram(utilities.get_melspectrogram_tensor(melspectrogram)[0]))
            visualization.restore_audio(melspectrogram_target_decompressed[0][0:1],
                                        melspectrogram_target_decompressed[0][1:],
                                        "audio_compress_target", args.output, audio_config=config.audio)
            logger.info("Saved audio_compress_target.wav")

    bch_parameters = None
    if args.bch is not None:
        if len(args.bch) == 0:
            bch_parameters = [config.bch.mi, config.bch.tau]
            binary_data = BCHCode(binary_data).encode(config.bch.mi, config.bch.tau)
        else:
            bch_parameters = args.bch
            binary_data = BCHCode(binary_data).encode(args.bch[0], args.bch[1])
        logger.info("Binary data encoded with BCH.")

    z = demo_app_utils.encode(binary_data, z_dim, config)
    logger.info("Binary data encoded to z (noise).")

    l_channel = demo_app_utils.get_l_channel(melspectrogram)
    cond = demo_app_utils.get_cond(l_channel, cinn_model)
    ab_channels = cinn_model.reverse_sample(z, cond)
    logger.info("Binary data hidden into mel spectrogram.")

    melspectrogram_data = torch.cat([l_channel, ab_channels[0].detach()], dim=1).float()

    centroids = None
    if args.compress:
        melspectrogram_data, centroids = utilities.compress_melspectrogram(melspectrogram_data[0])
        logger.info("Compressed mel spectrogram.")
    else:
        melspectrogram_data = melspectrogram_data.numpy()

    if args.demo:
        data["melspectrogram_target"] = utilities.get_melspectrogram_tensor(melspectrogram)
        data["ab_target"] = utilities.get_melspectrogram_tensor(melspectrogram)[:, 1:]
        data["z_target"] = torch.cat(z, dim=1).squeeze().detach()
        data["message_shape"] = message_shape

    write_melspectrogram_data(melspectrogram_data, centroids, message_shape, bch_parameters)
    logger.info("Mel spectrogram data saved successfully.")


def reveal(args):
    logger.info("Mode: revealing")

    if args.demo:
        container = os.path.join(args.output, MEL_SPECTROGRAM_WITH_MESSAGE_FILENAME)
    else:
        container = args.container

    melspectrogram_data, message_shape, centroids, bch_parameters = read_melspectrogram_data(container)
    logger.info("Mel spectrogram data read successfully.")

    is_compressed = np.any(centroids)
    cinn_model, z_dim = demo_app_utils.load_cinn(is_compressed, config)
    cinn_model.eval()

    if is_compressed:
        melspectrogram_data = utilities.decompress_melspectrogram(melspectrogram_data, centroids)
        logger.info("Decompressed mel spectrogram.")
    else:
        melspectrogram_data = torch.from_numpy(melspectrogram_data)

    z, _, _ = cinn_model(melspectrogram_data)
    logger.info("Extracted z (noise) from mel spectrogram.")

    binary_data = demo_app_utils.decode(z)
    logger.info("Binary data decoded from z (noise).")

    if np.any(bch_parameters):
        binary_data = BCHCode(binary_data).decode(bch_parameters[0], bch_parameters[1], message_shape)
        logger.info("Binary data decoded with BCH.")

    if args.demo:
        visualization.restore_audio(melspectrogram_data[0][0:1], melspectrogram_data[0][1:],
                                    "audio_result", args.output, audio_config=config.audio)
        logger.info("Saved audio_result.wav")
        data["melspectrogram_result"] = melspectrogram_data
        data["ab_result"] = melspectrogram_data[:, 1:].detach()
        data["z_result"] = torch.cat(z, dim=1).squeeze().detach()
        data["binary_data_result"] = binary_data

    if message_shape.size == 2:
        hidden_image = demo_app_utils.bin_to_image(binary_data, message_shape)
        logger.info("Binary data converted to image.")
        logger.info(f"Showing hidden image.")
        hidden_image.show()
    else:
        hidden_message = demo_app_utils.bin_to_text(binary_data)
        logger.info("Binary data converted to text.")
        logger.info(f"Revealed hidden message: {hidden_message}")

    if args.demo:
        print_statistics()
        save_images()
        os.remove(os.path.join(args.output, MEL_SPECTROGRAM_WITH_MESSAGE_FILENAME))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mel-steg-cINN v1.0')

    subparsers = parser.add_subparsers(help='sub-command help', dest='hide')
    parser_hide = subparsers.add_parser('hide', help='Hiding mode')
    group = parser_hide.add_mutually_exclusive_group(required=True)

    parser_hide.add_argument('-c', '--compress', help="Use lossy compression", action="store_true")
    parser_hide.add_argument('-b', '--bch', help="Use BCH correction code", nargs='*', type=int)
    parser_hide.add_argument('-d', '--demo', help="Simulate end-to-end steganographic process", action="store_true")
    group.add_argument('-m', '--message', help="Secret message", type=str)
    group.add_argument('-i', '--image', help="Secret binary image", type=str)

    parser.add_argument('--container', help="Path to container audio representation", type=str, required=True)
    parser.add_argument('-o', '--output', help="Output directory, default: CWD", type=str, default=None)

    args = parser.parse_args()
    args.output = args.output if args.output is not None else os.getcwd()

    try:
        validate_args(args)
        main(args)
    except ValueError as ex:
        traceback.print_exception(ex)
        logger.error(f"Validation error occurred: {ex}")
    except KeyboardInterrupt:
        logger.info("Canceled.")
