import argparse
import os
import sys
import traceback

import filetype
import numpy as np
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
import LUT

logger = utils.logger.get_logger(__name__)


# utils.logger.enable_debug_mode()

def main(args):
    if args.hide:
        hide(args)
    else:
        reveal(args)


def write_melspectrogram_data(melspectrogram_data: torch.Tensor, centroids, message_shape, bch_parameters):
    np.savez_compressed(
        os.path.join(args.output, "melspectrogram_with_message"),
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

    if not os.path.isdir(args.output):
        raise ValueError(f"Container is not a directory: {args.output}")

    if not (args.container.endswith(".npz") or filetype.is_audio(args.container)):
        raise ValueError(f"The container has unsupported type: {args.container.split(os.path.extsep)[-1]}")


def hide(args):
    cinn_model, z_dim = demo_app_utils.load_cinn(args.compress, config)
    cinn_model.eval()

    audio = demo_app_utils.load_audio(args.container, config)
    colormap = LUT.Colormap.from_colormap("parula_norm_lab")
    melspectrogram = audio.get_color_mel_spectrogram(True, colormap=colormap)

    if args.image:
        binary_data, message_shape = demo_app_utils.image_to_bin(Image.open(args.image))
    else:
        binary_data, message_shape = demo_app_utils.text_to_bin(args.message)

    print(message_shape)

    bch_parameters = None
    if args.bch is not None:
        if len(args.bch) == 0:
            bch_parameters = [config.bch.mi, config.bch.tau]
            binary_data = BCHCode(binary_data).encode(config.bch.mi, config.bch.tau)
        else:
            bch_parameters = args.bch
            binary_data = BCHCode(binary_data).encode(args.bch[0], args.bch[1])

    z = demo_app_utils.encode(binary_data, z_dim, config)

    l_channel = demo_app_utils.get_l_channel(melspectrogram)
    cond = demo_app_utils.get_cond(l_channel, cinn_model)
    ab_channels = cinn_model.reverse_sample(z, cond)

    melspectrogram_data = torch.cat([l_channel, ab_channels[0].detach()], dim=1).float()

    centroids = None
    if args.compress:
        melspectrogram_data, centroids = utilities.compress_melspectrogram(melspectrogram_data[0])
    else:
        melspectrogram_data = melspectrogram_data.numpy()

    write_melspectrogram_data(melspectrogram_data, centroids, message_shape, bch_parameters)


def reveal(args):
    melspectrogram_data, message_shape, centroids, bch_parameters = read_melspectrogram_data(args.container)
    is_compressed = np.any(centroids)
    cinn_model, z_dim = demo_app_utils.load_cinn(is_compressed, config)
    cinn_model.eval()

    if is_compressed:
        melspectrogram_data = utilities.decompress_melspectrogram(melspectrogram_data, centroids)
    else:
        melspectrogram_data = torch.from_numpy(melspectrogram_data)

    z, _, _ = cinn_model(melspectrogram_data)

    binary_data = demo_app_utils.decode(z)

    if np.any(bch_parameters):
        binary_data = BCHCode(binary_data).decode(bch_parameters[0], bch_parameters[1], message_shape)

    if message_shape.size == 2:
        hidden_image = demo_app_utils.bin_to_image(binary_data, message_shape)
        hidden_image.show()
    else:
        hidden_message = demo_app_utils.bin_to_text(binary_data)
        logger.info(f"Revealed hidden message: {hidden_message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mel-steg-cINN v1.0')

    subparsers = parser.add_subparsers(help='sub-command help', dest='hide')
    parser_hide = subparsers.add_parser('hide', help='Hiding mode')
    group = parser_hide.add_mutually_exclusive_group(required=True)

    parser_hide.add_argument('-c', '--compress', help="Use lossy compression", action="store_true")
    parser_hide.add_argument('-b', '--bch', help="Use BCH correction code", nargs='*', type=int)
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
