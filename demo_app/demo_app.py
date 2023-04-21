import argparse
import os
import sys

import filetype
import numpy as np
import torch
from PIL import Image

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import demo_app_utils
from demo_app_config import config
import utils.logger
import LUT

logger = utils.logger.get_logger(__name__)


# utils.logger.enable_debug_mode()


def main(args):
    if args.hide:
        hide(args)
    else:
        reveal(args)


def write_melspectrogram_data(melspectrogram_data: torch.Tensor, args):
    np.savez_compressed(
        os.path.join(args.output, "melspectrogram_with_message"),
        melspectrogram_data=melspectrogram_data.numpy(),
        is_image=(args.image is not None),
        centroids=None)


def read_melspectrogram_data(audio: str):
    data = np.load(audio, allow_pickle=True)
    return torch.from_numpy(data["melspectrogram_data"]), data["is_image"], data["centroids"]


def validate_args(args):
    if args.hide is not None:
        if args.image is not None and not filetype.is_image(args.image):
            raise ValueError(f"The file {args.image} is not an image!")

        if not (args.message is None or args.message.find(demo_app_utils.END_OF_MESSAGE_SEQUENCE) == -1):
            raise ValueError(f"Message cannot contain \'{demo_app_utils.END_OF_MESSAGE_SEQUENCE}\'")

    if not os.path.isfile(args.container):
        raise ValueError(f"Container is not a file: {args.container}")

    if not os.path.isdir(args.output):
        raise ValueError(f"Container is not a directory: {args.output}")

    if not (args.container.endswith(".npz") or filetype.is_audio(args.container)):
        raise ValueError(f"The container has unsupported type: {args.container.split(os.path.extsep)[-1]}")


def hide(args):
    cinn_model, z_dim = demo_app_utils.load_cinn(args, config)
    cinn_model.eval()

    audio = demo_app_utils.load_audio(args.container, config)
    colormap = LUT.Colormap.from_colormap("parula_norm_lab")
    melspectrogram = audio.get_color_mel_spectrogram(True, colormap=colormap)

    if args.image is not None:
        binary_data = demo_app_utils.image_to_bin(Image.open(args.image))
    else:
        binary_data = demo_app_utils.text_to_bin(args.message)

    z = demo_app_utils.encode(binary_data, z_dim, config)

    l_channel = demo_app_utils.get_l_channel(melspectrogram)
    cond = demo_app_utils.get_cond(l_channel, cinn_model)
    ab_channels = cinn_model.reverse_sample(z, cond)

    melspectrogram_data = torch.cat([l_channel, ab_channels[0].detach()], dim=1).float()

    write_melspectrogram_data(melspectrogram_data, args)


def reveal(args):
    cinn_model, z_dim = demo_app_utils.load_cinn(args, config)
    cinn_model.eval()
    melspectrogram_data, is_image, _ = read_melspectrogram_data(args.container)

    z, _, _ = cinn_model(melspectrogram_data)

    binary_data = demo_app_utils.decode(z)

    if is_image:
        hidden_image = demo_app_utils.bin_to_image(binary_data)
        hidden_image.show()
    else:
        hidden_message = demo_app_utils.bin_to_text(binary_data)
        logger.info(f"Revealed hidden message: {hidden_message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mel-steg-cINN v1.0')

    subparsers = parser.add_subparsers(help='sub-command help', dest='hide')
    parser_hide = subparsers.add_parser('hide', help='Hiding mode')
    group = parser_hide.add_mutually_exclusive_group(required=True)
    group.add_argument('-m', '--message', help="Secret message", type=str)
    group.add_argument('-i', '--image', help="Secret binary image", type=str)

    parser.add_argument('-c', '--compress', help="Use lossy compression", action="store_true")
    parser.add_argument('-b', '--bch', help="Use BCH correction code", action="store_true")
    parser.add_argument('--container', help="Path to container audio representation", type=str, required=True)
    parser.add_argument('-o', '--output', help="Output directory, default: CWD", type=str, default=None)

    args = parser.parse_args()
    args.output = args.output if args.output is not None else os.getcwd()

    try:
        validate_args(args)
        main(args)
    except ValueError as ex:
        logger.error(f"Validation error occurred: {ex}")
