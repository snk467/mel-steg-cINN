import os
import random

import PIL.Image as Image
import numpy as np
import soundfile
import torch
import torch.nn.functional as F
from IPython.display import display, Audio
from torchvision.transforms import ToPILImage

import LUT
import utils.utilities as utilities
from config import config


def expand2size(pil_img, size):
    width, height = pil_img.size
    background_color = (0, 0, 0)

    assert width == height, "Image must be a square."
    assert size >= height, "Size must be greater than original image size."

    if width == size:
        return pil_img
    else:
        result = Image.new(pil_img.mode, (size, size), background_color)
        result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
        return result


def get_img_simple(input_l, input_ab, label, verbose=False):
    l_channel = input_l.detach().cpu()
    ab_channels = input_ab.detach().cpu()

    if verbose:
        print("L shape:", l_channel.shape)
        print("ab shape:", ab_channels.shape)
        print("Label:", label)

    l_img = ToPILImage()(l_channel).convert('RGB')

    a_img = ToPILImage()(ab_channels[0]).convert('RGB')

    b_img = ToPILImage()(ab_channels[1]).convert('RGB')

    rgb_img = get_rgb_image_from_lab_channels(l_channel, ab_channels)

    max_size = max(l_img.size[0], a_img.size[0], b_img.size[0])

    l_img = expand2size(l_img, max_size)

    a_img = expand2size(a_img, max_size)

    b_img = expand2size(b_img, max_size)

    rgb_img = expand2size(rgb_img, max_size)

    border_width = 10
    border = Image.fromarray(np.zeros((max_size, border_width))).convert('RGB')

    img = Image.fromarray(np.hstack((np.array(l_img),
                                     np.array(border),
                                     np.array(a_img),
                                     np.array(border),
                                     np.array(b_img),
                                     np.array(border),
                                     np.array(rgb_img))))

    return img


def get_img(input_l, input_ab, label, input_clear_l, verbose=True):
    l_channel = input_l.detach().cpu()
    ab_channels = input_ab.detach().cpu()
    clear_l = input_clear_l

    if verbose:
        print("L shape:", l_channel.shape)
        print("ab shape:", ab_channels.shape)
        print("Label:", label)

    l_img = ToPILImage()(l_channel).convert('RGB')

    l_clear_img = ToPILImage()(clear_l).convert('RGB')

    a_img = ToPILImage()(ab_channels[0]).convert('RGB')

    b_img = ToPILImage()(ab_channels[1]).convert('RGB')

    rgb_img = get_rgb_image_from_lab_channels(clear_l, ab_channels)

    max_size = max(l_img.size[0], l_clear_img.size[0], a_img.size[0], b_img.size[0])

    l_img = expand2size(l_img, max_size)

    l_clear_img = expand2size(l_clear_img, max_size)

    a_img = expand2size(a_img, max_size)

    b_img = expand2size(b_img, max_size)

    rgb_img = expand2size(rgb_img, max_size)

    border_width = 10
    border = Image.fromarray(np.zeros((max_size, border_width))).convert('RGB')

    img = Image.fromarray(np.hstack((np.array(l_img),
                                     np.array(border),
                                     np.array(l_clear_img),
                                     np.array(border),
                                     np.array(a_img),
                                     np.array(border),
                                     np.array(b_img),
                                     np.array(border),
                                     np.array(rgb_img))))

    return img


def show_data(input_in, target_in, label, clear_input_in, restore_audio_track=False, audio_file_name=None):

    img = get_img(input_in, target_in, label, clear_input_in)

    if config.common.present_data:
        display(img)

    if restore_audio_track:
        filename = f"restored_audio_{label}" if audio_file_name is None else audio_file_name
        restore_audio(input_in, target_in, audio_file_name=filename)
        display(Audio(f"{filename}.wav"))

    return img


def __get_colors_from_tensors(L_channel: torch.Tensor, ab_channels: torch.Tensor):

    assert len(L_channel.shape) == len(ab_channels.shape)
    assert len(L_channel.shape) == 3 or (ab_channels.shape[0] == 1 and L_channel.shape[0] == 1)

    if len(L_channel.shape) == 4:
        L_np = L_channel.squeeze(dim=0).numpy()
    else:
        L_np = L_channel.numpy()

    if len(ab_channels.shape) == 4:
        ab_np = ab_channels.squeeze(dim=0).numpy()
    else:
        ab_np = ab_channels.numpy()

    Lab_np = np.concatenate((L_np, ab_np))
    Lab_np = np.moveaxis(Lab_np, 0, -1)

    return Lab_np


def restore_audio(l_channel: torch.Tensor, ab_channels: torch.Tensor,
                  audio_file_name: str, output_dir=None, audio_config=config.audio_parameters.resolution_512x512):

    colormap_lab = LUT.Colormap.from_colormap("parula_norm_lab")
    lab_np = __get_colors_from_tensors(l_channel, ab_channels)
    melspectrogram = utilities.MelSpectrogram.from_color(lab_np, True, colormap_lab, audio_config)
    filepath = os.path.join(os.getcwd() if output_dir is None else output_dir, f"{audio_file_name}.wav")
    soundfile.write(filepath, melspectrogram.get_audio().audio, audio_config.sample_rate)


def get_rgb_image_from_lab_channels(L_channel: torch.Tensor, ab_channels: torch.Tensor, colormap=None):
    if colormap is None:
        colormap = LUT.Colormap.from_colormap("parula_norm_lab")
    Lab_np = __get_colors_from_tensors(L_channel, ab_channels)
    indexes = colormap.get_indexes_from_colors(Lab_np)
    colormap_rgb = LUT.Colormap.from_colormap("parula_rgb")
    img_target = colormap_rgb.get_colors_from_indexes(indexes)
    img_target = ToPILImage()((img_target * 255).astype(np.uint8))
    return img_target


def predict_example(model, dataset, desc=None, restore_audio=False):
    example_id = random.randint(0, len(dataset) - 1)
    input, target, filename, clear_input = dataset[example_id]
    batched_input = torch.reshape(input, (1, *input.shape)).to(utilities.get_device(verbose=False)).float()
    output = model(batched_input)
    print(desc)
    print("Result:")
    show_data(batched_input[0], output[0], filename, clear_input, restore_audio,
              audio_file_name=f"result_audio_{filename[0]}")
    print("Target:")
    show_data(input, target, filename, clear_input, restore_audio, audio_file_name=f"target_audio_{filename[0]}")


def __sample_outputs(sigma, out_shape, batch_size):
    return [sigma * torch.FloatTensor(torch.Size((batch_size, o))).normal_().to(utilities.get_device(verbose=False)) for
            o in out_shape]


def predict_cinn_example(cinn_model, cinn_output_dimensions, dataset, config, desc=None, restore_audio=False):
    example_id = random.randint(0, len(dataset) - 1)
    cinn_model.to(utilities.get_device(verbose=False))
    input, target, filename, clear_input = dataset[example_id]
    sample_z = utilities.sample_z(cinn_output_dimensions, 1, config.alpha, device=utilities.get_device(verbose=False))
    x_l, x_ab, cond, ab_pred = cinn_model.prepare_batch((input, target, filename, clear_input))
    x_ab_sampled, _ = cinn_model.reverse_sample(sample_z, cond)
    print(desc)
    print("Target:")
    target_img = show_data(input, F.interpolate(target[None, :], x_ab_sampled[0][0].shape)[0], filename, clear_input,
                           restore_audio, audio_file_name=f"result_audio_{filename[0]}")
    print("Result:")
    result_img = show_data(x_l[0], x_ab_sampled[0], filename, clear_input, restore_audio,
                           audio_file_name=f"result_audio_{filename[0]}")

    return [target_img, result_img]


def predict_cinn_example_self_sampled_test(cinn_model, cinn_output_dimensions, dataset, config, desc=None,
                                           restore_audio=False):
    example_id = random.randint(0, len(dataset) - 1)
    input, target, filename, clear_input = dataset[example_id]

    cinn_model.to(utilities.get_device(verbose=False))

    x_l, x_ab, cond, ab_pred = cinn_model.prepare_batch((input, target, filename, clear_input))

    cinn_input = torch.cat((x_l, x_ab), dim=1).to(utilities.get_device(verbose=False))
    cinn_model.eval()
    z, zz, jac = cinn_model(cinn_input)
    x_ab_sampled, _ = cinn_model.reverse_sample(z, cond)

    print(desc)
    print("Target:")
    target_img = show_data(input, F.interpolate(target[None, :], x_ab_sampled[0][0].shape)[0], filename, clear_input,
                           restore_audio, audio_file_name=f"result_audio_{filename[0]}")
    print("Result:")
    result_img = show_data(x_l[0], x_ab_sampled[0], filename, clear_input, restore_audio,
                           audio_file_name=f"result_audio_{filename[0]}")

    return [target_img, result_img]
