from torchvision.transforms import ToPILImage
import PIL.Image as Image
import numpy as np
import math
import LUT
import torch
import torch.nn.functional as F
from IPython.display import display, Audio

import utilities
from config import config
import os
import soundfile
import random

def expand2size(pil_img, size):
    width, height = pil_img.size
    background_color = (0,0,0)

    assert width == height, "Image must be a square."
    assert size >= height, "Size must be greater than original image size."

    if width == size:
        return pil_img
    else:
        result = Image.new(pil_img.mode, (size, size), background_color)
        result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
        return result

def show_data(input_in, target_in, label, clear_input_in, restore_audio=False, audio_file_name=None):
    input = input_in.detach().cpu()
    target = target_in.detach().cpu()
    clear_input = clear_input_in

    print("L shape:", input.shape)
    print("ab shape:", target.shape)
    print("Label:", label)

    L_img = ToPILImage()(input).convert('RGB') 
    
    L_clear_img = ToPILImage()(clear_input).convert('RGB')       

    a_img = ToPILImage()(target[0]).convert('RGB') 

    b_img = ToPILImage()(target[1]).convert('RGB') 

    rgb_img = __get_rgb_image_from_lab_channels(clear_input, target)  
    
    max_size = max(L_img.size[0], L_clear_img.size[0], a_img.size[0], b_img.size[0])

    L_img = expand2size(L_img, max_size)
    
    L_clear_img = expand2size(L_clear_img, max_size)

    a_img = expand2size(a_img, max_size)

    b_img = expand2size(b_img, max_size)

    rgb_img = expand2size(rgb_img, max_size) 

    border_width = 10
    border = Image.fromarray(np.zeros((max_size, border_width))).convert('RGB')
    
    display(Image.fromarray(np.hstack((np.array(L_img),
                               np.array(border),
                               np.array(L_clear_img),
                               np.array(border),
                               np.array(a_img),
                               np.array(border),
                               np.array(b_img),
                               np.array(border),
                               np.array(rgb_img)))))
    
    if restore_audio:
        filename = f"restored_audio_{label}" if audio_file_name is None else audio_file_name
        __restore_audio(clear_input, target, audio_file_name = filename)
        display(Audio(f"{filename}.wav"))
   
def __get_colors_from_tensors(L_channel: torch.Tensor, ab_channels: torch.Tensor):
    colormap_lab = LUT.Colormap.from_colormap("parula_norm_lab")  

    result_size = L_channel.shape[1]

    L_channel = F.interpolate(L_channel[None, :], (result_size, result_size))[0]
    ab_channels = F.interpolate(ab_channels[None, :], (result_size, result_size))[0]

    L_np = L_channel.numpy()
    ab_np = ab_channels.numpy()      
    
    Lab_np = np.concatenate((L_np, ab_np))
    Lab_np = np.moveaxis(Lab_np, 0, -1)
    
    return Lab_np
    
def __restore_audio(L_channel: torch.Tensor, ab_channels: torch.Tensor, audio_file_name: str):
    colormap_lab = LUT.Colormap.from_colormap("parula_norm_lab")  
    Lab_np = __get_colors_from_tensors(L_channel, ab_channels)
    audio_config = config.audio_parameters.resolution_512x512
    melspectrogram = utilities.MelSpectrogram.from_color(Lab_np, True, colormap_lab, audio_config)    
    filepath = os.path.join(os.getcwd(), f"{audio_file_name}.wav")
    soundfile.write(filepath, melspectrogram.get_audio().audio, audio_config.sample_rate)     

def __get_rgb_image_from_lab_channels(L_channel: torch.Tensor, ab_channels: torch.Tensor):
    colormap_lab = LUT.Colormap.from_colormap("parula_norm_lab")      
    Lab_np = __get_colors_from_tensors(L_channel, ab_channels)    
    indexes = colormap_lab.get_indexes_from_colors(Lab_np)                            
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
    show_data(batched_input[0], output[0], filename, clear_input, restore_audio, audio_file_name=f"result_audio_{filename[0]}")
    print("Target:")
    show_data(input, target, filename, clear_input, restore_audio, audio_file_name=f"target_audio_{filename[0]}") 
    
def __sample_outputs(sigma, out_shape, batch_size):
    return [sigma * torch.FloatTensor(torch.Size((batch_size, o))).normal_().to(utilities.get_device(verbose=False)) for o in out_shape]
    
def predict_cinn_example(cinn_model, cinn_output_dimensions,dataset, config, desc=None, restore_audio=False):
    example_id = random.randint(0, len(dataset) - 1)
    input, target, filename, clear_input  = dataset[example_id]
    sample_z = __sample_outputs(config.sampling_temperature, cinn_output_dimensions, 1)
    x_l, x_ab, cond, ab_pred = cinn_model.prepare_batch((input, target, filename, clear_input))
    cond[1] = cond[1][None, :]
    x_ab_sampled, b = cinn_model.reverse_sample(sample_z, cond)   
    print(desc)
    print("Target:")
    show_data(input, F.interpolate(target[None, :], x_ab_sampled[0][0].shape)[0], filename, clear_input, restore_audio, audio_file_name=f"result_audio_{filename[0]}")
    print("Result:")
    show_data(x_l[0], x_ab_sampled[0], filename, clear_input, restore_audio, audio_file_name=f"result_audio_{filename[0]}")