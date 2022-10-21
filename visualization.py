from torchvision.transforms import ToPILImage
import PIL.Image as Image
import numpy as np
import math
import LUT
import torch
import torch.nn.functional as F

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

def show_data(input_in, target_in, label, clear_input_in):
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
    
    Image.fromarray(np.hstack((np.array(L_img),
                               np.array(border),
                               np.array(L_clear_img),
                               np.array(border),
                               np.array(a_img),
                               np.array(border),
                               np.array(b_img),
                               np.array(border),
                               np.array(rgb_img)))).show()

def __get_rgb_image_from_lab_channels(L_channel: torch.Tensor, ab_channels: torch.Tensor):
    colormap_lab = LUT.Colormap.from_colormap("parula_norm_lab")  

    result_size = max(L_channel.shape[1], ab_channels.shape[1])

    L_channel = F.interpolate(L_channel[None, :], (result_size, result_size))[0]
    ab_channels = F.interpolate(ab_channels[None, :], (result_size, result_size))[0]

    L_np = L_channel.numpy()
    ab_np = ab_channels.numpy()      
    
    Lab_np = np.concatenate((L_np, ab_np))
    Lab_np = np.moveaxis(Lab_np, 0, -1)
    
    indexes = colormap_lab.get_indexes_from_colors(Lab_np)                            
    colormap_rgb = LUT.Colormap.from_colormap("parula_rgb")
    img_target = colormap_rgb.get_colors_from_indexes(indexes)
    img_target = ToPILImage()((img_target * 255).astype(np.uint8))    
    
    return img_target