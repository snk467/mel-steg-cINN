from torchvision.transforms import ToPILImage
import PIL.Image as Image
import numpy as np
import LUT

def show_batch(input_in, target_in, label, clear_input_in):
    input = input_in.detach().cpu()
    target = target_in.detach().cpu()
    clear_input = clear_input_in

    print("L shape:", input.shape)
    print("ab shape:", target.shape)
    print("Label:", label)

    L_img = ToPILImage(input).convert('RGB') 
    
    L_clear_img = ToPILImage(clear_input).convert('RGB')       

    a_img = ToPILImage(target[0]).convert('RGB') 

    b_img = ToPILImage(target[1]).convert('RGB') 

    rgb_img = __get_rgb_image_from_lab_channels(clear_input, target)  
    
    border_width = 10
    border = Image.fromarray(np.zeros((target.shape[1], border_width))).convert('RGB')
    
    Image.fromarray(np.hstack((np.array(L_img),
                               np.array(border),
                               np.array(L_clear_img),
                               np.array(border),
                               np.array(a_img),
                               np.array(border),
                               np.array(b_img),
                               np.array(border),
                               np.array(rgb_img)))).show()

def __get_rgb_image_from_lab_channels(L_channel, ab_channels):
    colormap_lab = LUT.Colormap.from_colormap("parula_norm_lab")  
    
    L_np = L_channel.numpy()
    ab_np = ab_channels.numpy()    
    
    Lab_np = np.concatenate((L_np, ab_np))
    Lab_np = np.moveaxis(Lab_np, 0, -1)
    
    indexes = colormap_lab.get_indexes_from_colors(Lab_np)                            
    colormap_rgb = LUT.Colormap.from_colormap("parula_rgb")
    img_target = colormap_rgb.get_colors_from_indexes(indexes)
    img_target = ToPILImage((img_target * 255).astype(np.uint8))
    
    return img_target