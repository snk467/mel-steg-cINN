import os
import torch
import cinn.cinn_model
from exceptions import ArgumentError
from mel_steg_cinn_config import config
import mel_steg_cinn_config
import argparse
import helpers.utilities as utilities
import LUT
import numpy as np
import bitarray
import helpers.logger
import helpers.normalization
import soundfile
import matplotlib.pyplot as plt
from helpers.utilities import MelStegCinn
from PIL import Image
import scipy.cluster.vq as scipy_vq
import helpers.visualization as visualization

logger = helpers.logger.get_logger(__name__)

LOGO_IMAGE = "double-logo.png"

def main(args):
    
    if args.demo:
        demo(args)  
        return  
    
    if args.reveal:
        reveal(args)
    else:
        hide(args)
        
    # If reveal
        # Get input audio
        # Get melspectrogram from input
        # Send L, a, b channels to cINN
            # Get noise
        # Decode from noise
        # Print message  
    # Else
        # Get message from input
        # Get container audio
            # Get L channel of melspectrogram
        # Encode into noise
        # Send noise and L channel to cINN
            # Get a, b channels
        # Save Lab channels as melspectrogram
        # Convert melspectrogram to audio
        # Save audio

def hide(args):
    melStegCinn = MelStegCinn(config)
    cinn_model = melStegCinn.load_cinn()
    cinn_model.eval()
    audio = melStegCinn.load_audio(args.audio)
    colormap = LUT.Colormap.from_colormap("parula_norm_lab")
    
    melspectrogram = audio.get_color_mel_spectrogram(True, colormap=colormap)
    z = melStegCinn.encode(args.message)
    L_channel = melStegCinn.get_L_channel(melspectrogram)
    cond = melStegCinn.get_cond(L_channel)
    ab_channels = cinn_model.reverse_sample(z, cond)
    generated_melspectrogram = utilities.MelSpectrogram.from_color(torch.cat([L_channel, ab_channels[0]], dim=1).
                                                                        squeeze().
                                                                        permute((1,2,0)).
                                                                        detach().
                                                                        numpy(),
                                                                    normalized=True,
                                                                    colormap=colormap,
                                                                    config=config.audio_parameters)
    
    # Convert melspectrogram to audio
    generated_audio = generated_melspectrogram.get_audio()
    
    # Save audio
    soundfile.write(os.path.join(os.getcwd() if args.output is None else args.output,
                    f"{os.path.basename(args.audio).split('.')[0]}_with_message.wav"),
                    generated_audio.get_audio(),
                    config.audio_parameters.sample_rate)      

    
def reveal(args):
    melStegCinn = MelStegCinn(config)
    cinn_model = melStegCinn.load_cinn()
    cinn_model.eval()    
    audio = melStegCinn.load_audio(args.audio)
    colormap = LUT.Colormap.from_colormap("parula_norm_lab")
    melspectrogram = audio.get_color_mel_spectrogram(True, colormap=colormap)
    input_melspectrogram = melStegCinn.get_melspectrogram_tensor(melspectrogram)
    z, _, _ = cinn_model(input_melspectrogram)
    melStegCinn.decode(z)
    
def demo(args):
    melStegCinn = MelStegCinn(config)
    cinn_model = melStegCinn.load_cinn()
    cinn_model.eval()
    audio = melStegCinn.load_audio(args.audio)
    colormap = LUT.Colormap.from_colormap("parula_norm_lab")
    
    melspectrogram = audio.get_color_mel_spectrogram(True, colormap=colormap)
    
    soundfile.write(os.path.join(os.getcwd() if args.output is None else args.output,
                    f"{os.path.basename(args.audio).split('.')[0]}_base.wav"),
                    melspectrogram.get_audio().get_audio(),
                    config.audio_parameters.sample_rate)   
    
    visualization.get_rgb_image_from_lab_channels(melStegCinn.get_L_channel(melspectrogram), melStegCinn.get_ab_channels(melspectrogram), colormap=colormap).save("melspectrogram_without_message.png", "PNG")

    
    bin = utilities.image_to_bin(Image.open(LOGO_IMAGE))    
    z = melStegCinn.encode(bin)
    L_channel = melStegCinn.get_L_channel(melspectrogram)
    cond = melStegCinn.get_cond(L_channel)
    ab_channels = cinn_model.reverse_sample(z, cond)
    
    
    
    # # color <-> index
    # indexes = colormap.get_indexes_from_colors(generated_melspectrogram.mel_spectrogram_data)
    # generated_melspectrogram.mel_spectrogram_data = colormap.get_colors_from_indexes(indexes)
    
    audio = melspectrogram_tensor_to_audio(torch.cat([L_channel, ab_channels[0].detach()], dim=1), colormap)
    
    soundfile.write(os.path.join(os.getcwd() if args.output is None else args.output,
                    f"{os.path.basename(args.audio).split('.')[0]}_with_logo.wav"),
                    audio.get_audio(),
                    config.audio_parameters.sample_rate)   
    
    visualization.get_rgb_image_from_lab_channels(L_channel, ab_channels[0].detach(), colormap=colormap).save("melspectrogram_with_message.png", "PNG")
    
    input_melspectrogram, colormaps=compress_melspectrograms2(torch.cat([L_channel, ab_channels[0].detach()], dim=1)) # melStegCinn.get_melspectrogram_tensor(generated_melspectrogram)
    
    audio = melspectrogram_tensor_to_audio(input_melspectrogram, colormaps[0])
    
    soundfile.write(os.path.join(os.getcwd() if args.output is None else args.output,
                    f"{os.path.basename(args.audio).split('.')[0]}_compressed_and_with_logo.wav"),
                    audio.get_audio(),
                    config.audio_parameters.sample_rate)   
    
    visualization.get_rgb_image_from_lab_channels(L_channel, ab_channels[0].detach(), colormap=colormaps[0]).save("melspectrogram_compressed_and_with_message.png", "PNG")
    
    z_decode, _, _ = cinn_model(input_melspectrogram.double())
    
    print("Decode accuracy:", torch.sum(torch.sign(torch.cat(z, dim=1)) == torch.sign(torch.cat(z_decode, dim=1))) / torch.cat(z,dim=1).numel())
    
    bin_result = melStegCinn.decode(z_decode)
    
    print(len(bin_result))
    
    hidden_image = utilities.bin_to_image(bin_result)
    
    hidden_image.show()
    
def melspectrogram_tensor_to_audio(melspectrogram: torch.Tensor, colormap):
    melspectrogram_object = utilities.MelSpectrogram.from_color(melspectrogram.
                                                                        squeeze().
                                                                        permute((1,2,0)).
                                                                        detach().
                                                                        numpy(),
                                                                    normalized=True,
                                                                    colormap=colormap,
                                                                    config=config.audio_parameters)
    
    return melspectrogram_object.get_audio()

def compress_melspectrograms(mel_spectrograms: torch.Tensor):    
    colormap = LUT.ColormapTorch.from_colormap("parula_norm_lab")
    
    indexes = colormap.get_indexes_from_colors(mel_spectrograms)
    
    return colormap.get_colors_from_indexes(indexes)

def compress_melspectrograms2(mel_spectrograms: torch.Tensor):
    
    result_mel_spectrograms = []
    result_colormaps = []
    
    for i in range(mel_spectrograms.shape[0]):
        mel_spectrogram = mel_spectrograms[i]
        shape = mel_spectrogram.shape
        
        # (3, 512, 512) -> (512^2, 3)
        
        # assert (mel_spectrogram.numpy().reshape((shape[1]*shape[2], shape[0])).reshape(shape) == mel_spectrogram.numpy()).all()
        
        mel_spectrogram = mel_spectrogram.numpy().transpose(1,2,0).reshape((shape[1]*shape[2], shape[0]))
        
        # print(mel_spectrogram.shape)
        # print(np.array(LUT.Colormap.colormaps["parula_norm_lab"]).shape)
        
        centroids, labels = scipy_vq.kmeans2(mel_spectrogram, np.array(LUT.Colormap.colormaps["parula_norm_lab"]))
        
        colormap = LUT.Colormap.from_colors_list(centroids)
        indexes = labels.reshape((shape[1], shape[2]))
        
        result_mel_spectrograms.append(torch.Tensor(colormap.get_colors_from_indexes(indexes))[None, :].permute((0, 3, 1, 2)))   
        result_colormaps.append(colormap)
        
        # print(result_mel_spectrograms[-1])
        # print(mel_spectrograms[-1])
        
        # index = (np.random.randint(512), np.random.randint(512))
        # print(result_mel_spectrograms[-1][:, :, index[0], index[1]])
        # print(mel_spectrograms[i][:, index[0], index[1]])
        
        # print(mse_loss(result_mel_spectrograms[-1], mel_spectrograms[i][None, :]).item())
        
    return torch.cat(result_mel_spectrograms, dim=0), result_colormaps
    
    
# region W budowie
# def get_original_mel_spectrogram_data(mel_spectrogram: utilities.MelSpectrogram):
#     mel_spectrogram_data = mel_spectrogram.mel_spectrogram_data
#     local_config = config.audio_parameters
    
#     if mel_spectrogram_data is None:
#         return None
    
#     colormap = LUT.Colormap.from_colormap("parula_norm_lab")

#     # Convert color back to values
#     if colormap is not None:
#         # colormap = Colormap.from_colormap(self.colormap)
#         mel_spectrogram_data = colormap.get_values_from_colors(mel_spectrogram_data)

#     # Inverse scaling
#     if mel_spectrogram.range is not None:
#         mel_spectrogram_data = helpers.normalization.scale_global_minmax(mel_spectrogram_data, min(mel_spectrogram.range), max(mel_spectrogram.range), local_config.global_min, local_config.global_max)

#     # Inverse normalization
#     if mel_spectrogram.normalized:
#         mel_spectrogram_data = helpers.normalization.normalize(mel_spectrogram_data, local_config.mean, local_config.standard_deviation, inverse=True)

#     return mel_spectrogram_data

# get_color_mel_spectrogram_from_original(mel_spectrogram_data )
# endregion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mel-steg-cINN v1.0')
    parser.add_argument('-t', '--test', action="store_true")
    parser.add_argument('-r', '--reveal', action="store_true")
    parser.add_argument('--demo', action="store_true")
    parser.add_argument('-m', '--message', type=str)
    parser.add_argument('-a', '--audio', type=str)
    parser.add_argument('-o', '--output', type=str, default=None)
    args = parser.parse_args()
    
    if not args.test:
        if args.audio is None:
            raise ArgumentError
        main(args)
    else:
        helpers.logger.enable_debug_mode()
        melStegCinn = MelStegCinn(config)
        melStegCinn.load_cinn()
        z = melStegCinn.encode(message="Hello world!_EOM")
        melStegCinn.decode(z)
        
        
        