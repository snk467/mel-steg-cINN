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

logger = helpers.logger.get_logger(__name__)

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
    z = melStegCinn.encode("Demo message :-)")
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
    
    # color <-> index
    indexes = colormap.get_indexes_from_colors(generated_melspectrogram.mel_spectrogram_data)
    generated_melspectrogram.mel_spectrogram_data = colormap.get_colors_from_indexes(indexes)
    
    input_melspectrogram = melStegCinn.get_melspectrogram_tensor(generated_melspectrogram)
    z_decode, _, _ = cinn_model(input_melspectrogram)
    
    print("Decode accuracy:", torch.sum(torch.sign(torch.cat(z, dim=1)) == torch.sign(torch.cat(z_decode, dim=1))) / torch.cat(z,dim=1).numel())
    
    melStegCinn.decode(z_decode)
    
    
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
        
        
        