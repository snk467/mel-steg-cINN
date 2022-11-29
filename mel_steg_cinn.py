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
    input_melspectrogram = melStegCinn.get_melspectrogram_tensor(generated_melspectrogram)
    z, _, _ = cinn_model(input_melspectrogram)
    melStegCinn.decode(z)
    
    
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

class MelStegCinn:
    def __init__(self, config: mel_steg_cinn_config.Config):
        self.config = config
        self.cinn_utilities = None
        self.cinn_z_dimensions = None
        pass
    
    def load_cinn(self):
        cinn_builder = cinn.cinn_model.cINN_builder(self.config.cinnConfig)
        feature_net = cinn_builder.get_feature_net().double()
        fc_cond_net = cinn_builder.get_fc_cond_net().double()
        cinn_net, self.cinn_z_dimensions = cinn_builder.get_cinn()
        cinn_model = cinn.cinn_model.WrappedModel(feature_net, fc_cond_net, cinn_net.double())
        self.cinn_utilities = cinn.cinn_model.cINNTrainingUtilities(cinn_model, self.config.cinnConfig)
        self.cinn_utilities.load(self.config.model_path) 
        
        return self.cinn_utilities.model
        
    def load_audio(self, path: str):
        return utilities.Audio(utilities.load_audio(path)[0], self.config.audio_parameters)
    
    def encode(self, message: str):
        bin_message = bitarray.bitarray()
        bin_message.frombytes((message + config.end_of_message_string).encode('ascii'))
        
        desired_size =  sum([x for x in self.cinn_z_dimensions])
        
        z = []
        
        for bit in bin_message:
            sample = np.random.normal()
            if bit == 0:
                while not (sample < -np.abs(self.config.alpha)):
                    sample = np.random.normal() 
            else:
                while not (sample > np.abs(self.config.alpha)):
                    sample = np.random.normal()
                    
            z.append(sample) 
            
            if len(z) == desired_size:
                break
            
        if len(z) != desired_size:
            z.extend(np.random.normal(size=desired_size-len(z)))
            
        logger.debug(f"Z length: {len(z)}")
        
        # plt.hist(z, bins=100)
        # plt.show()
        
        z = torch.from_numpy(np.array(z))
        z = list(z.split(self.cinn_z_dimensions))
        
        logger.debug(f"Z length after split: {len(z)}")
        for i in range(len(z)):
            logger.debug(f"z[{i}].shape: {z[i].shape}")
            z[i] = z[i][None, :]
            logger.debug(f"z[{i}].shape(corrected): {z[i].shape}")
            
        logger.info(f"Encoded message: {message}")
        
        return z
    
    def decode(self, z: list[torch.Tensor]):
        
        logger.debug(f"Z input length: {len(z)}")
        z = torch.cat(z, dim=1).squeeze().detach().numpy()
        
        # plt.hist(z, bins=100)
        # plt.show()
        
        logger.debug(f"Z concatenated shape: {z.shape}")
        
        bin_message = []
        
        for sample in z:
            if sample < -np.abs(self.config.alpha):
                bin_message.append(False)
            elif sample > np.abs(self.config.alpha):
                bin_message.append(True)
            
        message_with_noise = bitarray.bitarray(bin_message).tobytes().decode('ascii', errors='replace')
        message = message_with_noise.split(config.end_of_message_string)[0]
        logger.info(f"Decoded message: {message[:100]}")        
    
    def get_L_channel(self, melspectrogram: utilities.MelSpectrogram):
        # Load tensor
        L = torch.from_numpy(melspectrogram.mel_spectrogram_data[:, :, 0])
        # Adjust axies 
        L = torch.reshape(L, (1, 1, L.shape[0], L.shape[1]))
        return L
    
    def get_melspectrogram_tensor(self, melspectrogram: utilities.MelSpectrogram):
        # Load tensor
        spectrogram_data = torch.from_numpy(melspectrogram.mel_spectrogram_data)
        # Adjust axies 
        spectrogram_data = torch.permute(spectrogram_data, (2, 0, 1))[None, :]
        
        return spectrogram_data
        
    
    def get_cond(self, L_channel: torch.Tensor):
        with torch.no_grad():
            features, _ = self.cinn_utilities.model.feature_network.features(L_channel)
            return [*features]
        

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
        
        
        