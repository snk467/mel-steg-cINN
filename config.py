from structured_config import ConfigFile, Structure
from munch import Munch
import os

# This is just a config structure, the values are just placeholders.
# Please define your configuration in config_local.yml file.
class Config(Structure):

    class audio_parameters(Structure):

        class resolution_512x512(Structure):
            segment_length = 102200
            sample_rate = 22050
            hop_length = 200
            window_length = 2048
            n_mels = 512
            mean = -29.76158387351581
            standard_deviation = 16.546569945047125
            global_min = -1.3392828702926636
            global_max = 3.1347999572753906

        class resolution_256x256(Structure):
            segment_length = 51000
            sample_rate = 22050
            hop_length = 200
            window_length = 1024
            n_mels = 256
            mean = -35.93647618796097
            standard_deviation = 17.02300907660197
            global_min = -1.9601174592971802
            global_max = 2.7394044399261475

        class resolution_128x128(Structure):
            segment_length = 25400
            sample_rate = 22050
            hop_length = 200
            window_length = 800
            n_mels = 128
            mean = -37.36893585525392
            standard_deviation = 17.880760586984575
            global_min = -2.3468124866485596
            global_max = 2.127270221710205

        class resolution_80x80(Structure):
            segment_length = 20224
            sample_rate = 22050
            hop_length = 256
            window_length = 1024
            n_mels = 80
            mean = -34.110233672775486
            standard_deviation = 17.9092818952677
            global_min = -3.237250328063965
            global_max = 1.4316513538360596

    class unet_training(Structure):  
        dataset_size = 10 #If null then whole dataset is taken
        batch_size = 10
        epochs = 5
        lr = 0.001
        loss_function = "MSELoss"
        model = "unet_256"

    class cinn_training(Structure):
        feature_net_path = "/home/snikiel/Documents/REPOS/mel-steg-cINN/models/model_unet_256_None_melspectrograms_parula_norm_lab_13100_80x80_21102022181501.pt"
        batch_size = 10
        dataset_size = 100

    class common(Structure):
        present_data = False
        save_model = False
        batch_checkpoint = 1
        dataset_location = "/home/snikiel/Documents/STUDIA/EiTI/Praca_magisterska/Testy/Output/melspectrograms_parula_norm_lab_10.hdf5"
        noise_mean = [0.0]
        noise_variance = [0.001, 0.001, 0.0]
        sweep_count = 2  

    sweep_config = {

        'method': 'grid',

        'metric':
        {
            'goal': 'minimize',
            'name': 'VALID_AVG_Loss'
        },

        'parameters': {

            'dataset_size': {
                'value': 10
            },

            'batch_size': {
                'value': 10
            },

            'epochs':{
                'value': 5
            },

            'lr':{ 
                'values': [0.000075, 0.00005, 0.000025, 0.00001]
                # 'distribution': 'uniform',
                # 'max': 1e-2,
                # 'min': 1e-5
            },

            'enable_lr_scheduler':{
                'value': False
            },

            'loss_function':
            {
                'value': "MSELoss"
            },

            'model':
            {
                'value': "unet_256"
            }
        }
    }

LOCAL_CONFIG_FILE = "config.yml"
config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), LOCAL_CONFIG_FILE)

if os.path.exists(config_path) or __name__ == "__main__":
    config = Config(config_path)
else:
    config = Config()