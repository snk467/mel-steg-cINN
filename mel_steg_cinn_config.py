from structured_config import ConfigFile, Structure
import os

class Config(Structure):

    # model_path = "/home/snikiel/Documents/REPOS/mgr_mel_steg_cinn/models/cinn_model_valiant-spaceship-529_512x512.pt"
    # model_path = "/home/snikiel/Documents/REPOS/mgr_mel_steg_cinn/models/steg_cinn_model_dark-night-54_3otrzb11.pt"
    model_path = "/home/snikiel/Documents/REPOS/mgr_mel_steg_cinn/models/steg_cinn_model_desert-waterfall-58_9rj1u4iz.pt"
    alpha = 0.1
    size = (512, 512)
    end_of_message_string = "_EOM"
    
    class cinnConfig(Structure):
        clamping = 1.0
        lr = 0.001
        lr_feature_net = 0.001
        n_epochs = 2
        n_its_per_epoch = 50
        weight_decay = 1.0e-5
        betas = (0.9, 0.999)
        init_scale = 0.03
        pre_low_lr = 0
        batch_size = 2
        dataset_size = 20
        pretrain_epochs = 0
        sampling_temperature = 1.0
        early_stopper_min_delta = 0.001
        early_stopper_patience = 10
        
    class audio_parameters(Structure):
        segment_length = 102200
        sample_rate = 22050
        hop_length = 200
        window_length = 2048
        n_mels = 512
        mean = -29.76158387351581
        standard_deviation = 16.546569945047125
        global_min = -1.3392828702926636
        global_max = 3.1347999572753906
    

config = Config()