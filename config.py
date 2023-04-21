import os

from structured_config import Structure


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
        input_dims = (64, 64)
        dataset_size = 10  # If null then whole dataset is taken
        batch_size = 10
        epochs = 5
        lr = 0.001
        loss_function = "MSELoss"
        model = "unet_256"

    class cinn_training(Structure):
        clamping = 1.5  # Clamping parameter in the coupling blocks (higher = less stable but more expressive)
        alpha = 0.1
        # Training hyperparameters: 
        lr = 5e-6
        lr_feature_net = 1e-4
        n_epochs = 5  # 120 * 4
        n_its_per_epoch = 32 * 8  # In case the epochs should be cut short after n iterations
        weight_decay = 1e-5
        betas = (0.9, 0.999)  # concerning adam optimizer
        init_scale = 0.030  # initialization std. dev. of weights (0.03 is approx xavier)
        pre_low_lr = 0  # for the first n epochs, lower the lr by a factor of 20
        batch_size = 2
        dataset_size = 20
        pretrain_epochs = 0
        sampling_temperature = 1.0
        early_stopper_min_delta = 0.001
        early_stopper_patience = 10

    class steg_cinn_training(Structure):
        clamping = 1.5  # Clamping parameter in the coupling blocks (higher = less stable but more expressive)
        alpha = 0.1
        mse_ab_importance = 1.0
        mse_z_importance = 1.5
        l_importance = 0.25
        rewrite_models = True
        # Training hyperparameters: 
        lr = 5e-6
        lr_feature_net = 1e-4
        n_epochs = 5  # 120 * 4
        n_its_per_epoch = 32 * 8  # In case the epochs should be cut short after n iterations
        weight_decay = 1e-5
        betas = (0.9, 0.999)  # concerning adam optimizer
        init_scale = 0.030  # initialization std. dev. of weights (0.03 is approx xavier)
        pre_low_lr = 0  # for the first n epochs, lower the lr by a factor of 20
        batch_size = 2
        dataset_size = 20
        pretrain_epochs = 0
        sampling_temperature = 1.0
        early_stopper_min_delta = 0.001
        early_stopper_patience = 10

    class cinn_management(Structure):
        # Architecture: 
        img_dims = (80, 80)  # Image size of L, and ab channels respectively

        # Files and checkpoints management:
        end_to_end = False
        feature_net_path = "/path/to/feature/net/model.pt"
        filename = 'output/full_model.pt'  # output filename
        load_file = False
        checkpoint_save_interval = 60
        checkpoint_save_overwrite = False  # Whether to overwrite the old checkpoint with the new one
        checkpoint_on_error = True  # Wheter to make a checkpoint with suffix _ABORT if an error occurs

    class common(Structure):
        present_data = False
        save_model = False
        batch_checkpoint = 1
        dataset_location = "/path/to/melspectrograms/dataset.hdf5"
        noise_mean = [0.0]
        noise_variance = [0.001, 0.001, 0.0]
        sweep_count = 2
        wandb_key = "wandb_key"

    unet_sweep_config = {

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

            'epochs': {
                'value': 5
            },

            'lr': {
                'values': [0.000075, 0.00005, 0.000025, 0.00001]
                # 'distribution': 'uniform',
                # 'max': 1e-2,
                # 'min': 1e-5
            },

            'enable_lr_scheduler': {
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
    config = Config(config_path, )
else:
    config = Config()
