from structured_config import Structure


class Config(Structure):
    cinn_model_path = "/home/snikiel/Documents/REPOS/mgr_mel_steg_cinn/data/saved_models/cinn_model_solar-planet-543.pt"
    compression_cinn_model_path = "/home/snikiel/Documents/REPOS/mgr_mel_steg_cinn/data/saved_models" \
                                  "/steg_cinn_model_fine-breeze-99.pt"

    dataset_path = "/home/snikiel/Documents/REPOS/mgr_mel_steg_cinn/data/dataset" \
                   "/melspectrograms_parula_norm_lab_13100_512x512.hdf5"
    alpha = 0.1
    batch_size = 2

    class cinn(Structure):
        clamping = 1.0
        init_scale = 0.03
        output_dim = (512, 512)


config = Config()
