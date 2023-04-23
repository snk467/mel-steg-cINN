from structured_config import Structure


class Config(Structure):
    cinn_model_path = "/home/snikiel/Documents/REPOS/mgr_mel_steg_cinn/data/saved_models/cinn_model_solar-planet-543.pt"
    compression_cinn_model_path = "/home/snikiel/Documents/REPOS/mgr_mel_steg_cinn/data/saved_models" \
                                  "/steg_cinn_model_fine-breeze-99.pt"
    alpha = 0.1
    size = (512, 512)
    end_of_message_string = "_EOM"

    class bch(Structure):
        mi = 2
        tau = 1  # Error correcting capability

    class cinn(Structure):
        clamping = 1.0
        init_scale = 0.03

    class audio(Structure):
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
