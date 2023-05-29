import numpy as np
import torch
import wandb
import os
import sys
import argparse
from PIL import Image

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import models.cinn.cinn_model
import utils.logger
import utils.metrics as metrics
from datasets import TestDataset
from utils import utilities
import experiments_config
import demo_app.demo_app_utils as demo_app_utils
from trainings.train_steg_cINN import compress_melspectrograms


logger = utils.logger.get_logger(__name__)


def hide(cond, z, cinn_model: models.cinn.cinn_model.WrappedModel):
    return cinn_model.reverse_sample(z, cond)[0]


def reveal(x: torch.Tensor, cinn_model: models.cinn.cinn_model.WrappedModel):
    z, _, _ = cinn_model(x)
    return torch.cat(z, dim=1).squeeze().detach()


def transfer(x_l, x_ab_pred, compress):

    if compress:
        mel_spectrogram = compress_melspectrograms(torch.cat((x_l, x_ab_pred), dim=1).detach().cpu())
    else:
        mel_spectrogram = torch.cat((x_l, x_ab_pred), dim=1).detach()

    return mel_spectrogram


def run_experiment(binary, compress, config: experiments_config.config):
    device = utilities.get_device()

    with wandb.init(project="cinn-experiments", entity="snikiel", config=config):
        test_set = TestDataset(config.dataset_path,
                               output_dim=config.cinn.output_dim)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

        cinn_model, cinn_output_dimensions = demo_app_utils.load_cinn(compress, config, device=device)

        wandb.log({"compress": compress})
        logger.info(f"compress: {compress}")

        if binary is None:
            wandb.log({"z_target": "random"})
            logger.info("z_target: random")
            z_target = utilities.sample_z(cinn_output_dimensions, config.batch_size, alpha=config.alpha, device=device)
        else:
            wandb.log({"bin_input_image": wandb.Image(binary[1])})
            wandb.log({"z_target": binary[2]})
            logger.info(f"z_target: {binary[2]}")
            z_target = demo_app_utils.encode(binary[0], cinn_output_dimensions, config)
            for i, z in enumerate(z_target):
                z_target[i] = z.repeat(config.batch_size, 1)

        z_target_squeezed = torch.cat(z_target, dim=1).squeeze().detach()

        mse_ab_all = []
        mae_ab_all = []
        mse_ab_after_compression_all = []
        mae_ab_after_compression_all = []
        mse_z_all = []
        acc_z_all = []

        for i_batch, x in enumerate(test_loader):
            
            if x[0].size()[0] != config.batch_size:
                continue
            
            x_l, x_ab_target, cond, _ = cinn_model.prepare_batch((*x, None))
            cinn_model.eval()

            x_ab_pred = hide(cond, z_target, cinn_model)

            mel_spectrogram = transfer(x_l.to(device), x_ab_pred, compress)

            x_ab_pred_compressed = mel_spectrogram[:, 1:]

            z_pred = reveal(mel_spectrogram.to(device), cinn_model)

            mse_ab = metrics.mse(x_ab_pred.to('cpu'), x_ab_target.to('cpu')).detach().numpy()
            mae_ab = metrics.mae(x_ab_pred.to('cpu'), x_ab_target.to('cpu')).detach().numpy()
            mse_ab_after_compression = metrics.mse(x_ab_pred_compressed.to('cpu'), x_ab_target.to('cpu')).detach().numpy()
            mae_ab_after_compression = metrics.mae(x_ab_pred_compressed.to('cpu'), x_ab_target.to('cpu')).detach().numpy()
            mse_z = metrics.mse(z_pred.to('cpu'), z_target_squeezed.to('cpu')).detach().numpy()
            acc_z = 1.0 - metrics.accuracy(z_pred.to('cpu'), z_target_squeezed.to('cpu')).detach().numpy()

            logger.info(f"Batch {i_batch + 1}: " 
                        f"MSE_ab: {mse_ab}, "
                        f"MAE_ab: {mae_ab}, "
                        f"MSE_ab (after compression): {mse_ab_after_compression}, "
                        f"MAE_ab (after compression): {mae_ab_after_compression}, "
                        f"MSE_z: {mse_z}, "
                        f"Accuracy_z: {acc_z}")

            wandb.log({"mse_ab": mse_ab,
                       "mae_ab": mae_ab,
                       "mse_ab_after_compression": mse_ab_after_compression,
                       "mae_ab_after_compression": mae_ab_after_compression,
                       "mse_z": mse_z,
                       "acc_z": acc_z})

            mse_ab_all.append(mse_ab)
            mae_ab_all.append(mae_ab)
            mse_ab_after_compression_all.append(mse_ab)
            mae_ab_after_compression_all.append(mae_ab)
            mse_z_all.append(mse_z)
            acc_z_all.append(acc_z)

        logger.info(f"Mean MSE_ab: {np.mean(mse_ab_all)}, "
                    f"Mean MAE_ab: {np.mean(mae_ab_all)}, "
                    f"Mean MSE_ab (after compression): {np.mean(mse_ab_after_compression_all)}, " 
                    f"Mean MAE_ab (after compression): {np.mean(mae_ab_after_compression_all)}, " 
                    f"Mean MSE_z: {np.mean(mse_z_all)}, "
                    f"Mean Accuracy_z: {np.mean(acc_z_all)}")

        wandb.log({"avg_mse_ab": np.mean(mse_ab_all),
                   "avg_mae_ab": np.mean(mae_ab_all),
                   "avg_mse_ab_after_compression": np.mean(mse_ab_after_compression_all),
                   "avg_mae_ab_after_compression": np.mean(mae_ab_after_compression_all),
                   "avg_mse_z": np.mean(mse_z_all),
                   "avg_acc_z": np.mean(acc_z_all)})

        wandb.finish()


def get_bin_list(input):
    if input is None:
        return None

    if not os.path.isdir(input):
        raise ValueError(f"{input} is not a directory!")

    binary_list = []

    for path in os.listdir(input):
        if os.path.isfile(os.path.join(input, path)):
            binary_list.append((demo_app_utils.image_to_bin(Image.open(os.path.join(input, path)))[0], Image.open(os.path.join(input, path)), path))

    return binary_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", help="Directory with binary images", type=str, default=None)
    parser.add_argument("-c", "--compress", help="Simulate compression", action='store_true')

    args = parser.parse_args()

    bin_list = get_bin_list(args.input)

    if bin_list is None:
        run_experiment(None, args.compress, experiments_config.config)
    else:
        for binary in bin_list:
            run_experiment(binary, args.compress, experiments_config.config)
