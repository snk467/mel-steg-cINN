#!/usr/bin/env python
import argparse
import os
from math import ceil

import numpy as np
import torch
import torch.nn
import torch.optim
import torchmetrics as torch_metrics
import wandb

import models.cinn.cinn_model
import models.cinn.cinn_model as model
import utils.logger
import utils.utilities as utilities
from config import config as main_config
from datasets import SpectrogramsDataset
from utils.metrics import Metrics
from utils.noise import GaussianNoise
from utils.visualization import predict_cinn_example, predict_cinn_example_self_sampled_test

# Get logger
global logger
logger = utils.logger.get_logger(__name__)

MODEL_FILE_NAME = "cinn_model.pt"


def prepare_training():
    global device
    device = utilities.get_device()

    global mse_loss
    mse_loss = torch.nn.MSELoss().to(device)

    # Set metrics functions   
    metrics_functions = {
        "MSE": torch_metrics.MeanSquaredError().to(device),
        "HuberLoss": torch.nn.HuberLoss().to(device),
        "MAE": torch_metrics.MeanAbsoluteError().to(device)
    }

    global metrics
    metrics = Metrics(metrics_functions)

    global tot_output_size
    tot_output_size = 2 * main_config.cinn_management.img_dims[0] * main_config.cinn_management.img_dims[1]


def loss(x_ab_pred, x_ab_pred_feature_net, zz, jac):
    mse = mse_loss(x_ab_pred, x_ab_pred_feature_net)

    neg_log_likeli = 0.5 * zz - jac

    l = torch.mean(neg_log_likeli) / tot_output_size

    mse_importance = 0.75

    return (1.0 - mse_importance) * l + mse_importance * mse, mse.item(), l.item()


def sample_outputs(sigma, out_shape, batch_size):
    return [sigma * torch.FloatTensor(torch.Size((batch_size, o))).normal_().to(device) for o in out_shape]


def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return torch.stack(list(tuple_of_tensors), dim=0)


def validate(cinn_model, cinn_output_dimensions, config, validation_loader):
    cinn_model.eval()

    avg_metrics = None

    for i, vdata in enumerate(validation_loader):
        if vdata[0].shape[0] != config.batch_size:
            break

        x_l, x_ab_target, cond, ab_pred = cinn_model.prepare_batch(vdata)

        z = utilities.sample_z(cinn_output_dimensions, config.batch_size, alpha=config.alpha)

        x_ab_output = cinn_model.reverse_sample(z, cond)

        _, batch_metrics = metrics.gather_batch_metrics(x_ab_output[0], x_ab_target.to(device))

        avg_metrics = metrics.add_metrics(avg_metrics, batch_metrics)

    avg_metrics = metrics.divide_metrics(avg_metrics, len(validation_loader))

    return avg_metrics


def train_one_epoch(cinn_model, training_loader, config, i_epoch, step,
                    cinn_training_utilities: model.cINNTrainingUtilities, cinn_output_dimensions):
    cinn_model.train()
    avg_loss = []
    batch_checkpoint = ceil(min(len(training_loader) / 10, config.n_its_per_epoch / 10, 3))

    for i_batch, x in enumerate(training_loader):

        x_l, x_ab_target, cond, _ = cinn_model.prepare_batch(x)

        # L, ab, _, _ = x  

        input = torch.cat((x_l, x_ab_target), dim=1).to(device)

        _, zz, jac = cinn_model(input)

        z = utilities.sample_z(cinn_output_dimensions, config.batch_size, alpha=config.alpha)
        cinn_model.eval()
        x_ab_pred = cinn_model.reverse_sample(z, cond)

        cinn_model.train()

        train_loss, mse, nll = loss(x_ab_pred[0], x_ab_target.to(device), zz, jac)
        train_loss.backward()
        cinn_training_utilities.optimizer_step()

        # Report
        if i_batch % batch_checkpoint == (batch_checkpoint - 1):
            step += 1
            metrics.log_metrics({'batch_loss': train_loss.item(), 'mse': mse, 'nll': nll}, "train", step, i_batch)

        avg_loss.append(train_loss.item())

        if i_batch + 1 >= config.n_its_per_epoch:
            break

    return np.mean(avg_loss), step


def train(config=None, load=None):
    with wandb.init(project="cINN", entity="snikiel", config=config):
        wandb.log({"main_setup": main_config.to_dict()})
        prepare_training()
        config = wandb.config
        logger.info(config)
        training_set = SpectrogramsDataset(main_config.common.dataset_location,
                                           train=True,
                                           size=config.dataset_size,
                                           augmentor=GaussianNoise(main_config.common.noise_mean,
                                                                   main_config.common.noise_variance),
                                           output_dim=main_config.cinn_management.img_dims)

        validation_set = SpectrogramsDataset(main_config.common.dataset_location,
                                             train=False,
                                             size=config.dataset_size,
                                             augmentor=GaussianNoise(main_config.common.noise_mean,
                                                                     main_config.common.noise_variance),
                                             output_dim=main_config.cinn_management.img_dims)

        # Create data loaders for our datasets; shuffle for training, not for validation
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=config.batch_size, shuffle=True,
                                                      num_workers=2)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config.batch_size, shuffle=False,
                                                        num_workers=2)

        early_stopper = model.EarlyStopper(patience=config.early_stopper_patience,
                                           min_delta=config.early_stopper_min_delta)

        cinn_training_utilities, cinn_output_dimensions = models.cinn.cinn_model.get_cinn_model(config, MODEL_FILE_NAME, load,
                                                                                                device=device)
        cinn_model = cinn_training_utilities.model

        logger.info(f"Training feature net: {main_config.cinn_management.end_to_end}")

        logger.debug(f"cinn model device: {next(cinn_model.parameters()).device}")

        step = 0

        for i_epoch in range(config.n_epochs):
            logger.info('EPOCH {}:'.format(i_epoch + 1))
            logger.info("       Model training.")
            avg_loss, step = train_one_epoch(cinn_model, training_loader, config, i_epoch, step,
                                             cinn_training_utilities, cinn_output_dimensions)
            metrics.log_metrics({'loss': avg_loss}, "TRAIN AVG", step)

            logger.info("       Model validation.")
            avg_metrics = validate(cinn_model, cinn_output_dimensions, config, validation_loader)
            # Report
            metrics.log_metrics(avg_metrics, "VALID AVG", step)
            cinn_training_utilities.scheduler_step(avg_loss)

            if early_stopper.early_stop(avg_metrics["MSE"]):
                break

        cinn_model.eval()
        logger.info("Generating examples.")
        training_examples = predict_cinn_example(cinn_model, cinn_output_dimensions, training_set, config,
                                                 desc="Training set example", restore_audio=False)
        wandb.log({"training_examples": [wandb.Image(image) for image in training_examples]})
        print()
        validation_examples = predict_cinn_example(cinn_model, cinn_output_dimensions, validation_set, config,
                                                   desc="Validation set example", restore_audio=False)
        wandb.log({"validation_examples": [wandb.Image(image) for image in validation_examples]})
        print()
        overfitting_examples = predict_cinn_example_self_sampled_test(cinn_model, cinn_output_dimensions, training_set,
                                                                      config, desc="Training set self-sampled example",
                                                                      restore_audio=False)
        wandb.log({"self-sampled_examples": [wandb.Image(image) for image in overfitting_examples]})

        # Save model
        model_path = None
        if main_config.common.save_model:
            logger.info("Saving cINN model.")
            model_path = os.path.join(os.getcwd(), MODEL_FILE_NAME)
            cinn_training_utilities.save(model_path)
            wandb.save(model_path, base_path=os.getcwd(), policy="now")

        wandb.finish()

        if model_path is not None:
            os.remove(model_path)

    # os.makedirs(os.path.dirname(config.filename), exist_ok=True)
    # model.save(config.filename)


def run(config, load=None):
    # Initialize Weights & Biases
    wandb.login(key=main_config.common.wandb_key)
    train(config, load)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train cINN for mel-spectrogram colorization.')
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--dataset_size', type=int, required=False)
    parser.add_argument('--betas', type=tuple, required=False)
    parser.add_argument('--clamping', type=float, required=False)
    parser.add_argument('--init_scale', type=float, required=False)
    parser.add_argument('--lr', type=float, required=False)
    parser.add_argument('--lr_feature_net', type=float, required=False)
    parser.add_argument('--n_epochs', type=int, required=False)
    parser.add_argument('--n_its_per_epoch', type=int, required=False)
    parser.add_argument('--sampling_temperature', type=float, required=False)
    parser.add_argument('--weight_decay', type=float, required=False)
    parser.add_argument('--load', type=str, required=False)
    parser.add_argument('--early_stopper_min_delta', type=float, required=False)
    args = parser.parse_args()

    sweep_args_dict = {k: vars(args)[k] for k in vars(args).keys() - {'load'}}
    sweep_args_present = all(value is not None for value in sweep_args_dict.values())

    if sweep_args_present:
        run(args, args.load)
    else:
        run(main_config.cinn_training, args.load)
