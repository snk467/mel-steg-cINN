#!/usr/bin/env python
import argparse
import copy
import math
import os
from math import ceil

import numpy as np
import torch
import torch.nn
import torch.optim
import torchmetrics as torch_metrics

import cinn.cinn_model as model
import wandb
from config import config as main_config
from datasets import SpectrogramsDataset
import helpers.logger
import helpers.visualization
import helpers.utilities as utilities
from helpers.metrics import Metrics
from helpers.noise import GaussianNoise
from helpers.visualization import predict_cinn_example, predict_cinn_example_overfitting_test
import LUT

from torchmetrics import Metric

class CustomAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.correct += torch.sum(torch.sign(preds) == torch.sign(target))
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total

# Get logger
global logger
logger = helpers.logger.get_logger(__name__)

MODEL_FILE_NAME = "cinn_model.pt"
STEG_MODEL_FILE_NAME = "steg_cinn_model.pt"

def prepare_training():
    global device
    device = utilities.get_device()  
    
    global mse_loss
    mse_loss = torch.nn.MSELoss().to(device)
    
    # if config.cinn_management.load_file:
    #     model.load(config.cinn_training.load_file)
    
    # Set metrics functions   
    metrics_functions = {
        "MSE": torch_metrics.MeanSquaredError().to(device),
        "HuberLoss": torch.nn.HuberLoss().to(device),
        "MAE": torch_metrics.MeanAbsoluteError().to(device),
        "Accuracy": CustomAccuracy().to(device)
                
    }
    
    global metrics
    metrics = Metrics(metrics_functions)  

    global tot_output_size
    tot_output_size = 2 * main_config.cinn_management.img_dims[0] * main_config.cinn_management.img_dims[1]

def loss(z_pred, z, ab_pred, ab_target, zz, jac):
    
    mse_z = mse_loss(z_pred, z)
    
    mse_ab = mse_loss(ab_pred, ab_target)
    
    neg_log_likeli = 0.5 * zz - jac

    l = torch.mean(neg_log_likeli) / tot_output_size
    
    return 0.1 * torch.exp(l) + mse_z + mse_ab, mse_z.item(), mse_ab.item(), l.item()

def sample_outputs(sigma, out_shape, batch_size, device=utilities.get_device(verbose=False)):
    return [sigma * torch.FloatTensor(torch.Size((batch_size, o))).normal_().to(device) for o in out_shape]

def sample_z(out_shapes, batch_size, alpha=None, device=utilities.get_device(verbose=False)):
    
    samples = []
    
    for out_shape in out_shapes:
        sample = torch.normal(mean=0.0, std=1.0, size=(batch_size, out_shape), device=device)
        
        if alpha is not None: 
            def get_value_out_of_range():
                value = 0.0
                while np.abs(value) < alpha:
                    value = np.random.normal(loc=0.0, scale=1.0)
                return value
            
            sample = torch.where(torch.abs(sample) > torch.tensor(alpha), sample, torch.tensor(get_value_out_of_range()))
            
        samples.append(sample)
        
    return samples

def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)

def validate(revealing_cinn_model_utilities, hiding_cinn_model_utilities, hiding_cinn_output_dimensions, config, validation_loader, alpha = None):

    revealing_model = revealing_cinn_model_utilities.model
    revealing_model.eval()
    
    hiding_model = hiding_cinn_model_utilities.model
    hiding_model.eval()

    avg_metrics = None

    for i, vdata in enumerate(validation_loader):        
        if (i + 1) % 100 == 0:
            break
        
        z, _, _, _, z_pred, _, _ = process_batch(config, hiding_cinn_model_utilities, hiding_cinn_output_dimensions, revealing_model, hiding_model, vdata)
        
        z_pred = torch.cat(z_pred, dim=1)
        z = torch.cat(z, dim=1)

        _, batch_metrics = metrics.gather_batch_metrics(z_pred, z.to(device))

        avg_metrics = metrics.add_metrics(avg_metrics, batch_metrics)                
    
    avg_metrics = metrics.divide_metrics(avg_metrics, len(validation_loader) )

    return avg_metrics

def train_one_epoch(training_loader,
                    config,
                    i_epoch,
                    step,
                    revealing_cinn_model_utilities: model.cINNTrainingUtilities,
                    revealing_cinn_output_dimensions,
                    hiding_cinn_model_utilities,
                    hiding_cinn_output_dimensions):

    revealing_model = revealing_cinn_model_utilities.model
    hiding_model = hiding_cinn_model_utilities.model
    hiding_model.eval()
    revealing_model.train()

    # Ustawainie lr odpowiednio do epoki
    if i_epoch < 0:
        for param_group in revealing_cinn_model_utilities.optimizer.param_groups:
            param_group['lr'] = config.lr * 2e-2

    if i_epoch == 0:
        for param_group in revealing_cinn_model_utilities.optimizer.param_groups:
            param_group['lr'] = config.lr

    if main_config.cinn_management.end_to_end and i_epoch <= config.pretrain_epochs:
        for param_group in revealing_cinn_model_utilities.feature_optimizer.param_groups:
            param_group['lr'] = 0
        if i_epoch == config.pretrain_epochs:
            for param_group in revealing_cinn_model_utilities.feature_optimizer.param_groups:
                param_group['lr'] = 1e-4

    avg_loss = []
    batch_checkpoint = ceil(min(len(training_loader) / 10, config.n_its_per_epoch / 10))

    for i_batch , x in enumerate(training_loader):
        
        z, x_ab_with_message, x_ab_target, input_melspectrogram, z_pred, zz, jac = process_batch(config, hiding_cinn_model_utilities, hiding_cinn_output_dimensions, revealing_model, hiding_model, x)
        
        revealing_model.eval()      
          
        x_ab_pred = revealing_model.reverse_sample(z_pred, utilities.get_cond(input_melspectrogram[:, 0:1, :, :], revealing_cinn_model_utilities))
        
        revealing_model.train()

        train_loss, mse_z, mse_ab, nll = loss(torch.cat(z_pred, dim=1), torch.cat(z, dim=1).to(device), x_ab_pred[0], x_ab_target.to(device), zz, jac)
        train_loss.backward()
        revealing_cinn_model_utilities.optimizer_step()
        
        # Report
        if i_batch % batch_checkpoint == (batch_checkpoint - 1):
            step +=1
            metrics.log_metrics({'batch_loss': train_loss.item(), 'mse_z': mse_z, 'mse_ab': mse_ab, 'nll': math.exp(nll)}, "train", step, i_batch)

        avg_loss.append(train_loss.item())

        if i_batch+1 >= config.n_its_per_epoch:
            break

    return np.mean(avg_loss), step

def process_batch(config, hiding_cinn_model_utilities, hiding_cinn_output_dimensions, revealing_model, hiding_model, x):
    x_l, x_ab_target, _, _ = x
    x_l = x_l.to('cpu')    
            
    z = sample_z(hiding_cinn_output_dimensions, config.batch_size, device='cpu')
        
    cond = utilities.get_cond(x_l, hiding_cinn_model_utilities) 
        
    x_ab_with_message = hiding_model.reverse_sample(z, cond)
        
    input_melspectrogram = compress_melspectrograms(config, x_l, x_ab_with_message[0]).float()
        
    z_pred, zz, jac = revealing_model(input_melspectrogram)
    return z,x_ab_with_message,x_ab_target,input_melspectrogram,z_pred,zz,jac

def compress_melspectrograms(config, x_l, x_ab_with_message):    
    colormap = LUT.ColormapTorch.from_colormap("parula_norm_lab").to(device)
    
    indexes = colormap.get_indexes_from_colors(torch.cat([x_l, x_ab_with_message], dim=1).to(device))
    return colormap.get_colors_from_indexes(indexes)


def train(config=None, load=None):  
    with wandb.init(project="steg-cINN", entity="snikiel", config=config): 
        wandb.log({"main_setup": main_config.to_dict()})
        prepare_training()
        config = wandb.config        
        logger.info(config)
        training_set = SpectrogramsDataset(main_config.common.dataset_location,
                                        train=True,
                                        size=config.dataset_size,
                                        augmentor=GaussianNoise(main_config.common.noise_mean, main_config.common.noise_variance),
                                        output_dim=main_config.cinn_management.img_dims)

        validation_set = SpectrogramsDataset(main_config.common.dataset_location,
                                        train=False,
                                        size=config.dataset_size,
                                        augmentor=GaussianNoise(main_config.common.noise_mean, main_config.common.noise_variance),
                                        output_dim=main_config.cinn_management.img_dims)

        # Create data loaders for our datasets; shuffle for training, not for validation
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=config.batch_size, shuffle=True, num_workers=2)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config.batch_size, shuffle=False, num_workers=2)

        early_stopper = model.EarlyStopper(patience=config.early_stopper_patience, min_delta=config.early_stopper_min_delta)

        revealing_cinn_model_utilities, revealing_cinn_output_dimensions = utilities.get_cinn_model(config, MODEL_FILE_NAME, load, device=device)
        
        revealing_cinn_model = revealing_cinn_model_utilities.model
        
        hiding_cinn_model_utilities, hiding_cinn_output_dimensions = utilities.get_cinn_model(config, MODEL_FILE_NAME, load)

        logger.info(f"Training feature net: {main_config.cinn_management.end_to_end}")
            
        logger.debug(f"cinn model device: {next(revealing_cinn_model.parameters()).device}")
        
        step = 0

        for i_epoch in range(-config.pre_low_lr, config.n_epochs):

            logger.info('EPOCH {}:'.format(i_epoch + 1))
            logger.info("       Model training.")
            avg_loss, step = train_one_epoch(training_loader,
                                             config,
                                             i_epoch,
                                             step,
                                             revealing_cinn_model_utilities,
                                             revealing_cinn_output_dimensions,
                                             hiding_cinn_model_utilities,
                                             hiding_cinn_output_dimensions)
            
            metrics.log_metrics({'loss': avg_loss}, "TRAIN AVG", step)

            logger.info("       Model validation.")
            avg_metrics = validate(revealing_cinn_model_utilities, hiding_cinn_model_utilities, hiding_cinn_output_dimensions, config, validation_loader)
            # Report
            metrics.log_metrics(avg_metrics, "VALID AVG", step)

            if i_epoch >= config.pretrain_epochs * 2:
                revealing_cinn_model_utilities.scheduler_step(avg_loss)
                
            if early_stopper.early_stop(avg_metrics["MSE"]):
                break
            
            # hiding_cinn_model_utilities.model.load_state_dict(revealing_cinn_model_utilities.model.state_dict())

            # if i_epoch > 0 and (i_epoch % config.checkpoint_save_interval) == 0:
            #     model.save(config.filename + '_checkpoint_%.4i' % (i_epoch * (1-config.checkpoint_save_overwrite)))

        revealing_cinn_model.eval()
        logger.info("Generating examples.")
        training_examples = predict_cinn_example(revealing_cinn_model, revealing_cinn_output_dimensions, training_set, config, desc="Training set example", restore_audio=False)
        wandb.log({"training_examples": [wandb.Image(image) for image in training_examples]})
        print()
        validation_examples = predict_cinn_example(revealing_cinn_model, revealing_cinn_output_dimensions, validation_set, config, desc="Validation set example", restore_audio=False)
        wandb.log({"validation_examples": [wandb.Image(image) for image in validation_examples]})
        print()
        overfitting_examples = predict_cinn_example_overfitting_test(revealing_cinn_model, revealing_cinn_output_dimensions, training_set, config, desc="Training set overfitting example", restore_audio=False)
        wandb.log({"overfitting_examples": [wandb.Image(image) for image in overfitting_examples]})
        
        # Save model
        model_path = None
        if main_config.common.save_model:   
            logger.info("Saving steg-cINN model.")
            model_path = os.path.join(os.getcwd(), STEG_MODEL_FILE_NAME)
            revealing_cinn_model_utilities.save(model_path)
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
    parser.add_argument('--pre_low_lr', type=float, required=False)
    parser.add_argument('--pretrain_epochs', type=int, required=False)
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
    