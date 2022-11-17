#!/usr/bin/env python
import argparse
import gc
import os
import random
import sys
from math import ceil

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim
import torchmetrics as torch_metrics
import tqdm
from torch.autograd import Variable
from torch.nn.functional import avg_pool2d  # , interpolate

import colorization_cINN.model as model
import logger as logger_module
import utilities
import visualization
import wandb
from config import config as main_config
from datasets import SpectrogramsDataset
from metrics import Metrics
from noise import GaussianNoise
from visualization import predict_cinn_example

# Get logger
global logger
logger = logger_module.get_logger(__name__)

MODEL_FILE_NAME = "cinn_model.pt"

def prepare_training():
    global device
    device = utilities.get_device()  
    
    # if config.cinn_management.load_file:
    #     model.load(config.cinn_training.load_file)
    
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

def loss(zz, jac):
    neg_log_likeli = 0.5 * zz - jac

    l = torch.mean(neg_log_likeli) / tot_output_size
    return l

def sample_outputs(sigma, out_shape, batch_size):
    return [sigma * torch.FloatTensor(torch.Size((batch_size, o))).normal_().to(device) for o in out_shape]

def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)

def validate(cinn_model, cinn_output_dimensions, config, validation_loader):

    cinn_model.eval()

    avg_metrics = None

    for i, vdata in enumerate(validation_loader):
        x_l, x_ab_target, cond, ab_pred = cinn_model.prepare_batch(vdata)           

        z = sample_outputs(config.sampling_temperature, cinn_output_dimensions, vdata[0].shape[0])

        x_ab_output = cinn_model.reverse_sample(z, cond)

        _, batch_metrics = metrics.gather_batch_metrics(x_ab_output[0], x_ab_target)

        avg_metrics = metrics.add_metrics(avg_metrics, batch_metrics)                
    
    avg_metrics = metrics.divide_metrics(avg_metrics, len(validation_loader) )

    return avg_metrics

def train_one_epoch(cinn_model, training_loader, config, i_epoch, step, cinn_training_utilities: model.cINNTrainingUtilities):

    cinn_model.train()

    # Ustawainie lr odpowiednio do epoki
    if i_epoch < 0:
        for param_group in cinn_training_utilities.optimizer.param_groups:
            param_group['lr'] = config.lr * 2e-2

    if i_epoch == 0:
        for param_group in cinn_training_utilities.optimizer.param_groups:
            param_group['lr'] = config.lr

    if main_config.cinn_management.end_to_end and i_epoch <= config.pretrain_epochs:
        for param_group in cinn_training_utilities.feature_optimizer.param_groups:
            param_group['lr'] = 0
        if i_epoch == config.pretrain_epochs:
            for param_group in cinn_training_utilities.feature_optimizer.param_groups:
                param_group['lr'] = 1e-4

    avg_loss = []
    batch_checkpoint = ceil(min(len(training_loader) / 10, config.n_its_per_epoch / 10))

    for i_batch , x in enumerate(training_loader):

        L, ab, _, _ = x  
        input = torch.cat((L, ab), dim=1).to(device)

        zz, jac = cinn_model(input)

        train_loss = loss(zz, jac)
        train_loss.backward()
        cinn_training_utilities.optimizer_step()
        
        # Report
        if i_batch % batch_checkpoint == (batch_checkpoint - 1):
            step +=1
            metrics.log_metrics({'batch_loss': train_loss.item()}, "train", step, i_batch)

        avg_loss.append(train_loss.item())

        if i_batch+1 >= config.n_its_per_epoch:
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
                                        augmentor=GaussianNoise([0.0], [0.001, 0.001, 0.0]))

        validation_set = SpectrogramsDataset(main_config.common.dataset_location,
                                        train=False,
                                        size=config.dataset_size,
                                        augmentor=GaussianNoise([0.0], [0.001, 0.001, 0.0]))

        # Create data loaders for our datasets; shuffle for training, not for validation
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=config.batch_size, shuffle=True, num_workers=2)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config.batch_size, shuffle=False, num_workers=2)


        cinn_builder = model.cINN_builder(config)
    
        feature_net = cinn_builder.get_feature_net()
        fc_cond_net = cinn_builder.get_fc_cond_net()
        cinn, cinn_output_dimensions = cinn_builder.get_cinn()
                
        cinn_model = model.WrappedModel(feature_net, fc_cond_net, cinn)
        cinn_training_utilities = model.cINNTrainingUtilities(cinn_model, config)
        
        if load is not None:
            restored_model = wandb.restore(MODEL_FILE_NAME, run_path=load)# "lavanyashukla/save_and_restore/10pr4joa"
            logger.info(f"Loading model: {load} from: {restored_model.name}")
            cinn_training_utilities.load(restored_model.name)
            os.remove(restored_model.name)
            cinn_model = cinn_training_utilities.model
            
        logger.debug(f"cinn model device: {next(cinn_model.parameters()).device}")
        
        step = 0

        for i_epoch in range(-config.pre_low_lr, config.n_epochs):
            logger.info('EPOCH {}:'.format(i_epoch + 1))
            logger.info("       Model training.")
            avg_loss, step = train_one_epoch(cinn_model, training_loader, config, i_epoch, step, cinn_training_utilities)
            metrics.log_metrics({'loss': avg_loss}, "TRAIN AVG", step)

            logger.info("       Model validation.")
            avg_metrics = validate(cinn_model, cinn_output_dimensions, config, validation_loader)
            # Report
            metrics.log_metrics(avg_metrics, "VALID AVG", step)

            if i_epoch >= config.pretrain_epochs * 2:
                cinn_training_utilities.scheduler_step(avg_loss)

            # if i_epoch > 0 and (i_epoch % config.checkpoint_save_interval) == 0:
            #     model.save(config.filename + '_checkpoint_%.4i' % (i_epoch * (1-config.checkpoint_save_overwrite)))

        logger.info("Generating examples.")
        training_examples = predict_cinn_example(cinn_model, cinn_output_dimensions, training_set, config, desc="Training set example", restore_audio=True)
        wandb.log({"training_examples": [wandb.Image(image) for image in training_examples]})
        print()
        validation_examples = predict_cinn_example(cinn_model, cinn_output_dimensions, validation_set, config, desc="Validation set example", restore_audio=True)
        wandb.log({"validation_examples": [wandb.Image(image) for image in validation_examples]})
            
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
    parser.add_argument('--n_epochs', type=int, required=False)
    parser.add_argument('--n_its_per_epoch', type=int, required=False)
    parser.add_argument('--pre_low_lr', type=float, required=False)
    parser.add_argument('--pretrain_epochs', type=int, required=False)
    parser.add_argument('--sampling_temperature', type=float, required=False)
    parser.add_argument('--weight_decay', type=float, required=False)
    parser.add_argument('--load', type=str, required=False)
    args = parser.parse_args()
    
    sweep_args_dict = {k: args[k] for k in vars(args).keys() - {'load'}}
    sweep_args_present = all(value is not None for value in sweep_args_dict.values()) 
    
    if sweep_args_present:
        run(args, args.load)
    else:
        run(main_config.cinn_training, args.load)
    