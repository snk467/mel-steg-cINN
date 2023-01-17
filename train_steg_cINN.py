#!/usr/bin/env python
import argparse
import copy
import math
import os
from math import ceil
import shutil

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
from helpers.visualization import predict_cinn_example, predict_cinn_example_self_sampled_test
import LUT
import scipy.cluster.vq as scipy_vq

from torchmetrics import Metric
    
def accuracy_loss(output, target):
    assert target.shape == output.shape
    return 1.0 - torch.sum(torch.sign(target) == torch.sign(output)) / target.numel()

def accuracy_metrics(output, target):
    return -(accuracy_loss(output, target) - 1.0)

# Get logger
global logger
logger = helpers.logger.get_logger(__name__)

CINN_MODEL_FILE_NAME = "cinn_model.pt"
STEG_CINN_MODEL_FILE_NAME = "steg_cinn_model.pt"

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
        "Accuracy": accuracy_metrics
                
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
    
    # mse_z_importance = 0.6
    mse_ab_importance = 0.2
    l_importance =  0.2 # max(1.0 - mse_z_importance - mse_ab_importance, 0.0)    
    
    acc = accuracy_loss(z, z_pred)
    
    # return (l_importance * l) + acc + mse_z + (mse_ab_importance * mse_ab), acc.item(), mse_z.item(), mse_ab.item(), l.item()
    
    return (l_importance * l) + mse_z, acc.item(), mse_z.item(), mse_ab.item(), l.item()

def sample_outputs(sigma, out_shape, batch_size, device=utilities.get_device(verbose=False)):
    return [sigma * torch.FloatTensor(torch.Size((batch_size, o))).normal_().to(device) for o in out_shape]

def generate_z_batch(bin: list, cinn_z_dimensions, batch_size, alpha):
    
    desired_size =  sum([x for x in cinn_z_dimensions])
    
    assert desired_size == len(bin)
        
    zs = []
    for i in range(batch_size):
        zs.append(generate_z(bin, cinn_z_dimensions, alpha, desired_size))
        
    for j in range(len(zs)):
        if j != 0:
            for k in range(len(cinn_z_dimensions)):
                zs[0][k] = torch.cat([zs[0][k], zs[j][k]])
        
    return zs[0]

def generate_z(bin, cinn_z_dimensions, alpha, desired_size):
    z = []
    assert desired_size % 2 == 0
    
    half_size = desired_size // 2
    
    for bit in bin:
        sample = np.random.normal()
        if bit == 0:
            while not (sample < -np.abs(alpha)):
                sample = np.random.normal() 
        else:
            while not (sample > np.abs(alpha)):
                sample = np.random.normal()
                
        z.append(sample) 
        
        if len(z) == half_size:
            break
        
    if len(z) != half_size:
        z.extend(np.random.normal(size=desired_size-len(z)))
        
    z.extend(z)
        
    logger.debug(f"Z length: {len(z)}")
    logger.debug(f"desired_size: {desired_size}")
    
    # plt.hist(z, bins=100)
    # plt.show()
    
    z = torch.from_numpy(np.array(z)).float()
    z = list(z.split(cinn_z_dimensions))
    
    logger.debug(f"Z length after split: {len(z)}")
    for i in range(len(z)):
        logger.debug(f"z[{i}].shape: {z[i].shape}")
        z[i] = z[i][None, :]
        logger.debug(f"z[{i}].shape(corrected): {z[i].shape}")
    return z


def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)

def validate(validation_loader, config, revealing_cinn_model_utilities, hiding_cinn_model_utilities, hiding_cinn_output_dimensions, alpha = None):

    revealing_model = revealing_cinn_model_utilities.model
    revealing_model.eval()
    
    hiding_model = hiding_cinn_model_utilities.model
    hiding_model.eval()

    avg_metrics = None

    count = 0
    for i, vdata in enumerate(validation_loader):        
        if i == ceil(0.1 * config.n_its_per_epoch) or vdata[0].shape[0] != config.batch_size:
            break
        
        count += 1
        
        z, _, _, input_melspectrogram = process_batch(config,
                                                        hiding_cinn_model_utilities,
                                                        hiding_cinn_output_dimensions,
                                                        hiding_model,
                                                        vdata)
        
        z_pred, _, _ = revealing_model(input_melspectrogram)
        
        z_pred = torch.cat(z_pred, dim=1)
        z = torch.cat(z, dim=1)

        _, batch_metrics = metrics.gather_batch_metrics(z_pred, z.to(device))

        avg_metrics = metrics.add_metrics(avg_metrics, batch_metrics)   

    avg_metrics = metrics.divide_metrics(avg_metrics, count)

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
    batch_checkpoint = ceil(min(len(training_loader) / 10, config.n_its_per_epoch / 10, 3))

    for i_batch , x in enumerate(training_loader):
        
        z, x_ab_with_message, x_ab_target, input_melspectrogram = process_batch(config,
                                                                                hiding_cinn_model_utilities,
                                                                                hiding_cinn_output_dimensions,
                                                                                hiding_model,
                                                                                x)
        
        z_pred, zz, jac = revealing_model(input_melspectrogram)
        
        revealing_model.eval()      
          
        x_ab_pred = revealing_model.reverse_sample(z_pred, utilities.get_cond(input_melspectrogram[:, 0:1, :, :], revealing_cinn_model_utilities))
        
        revealing_model.train()

        train_loss, acc, mse_z, mse_ab, nll = loss(torch.cat(z_pred, dim=1), torch.cat(z, dim=1).to(device), x_ab_pred[0], x_ab_target.to(device), zz, jac)
        train_loss.backward()
        revealing_cinn_model_utilities.optimizer_step()
        
        # Report
        if i_batch % batch_checkpoint == (batch_checkpoint - 1):
            step +=1
            metrics.log_metrics({'batch_loss': train_loss.item(), 'acc': -(acc - 1.0), 'mse_z': mse_z, 'mse_ab': mse_ab, 'nll': nll}, "train", step, i_batch)

        avg_loss.append(train_loss.item())

        if i_batch+1 >= config.n_its_per_epoch:
            break

    return np.mean(avg_loss), step

def process_batch(config, hiding_cinn_model_utilities, hiding_cinn_output_dimensions, hiding_model, x):
    x_l, x_ab_target, _, _ = x
    x_l = x_l.to('cpu')    
    
    # m = np.random.randint(2, size=tot_output_size)
            
    # z = generate_z_batch(m,
    #                     hiding_cinn_output_dimensions,
    #                     config.batch_size,
    #                     alpha=config.alpha)

    z = utilities.sample_z(hiding_cinn_output_dimensions, config.batch_size, config.alpha, device='cpu')
        
    cond = utilities.get_cond(x_l, hiding_cinn_model_utilities) 
        
    hiding_model.eval()
        
    x_ab_with_message = hiding_model.reverse_sample(z, cond)
        
    # input_melspectrogram = compress_melspectrograms(config, x_l, x_ab_with_message[0])
    input_melspectrogram = compress_melspectrograms2(torch.cat([x_l, x_ab_with_message[0].detach()], dim=1))
    
    # print("HEJ!", input_melspectrogram.shape)
    
    # input_melspectrogram = torch.cat((x_l, x_ab_with_message[0]), dim=1).to(device)
    
    return z,x_ab_with_message,x_ab_target,input_melspectrogram.float()

def compress_melspectrograms(config, x_l, x_ab_with_message):    
    colormap = LUT.ColormapTorch.from_colormap("parula_norm_lab").to(device)
    
    indexes = colormap.get_indexes_from_colors(torch.cat([x_l, x_ab_with_message], dim=1).to(device))
    
    return colormap.get_colors_from_indexes(indexes)

def compress_melspectrograms2(mel_spectrograms: torch.Tensor):
    
    result_mel_spectrograms = []
    # result_colormaps = []
    
    for i in range(mel_spectrograms.shape[0]):
        mel_spectrogram = mel_spectrograms[i]
        shape = mel_spectrogram.shape
        
        # (3, 512, 512) -> (512^2, 3)
        
        # assert (mel_spectrogram.numpy().reshape((shape[1]*shape[2], shape[0])).reshape(shape) == mel_spectrogram.numpy()).all()
        
        mel_spectrogram = mel_spectrogram.numpy().transpose(1,2,0).reshape((shape[1]*shape[2], shape[0]))
        
        # print(mel_spectrogram.shape)
        # print(np.array(LUT.Colormap.colormaps["parula_norm_lab"]).shape)
        
        centroids, labels = scipy_vq.kmeans2(mel_spectrogram, np.array(LUT.Colormap.colormaps["parula_norm_lab"]))
        
        colormap = LUT.Colormap.from_colors_list(centroids)
        indexes = labels.reshape((shape[1], shape[2]))
        
        result_mel_spectrograms.append(torch.Tensor(colormap.get_colors_from_indexes(indexes))[None, :].permute((0, 3, 1, 2)))   
        # result_colormaps.append(colormap)
        
        # print(result_mel_spectrograms[-1])
        # print(mel_spectrograms[-1])
        
        # index = (np.random.randint(512), np.random.randint(512))
        # print(result_mel_spectrograms[-1][:, :, index[0], index[1]])
        # print(mel_spectrograms[i][:, index[0], index[1]])
        
        # print(mse_loss(result_mel_spectrograms[-1], mel_spectrograms[i][None, :]).item())
        
    return torch.cat(result_mel_spectrograms, dim=0).to(device)
        
        
        
        
        
        
        


def train(config=None, load=None, revealing_load=None):  
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
        
        logger.info(f"Train dataset real size: {config.n_its_per_epoch}")   
        logger.info(f"Validation dataset real size: {ceil(0.1 * config.n_its_per_epoch)}")   

        # Create data loaders for our datasets; shuffle for training, not for validation
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=config.batch_size, shuffle=True, num_workers=2)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config.batch_size, shuffle=False, num_workers=2)

        early_stopper = model.EarlyStopper(patience=config.early_stopper_patience, min_delta=config.early_stopper_min_delta)


        logger.info("Downloading revealing model...")
        try:
            revealing_cinn_model_utilities, revealing_cinn_output_dimensions = utilities.get_cinn_model(config, STEG_CINN_MODEL_FILE_NAME, revealing_load, device=device)
        except ValueError:
            revealing_cinn_model_utilities, revealing_cinn_output_dimensions = utilities.get_cinn_model(config, CINN_MODEL_FILE_NAME, revealing_load, device=device)

        logger.info("Downloading hiding model...")
        try:
            hiding_cinn_model_utilities, hiding_cinn_output_dimensions = utilities.get_cinn_model(config, STEG_CINN_MODEL_FILE_NAME, load)
        except ValueError:
            hiding_cinn_model_utilities, hiding_cinn_output_dimensions = utilities.get_cinn_model(config, CINN_MODEL_FILE_NAME, load)
        
        logger.info(f"Training feature net: {main_config.cinn_management.end_to_end}")
        logger.info(f"Revealing model device: {next(revealing_cinn_model_utilities.model.parameters()).device}")
        logger.info(f"Hiding model device: {next(hiding_cinn_model_utilities.model.parameters()).device}")
        
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
            avg_metrics = validate(validation_loader,
                                   config,
                                   revealing_cinn_model_utilities,
                                   hiding_cinn_model_utilities,
                                   hiding_cinn_output_dimensions,)
            # Report
            metrics.log_metrics(avg_metrics, "VALID AVG", step)

            if i_epoch >= config.pretrain_epochs * 2:
                revealing_cinn_model_utilities.scheduler_step(avg_loss)
                
            if early_stopper.early_stop(avg_metrics["MSE"]):
                break
            
            if config.rewrite_models:                
                hiding_cinn_model_utilities.model.load_state_dict(revealing_cinn_model_utilities.model.state_dict())

        revealing_cinn_model_utilities.model.eval()
        logger.info("Generating examples.")
        training_examples = predict_cinn_example(revealing_cinn_model_utilities.model, revealing_cinn_output_dimensions, training_set, config, desc="Training set example", restore_audio=False)
        wandb.log({"training_examples": [wandb.Image(image) for image in training_examples]})
        print()
        validation_examples = predict_cinn_example(revealing_cinn_model_utilities.model, revealing_cinn_output_dimensions, validation_set, config, desc="Validation set example", restore_audio=False)
        wandb.log({"validation_examples": [wandb.Image(image) for image in validation_examples]})
        print()
        overfitting_examples = predict_cinn_example_self_sampled_test(revealing_cinn_model_utilities.model, revealing_cinn_output_dimensions, training_set, config, desc="Training set self_sampled example", restore_audio=False)
        wandb.log({"self-sampled_examples": [wandb.Image(image) for image in overfitting_examples]})
        
        # Save model
        model_path = None
        if main_config.common.save_model:   
            logger.info("Saving steg-cINN model.")
            model_path = os.path.join(os.getcwd(),"tmp", STEG_CINN_MODEL_FILE_NAME)
            revealing_cinn_model_utilities.save(model_path)
            wandb.save(model_path, base_path=os.getcwd(), policy="now")

        wandb.finish()
        
        if model_path is not None:
            shutil.rmtree(os.path.join(os.getcwd(), "tmp"))

    # os.makedirs(os.path.dirname(config.filename), exist_ok=True)
    # model.save(config.filename)


def run(config, load=None, revealing_load=None):   
    # Initialize Weights & Biases
    wandb.login(key=main_config.common.wandb_key)
    train(config, load, revealing_load)      
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train cINN for mel-spectrogram colorization.')
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--dataset_size', type=int, required=False)
    parser.add_argument('--betas', type=tuple, required=False)
    parser.add_argument('--clamping', type=float, required=False)
    parser.add_argument('--alpha', type=float, required=False)
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
    parser.add_argument('--revealing_load', type=str, required=False)
    parser.add_argument('--early_stopper_min_delta', type=float, required=False)
    args = parser.parse_args()
    
    sweep_args_dict = {k: vars(args)[k] for k in vars(args).keys() - {'load'}}
    sweep_args_present = all(value is not None for value in sweep_args_dict.values()) 
    
    if sweep_args_present:
        run(args, args.load, args.revealing_load)
    else:
        run(main_config.steg_cinn_training, args.load, args.revealing_load)
    