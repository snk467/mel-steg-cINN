from math import ceil
import time
import torch
import random
import wandb
import os
from config import config
import argparse
import torchmetrics as torch_metrics
from datasets import SpectrogramsDataset 
from unet.unet_models import *
from helpers.metrics import Metrics
import helpers.utilities as utilities
import helpers.logger
from helpers.noise import *
from helpers.visualization import show_data

MODEL_FILE_NAME = "unet_model.pt"

def train_one_epoch(model, training_loader, optimizer, epoch, step):
    running_metrics = None
    avg_metrics = None
    
    batch_checkpoint = ceil(len(training_loader) / 10)

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pairs\
        inputs, targets, _, _ = data
        
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        # print(inputs.shape)
        outputs = model(inputs)

        # Compute the loss and its gradients
        batch_loss, batch_metrics = metrics.gather_batch_metrics(outputs, targets)
        batch_loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data
        running_metrics = metrics.add_metrics(running_metrics, batch_metrics)
        avg_metrics = metrics.add_metrics(avg_metrics, batch_metrics)
        
        # Report
        if i % batch_checkpoint == (batch_checkpoint - 1):
            step +=1

            # Calculate current checkpoint metrics
            current_metrics = metrics.divide_metrics(running_metrics, batch_checkpoint)

            # Log metrics           
            metrics.log_metrics(batch_metrics, "train", step, batch_id=i)
            
            wandb.log({"epoch": epoch + ((i+1)/len(training_loader))}, step=step)

            # Reset batch metrics
            running_metrics = None

        if (i + 1) == len(training_loader):
            avg_metrics = metrics.divide_metrics(avg_metrics, len(training_loader))
    
    torch.cuda.empty_cache()
    
    
    return avg_metrics, step

def validate(model, validation_loader):
    avg_metrics = None

    for i, vdata in enumerate(validation_loader):
        vinputs, vtargets, _, _ = vdata
        
        vinputs = vinputs.to(device).float()
        vtargets = vtargets.to(device).float()
        
        voutputs = model(vinputs)

        _, batch_metrics = metrics.gather_batch_metrics(voutputs, vtargets)

        avg_metrics = metrics.add_metrics(avg_metrics, batch_metrics)
    
    avg_metrics = metrics.divide_metrics(avg_metrics, len(validation_loader) )

    return avg_metrics

def train(config = None):    
    with wandb.init(project="UNET", entity="snikiel", config=config):
        config = wandb.config    
    
        # Create datasets for training & validation
        logger.info("Import training set.")
        
        logger.info(config.input_dims)
        
        training_set = SpectrogramsDataset(common_config.dataset_location,
                                           train=True,
                                           size=config.dataset_size,
                                           augmentor=GaussianNoise(common_config.noise_mean, common_config.noise_variance),
                                           output_dim=config.input_dims)
        logger.info("Import validation set.")
        validation_set = SpectrogramsDataset(common_config.dataset_location,
                                             train=False,
                                             size=config.dataset_size,
                                             augmentor=GaussianNoise(common_config.noise_mean, common_config.noise_variance),
                                             output_dim=config.input_dims)

        # Create data loaders for our datasets; shuffle for training, not for validation
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=config.batch_size, shuffle=True, num_workers=2)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config.batch_size, shuffle=False, num_workers=2)

        # Show first element
        if common_config.present_data:
            print("Input example:")
            example_id = random.randint(0, len(training_set) - 1)
            show_data(*training_set[example_id])
        
        # Create model
        model = models[config.model](n_channels=1)
        model = model.to(device).float()

        # Optimizers specified in the torch.optim package
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        
        # LR scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=False, patience=2, min_lr=0.0, factor=0.1)
        
        # Get loss function
        metrics.metrics_functions["Loss"] = loss_functions[config.loss_function].to(device) 

        # Print initial parameters
        print_initial_parameters(config)
        
        step = 0
        epoch_durations = []
        for epoch in range(config.epochs):
            logger.info('EPOCH {}:'.format(epoch + 1))
            epoch_start_time = time.time()

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            logger.info("       Model training.")
            train_metrics, step = train_one_epoch(model, training_loader, optimizer, epoch, step)

            # We don't need gradients on to do reporting
            model.train(False)

            # Validate model
            logger.info("       Model validation.")
            validation_metrics = validate(model, validation_loader)
                
            wandb.log({"learning_rate": scheduler.optimizer.param_groups[0]['lr']}, step=step)

            # Log metrics
            metrics.log_metrics(train_metrics, "TRAIN AVG", step)
            metrics.log_metrics(validation_metrics, "VALID AVG", step)

            # Log epoch duration
            epoch_duration = time.time() - epoch_start_time
            wandb.log({"epoch_runtime (seconds)": epoch_duration}, step=step)
            epoch_durations.append(epoch_duration)

        # Predict first element from train
        if common_config.present_data:
            predict_example(model, training_set, desc="Training set example")
            print()
            predict_example(model, validation_set, desc="Validation set example")
        
        # Save model
        model_path = None
        if common_config.save_model:
            logger.info("Saving UNET model.")
            model_path = os.path.join(os.getcwd(), MODEL_FILE_NAME)
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path, base_path=os.getcwd())
        
        # Log average epoch duration
        avg_epoch_runtime = sum(epoch_durations) / len(epoch_durations)
        wandb.log({"avg epoch runtime (seconds)": avg_epoch_runtime})
        
        # Finish the Weights & Biases run
        wandb.finish()
        
        if model_path is not None:
            os.remove(model_path)
    
def print_initial_parameters(config):
    logger.info("INITIAL PARAMETERS")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Number of epochs {config.epochs}")
    logger.info(f"Learning rate: {config.lr}")
    logger.info(f"Loss function: {config.loss_function}")
    logger.info(f"Model: {config.model}")
    
def predict_example(model, dataset, desc=None):
    example_id = random.randint(0, len(dataset) - 1)
    input, target, filename, clear_input = dataset[example_id]
    batched_input = torch.reshape(input, (1, *input.shape)).to(device).float()        
    output = model(batched_input)
    print(desc)
    print("Result:")
    show_data(batched_input[0], output[0], filename, clear_input)
    print("Target:")
    show_data(input, target, filename, clear_input) 

def prepare_globals():    
    # Get logger
    global logger
    logger = helpers.logger.get_logger(__name__)

    # Initialize Weights & Biases
    wandb.login(key=config.common.wandb_key)
    
    global device
    device = utilities.get_device() 
    
    # Set metrics functions   
    metrics_functions = {
        "Loss": None,
        "MSE": torch_metrics.MeanSquaredError().to(device),
        "HuberLoss": torch.nn.HuberLoss().to(device),
        "MAE": torch_metrics.MeanAbsoluteError().to(device),
        # "RMSE": torch_metrics.MeanSquaredError(squared=False).to(device),
        # "MAPE": torch_metrics.MeanAbsolutePercentageError().to(device)
        
    }
    
    global metrics
    metrics = Metrics(metrics_functions)  
        
    global loss_functions 
    loss_functions = { 
        "MSELoss": torch.nn.MSELoss(),
        "HuberLoss": torch.nn.HuberLoss(),
        "L1Loss": torch.nn.L1Loss()
    }
    
    global common_config
    common_config = config.common
    print(common_config)

    global models
    models = {
        "custom_unet": custom_UNet,
        "unet": UNet,
        "unet_32": UNet_32,
        "unet_128": UNet_128,
        "unet_256": UNet_256
    }
        
def run(sweep=False):
    
    prepare_globals()
    
    if sweep:       
        logger.info(config.unet_sweep_config) 
        sweep_id = wandb.sweep(config.unet_sweep_config, project="UNET", entity="snikiel")
        wandb.agent(sweep_id, function=train, count=config.common.sweep_count)
    else:
        train(config.unet_training)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train UNET for mel-spectrogram colorization.')
    parser.add_argument('--sweep', action='store_true', help='run Weights & Biases sweep')

    args = parser.parse_args()
    
    run(args.sweep)