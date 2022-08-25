import time
import torch
import torchvision
from torchmetrics import Accuracy
import wandb
from Datasets.SpectrogramsDataset import SpectrogramsDataset 
import torch.nn.functional as torch_func
import Configuration
from Models.UNET.unet_model import UNet
import Logger
import munch
import argparse
from PIL import Image        
import torchvision.transforms as torch_trans
import LUT
import numpy as np

def train_one_epoch(model, training_loader, optimizer, config, epoch, step):
    running_metrics = prepare_metrics()
    avg_metrics = prepare_metrics()

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pairs\
        inputs, targets, _ = data
        
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()
        

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        # print(inputs.shape)
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss, metrics = gather_batch_metrics(outputs, targets)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data
        running_metrics.loss += loss.item()
        # running_metrics.accuracy += metrics.accuracy

        avg_metrics.loss += loss.item()
        # avg_metrics.accuracy += metrics.accuracy

        # Report
        if i % global_config.batch_checkpoint == (global_config.batch_checkpoint - 1):
            step +=1

            # Calculate current checkpoint metrics
            current_loss = running_metrics.loss / global_config.batch_checkpoint
            # current_accuracy = running_metrics.accuracy / global_config.batch_checkpoint

            # Log to stdout
            logger.info(f"      batch {i + 1} loss: {current_loss} accuracy: {None} %")

            # Log to Weights & Biases
            wandb.log({"train_loss": current_loss, "epoch": epoch + ((i+1)/len(training_loader))}, step=step)

            # Reset batch metrics
            running_metrics = prepare_metrics()

        if (i + 1) == len(training_loader):
            avg_metrics.loss = avg_metrics.loss / len(training_loader)
            # avg_metrics.accuracy = avg_metrics.accuracy / len(training_loader)
            
    return avg_metrics, step

def validate(model, validation_loader):
    avg_metrics = prepare_metrics()

    for i, vdata in enumerate(validation_loader):
        vinputs, vtargets, _ = vdata
        
        vinputs = vinputs.to(device).float()
        vtargets = vtargets.to(device).float()
        
        voutputs = model(vinputs)

        vloss, metrics = gather_batch_metrics(voutputs, vtargets)

        avg_metrics.loss += vloss.item()
        # avg_metrics.accuracy += metrics.accuracy


    avg_metrics.loss = avg_metrics.loss / len(validation_loader) 
    # avg_metrics.accuracy = avg_metrics.accuracy / len(validation_loader) 

    return avg_metrics

def get_rgb_image_from_labels(labels):
    colormap = LUT.Colormap.from_colormap("parula_rgb")  
    indexes = torch.argmax(labels, dim=1)
    rgb_target = colormap.get_colors_from_indexes(indexes[0].numpy())
    img_target = toImage((rgb_target * 255).astype(np.uint8))
    return img_target

def train(config=None):    
    with wandb.init(project="mel-steg-cINN", entity="snikiel", config=config):
        config = wandb.config    
    
        # Create datasets for training & validation
        logger.info("Import training set.")
        training_set = SpectrogramsDataset(global_config.spectrogram_files_directory, train=True, size=global_config.dataset_size)
        logger.info("Import validation set.")
        validation_set = SpectrogramsDataset(global_config.spectrogram_files_directory, train=False, size=global_config.dataset_size)

        # Create data loaders for our datasets; shuffle for training, not for validation
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=config.batch_size, shuffle=True, num_workers=2)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config.batch_size, shuffle=False, num_workers=2)

        
        first_batch = next(iter(training_loader))
        print(type(first_batch))
        print(len(first_batch))
        inputs = first_batch[0]
        labels = first_batch[1]
        filename = first_batch[2]
        print(inputs.shape)
        print(labels.shape)        

        
        img = toImage(inputs[0]) 
        print(f"Input ( {filename[0]} ):")
        img.show()        
        
        img_target = get_rgb_image_from_labels(labels)  
        print(f"Target ( {filename[0]} ):")
        img_target.show()
        
        
        #TODO: zamknąć prezentację danych w jednej funkcji
        
        
        
        # Create model
        model = UNet(n_channels=1)
        model = model.to(device).float()

        # Optimizers specified in the torch.optim package
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        
        # LR scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=3, min_lr=0, factor=0.1)
        
        # Define loss function
        global loss_function
        loss_function = get_loss_function(config.loss_function).to(device)

        step = 0
        epoch_durations = []
        for epoch in range(config.epochs):
            logger.info('EPOCH {}:'.format(epoch + 1))
            epoch_start_time = time.time()

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            logger.info("       Model training.")
            train_metrics, step = train_one_epoch(model, training_loader, optimizer, config, epoch, step)

            # We don't need gradients on to do reporting
            model.train(False)

            # Validate model
            logger.info("       Model validation.")
            validation_metrics = validate(model, validation_loader)
            
            # LR schedule
            if config.enable_lr_scheduler:
                scheduler.step(train_metrics.loss)
            wandb.log({"learning_rate": scheduler.optimizer.param_groups[0]['lr']}, step=step)

            # Log to Weights & Biases
            wandb.log({"avg_train_loss": train_metrics.loss}, step=step)
            wandb.log({"avg_val_loss": validation_metrics.loss}, step=step)

            # Print epoch statistics
            logger.info(f"      AVG_LOSS train {train_metrics.loss} valid {validation_metrics.loss}")
            # ogger.info(f"      AVG_ACCURACY train {train_metrics.accuracy * 100} % valid {validation_metrics.accuracy * 100} %")

            # Log epoch duration
            epoch_duration = time.time() - epoch_start_time
            wandb.log({"epoch_runtime (seconds)": epoch_duration}, step=step)
            epoch_durations.append(epoch_duration)

        # Log average epoch duration
        avg_epoch_runtime = sum(epoch_durations) / len(epoch_durations)
        wandb.log({"avg epoch runtime (seconds)": avg_epoch_runtime})

# TODO: Refactor accuracy function
def accuracy(outputs, targets):
    threshold = 0.5
    thresholded_outputs = (outputs > threshold).float() 
    return torch.all(thresholded_outputs == targets, dim=1).float().sum() / targets.sum()
    
    #return (torch.argmax((outputs > threshold).float(), dim=1) == torch.argmax(targets, dim=1)).sum() / (targets.shape[0] * targets.shape[2] * targets.shape[3])
    
    
def gather_batch_metrics(outputs, targets):
    
    loss = loss_function(outputs, targets)

    metrics = munch.Munch()

    # metrics.accuracy = accuracy_function(outputs, targets)

    return loss, metrics

def prepare_metrics():
    metrics = munch.Munch()
    metrics.loss = 0.0
    # metrics.accuracy = 0.0

    return metrics

def test_CUDA():
    if torch.cuda.is_available():
        logger.info("PyTorch is running on CUDA!")
        logger.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
        device_id = torch.cuda.current_device()
        logger.info(f"Device ID: {device_id}")
        logger.info(f"Device name: {torch.cuda.get_device_name(device_id)}")
        return True
    else:
        logger.warning("PyTorch is not running on CUDA!")
        return False
    
def get_loss_function(loss_function_name):
    loss_func = None
    if loss_function_name == "MSELoss":
        loss_func = torch.nn.MSELoss()
    if loss_function_name == "HuberLoss":
        loss_func = torch.nn.HuberLoss()

    #TODO: MSELoss, HuberLoss
    return loss_func

def prepare_globals(present_data=False):    
    
    # Get logger
    global logger
    logger = Logger.get_logger(__name__)

    # Load configuration
    global config
    config = Configuration.load()

    # Initialize Weights & Biases
    wandb.login(key='04b4aa0a2ed5be3c78c42fcf424d91250474f4ff')
    
    is_cuda = test_CUDA()
    
    global device
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')   
        
    global toImage
    toImage = torch_trans.ToPILImage()
    # Set metrics functions    
    global accuracy_function
    accuracy_function = None
    
    global global_config
    global_config = config.unet_training.global_parameters
    global_config.update({'present_data': present_data})

    if global_config.present_data:
        with Image.open("/notebooks/mel-steg-cINN/Lenna_(test_image).png") as img:
            img.show()
        
def run(sweep=False, present_data=False):
    
    prepare_globals(present_data)
    
    if sweep:        
        sweep_id = wandb.sweep(config.unet_training.sweep_config, project="mel-steg-cINN", entity="snikiel")
        wandb.agent(sweep_id, function=train, count=20)
    else:
        train(config.unet_training.regular_config)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train UNET for mel-spectrogram colorization.')
    parser.add_argument('--sweep', action='store_true', help='run Weights & Biases sweep')

    args = parser.parse_args()
    
    run(args.sweep)