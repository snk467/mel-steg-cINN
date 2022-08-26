import time
import torch
import torchvision
import torchmetrics as torch_metrics
import random
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
    running_metrics = None
    avg_metrics = None

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
        running_metrics = add_metrics(running_metrics, metrics)
        avg_metrics = add_metrics(avg_metrics, metrics)
        
        # Report
        if i % global_config.batch_checkpoint == (global_config.batch_checkpoint - 1):
            step +=1

            # Calculate current checkpoint metrics
            current_metrics = divide_metrics(running_metrics, global_config.batch_checkpoint)

            # Log metrics           
            log_metrics(metrics, "train", step, batch_id=i)
            
            wandb.log({"epoch": epoch + ((i+1)/len(training_loader))}, step=step)

            # Reset batch metrics
            running_metrics = None

        if (i + 1) == len(training_loader):
            avg_metrics = divide_metrics(avg_metrics, len(training_loader))
            
    return avg_metrics, step

def validate(model, validation_loader):
    avg_metrics = None

    for i, vdata in enumerate(validation_loader):
        vinputs, vtargets, _ = vdata
        
        vinputs = vinputs.to(device).float()
        vtargets = vtargets.to(device).float()
        
        voutputs = model(vinputs)

        vloss, metrics = gather_batch_metrics(voutputs, vtargets)

        avg_metrics = add_metrics(avg_metrics, metrics)
    
    avg_metrics = divide_metrics(avg_metrics, len(validation_loader) )

    return avg_metrics



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

        # Show first element
        if global_config.present_data:
            print("Input example:")
            example_id = random.randint(0, len(training_set) - 1)
            show_element(*training_set[example_id])
        
        # Create model
        model = UNet(n_channels=1)
        model = model.to(device).float()

        # Optimizers specified in the torch.optim package
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        
        # LR scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=3, min_lr=0, factor=0.1)
        
        # Get loss function
        metrics_functions["Loss"] = get_loss_function(config.loss_function).to(device) 

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
                scheduler.step(train_metrics["Loss"])
                
            wandb.log({"learning_rate": scheduler.optimizer.param_groups[0]['lr']}, step=step)

            # Log metrics
            log_metrics(train_metrics, "TRAIN AVG", step)
            log_metrics(validation_metrics, "VALID AVG", step)

            # Log epoch duration
            epoch_duration = time.time() - epoch_start_time
            wandb.log({"epoch_runtime (seconds)": epoch_duration}, step=step)
            epoch_durations.append(epoch_duration)

        # Predict first element
        if global_config.present_data:
            example_id = 0 # random.randint(0, len(training_set) - 1)
            input, target, filename = training_set[example_id]
            batched_input = torch.reshape(input, (1, *input.shape)).to(device).float()        
            output = model(batched_input)
            print("Result:")
            show_element(batched_input[0], output[0], filename)
            print("Target:")
            show_element(input, target, filename)
        
        
        # Log average epoch duration
        avg_epoch_runtime = sum(epoch_durations) / len(epoch_durations)
        wandb.log({"avg epoch runtime (seconds)": avg_epoch_runtime})
        
        # Finish the Weights & Biases run
        wandb.finish()
    
def gather_batch_metrics(outputs, targets):
    metrics = dict(metrics_functions)     
    loss = None
    for func in metrics_functions:
        metrics[func] = metrics_functions[func](outputs, targets)    

    return metrics["Loss"], metrics

def add_metrics(metrics1, metrics2):
    if metrics1 is None:
        return dict(metrics2)

    metrics_sum = dict(metrics1)
    for metrics in metrics2:                
        metrics_sum[metrics] += metrics2[metrics]
    
    return metrics_sum

def divide_metrics(metrics, divisor):
    
    if metrics is None:
        return None
    
    metrics_divided = dict(metrics)
    
    for metrics_name in metrics:                
            metrics_divided[metrics_name] /= divisor
            
    return metrics_divided  

def log_metrics(metrics, phase, step, batch_id=None):
    
    if batch_id is not None:
        logger_message = f"        batch {batch_id + 1}"
    else:
        logger_message =f"{phase} METRICS"
    
    for metrics_name in metrics:
        # Log to stdout
        logger_message += f" {metrics_name}: {metrics[metrics_name]}"

        # Log to Weights & Biases
        wandb.log({f'{phase.replace(" ", "_")}_{metrics_name}': metrics[metrics_name]}, step=step)
        
    logger.info(logger_message)
    
def show_element(input_in, target_in, filename_in):
    input = input_in.detach().cpu()
    target = target_in.detach().cpu()
    filename = filename_in

    print("L shape:", input.shape)
    print("ab shape:", target.shape)
    print("Filename:", filename)

    L_img = toImage(input).convert('RGB')       

    a_img = toImage(target[0]).convert('RGB') 

    b_img = toImage(target[1]).convert('RGB') 

    rgb_img = get_rgb_image_from_lab_channels(input, target)  
    
    Image.fromarray(np.hstack((np.array(L_img), np.array(a_img), np.array(b_img), np.array(rgb_img)))).show()

    
def get_rgb_image_from_lab_channels(L_channel, ab_channels):
    colormap_lab = LUT.Colormap.from_colormap("parula_norm_lab")  
    
    L_np = L_channel.numpy()
    ab_np = ab_channels.numpy()    
    
    Lab_np = np.concatenate((L_np, ab_np))
    Lab_np = np.moveaxis(Lab_np, 0, -1)
    
    indexes = colormap_lab.get_indexes_from_colors(Lab_np)                            
    colormap_rgb = LUT.Colormap.from_colormap("parula_rgb")
    img_target = colormap_rgb.get_colors_from_indexes(indexes)
    img_target = toImage((img_target * 255).astype(np.uint8))
    
    return img_target

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
    if loss_function_name == "L1Loss":
        loss_func = torch.nn.L1Loss()
    return loss_func

def prepare_globals(present_data=False):    
    
    # Get logger
    global logger
    logger = Logger.get_logger(__name__)

    # Load configuration
    global config
    config = Configuration.load()

    # Initialize Weights & Biases
    wandb.login(key='***REMOVED***')
    
    is_cuda = test_CUDA()
    
    global device
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')   
        
    global toImage
    toImage = torch_trans.ToPILImage()
    
    # Set metrics functions    
    global metrics_functions
    metrics_functions = {
        "Loss": None,
        "MSE": torch_metrics.MeanSquaredError().to(device),
        "HuberLoss": torch.nn.HuberLoss().to(device),
        "MAE": torch_metrics.MeanAbsoluteError().to(device),
        # "RMSE": torch_metrics.MeanSquaredError(squared=False).to(device),
        # "MAPE": torch_metrics.MeanAbsolutePercentageError().to(device)
        
    }
    
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
        wandb.agent(sweep_id, function=train, count=config.unet_training.global_parameters.sweep_count)
    else:
        train(config.unet_training.regular_config)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train UNET for mel-spectrogram colorization.')
    parser.add_argument('--sweep', action='store_true', help='run Weights & Biases sweep')

    args = parser.parse_args()
    
    run(args.sweep)