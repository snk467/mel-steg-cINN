import time
import torch
import torchvision
import wandb
from Datasets.SpectrogramsDataset import SpectrogramsDataset 
import Configuration
from Models.UNET.unet_model import UNet
import Logger

def train_one_epoch(model, training_loader, optimizer, config, epoch, step):
    running_loss = 0.
    last_loss = 0.
    number_of_batches = len(training_loader)

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_function(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data
        running_loss += loss.item()

        # Report
        if i % config.batch_checkpoint == (config.batch_checkpoint - 1):
            step +=1
            last_loss = running_loss / config.batch_checkpoint
            logger.log(f"  batch {i + 1} loss: {last_loss}")
            wandb.log({"train_loss": running_loss/config.batch_checkpoint, "epoch": epoch + ((i+1)/len(training_loader))}, step=step)
            running_loss = 0.

    return last_loss, step

def validate(model, validation_loader, step):
    running_vloss = 0.

    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = loss_function(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)    
    wandb.log({"val_loss": avg_vloss}, step=step)

    return avg_vloss

def loss_function(validation_outputs, validation_labels):
    return 1000.

def train(config):
    # Create datasets for training & validation
    training_set = SpectrogramsDataset(config.spectrogram_files_directory, train=True)
    validation_set = SpectrogramsDataset(config.spectrogram_files_directory, train=False)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=config.parameters.batch_size, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config.parameters.batch_size, shuffle=False, num_workers=2)

    # Create model
    model = UNet()

    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.Adam(model.parameters(), lr=config.parameters.learning_rate)

    step = 0
    epoch_durations = []
    for epoch in range(config.parameters.epochs):
        logger.info('EPOCH {}:'.format(epoch + 1))
        epoch_start_time = time.time()

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss, step = train_one_epoch(model, training_loader, optimizer, config, epoch, step)

        # We don't need gradients on to do reporting
        model.train(False)

        # Validate model
        avg_vloss = validate(model, validation_loader, step)

        # Log epoch duration
        epoch_duration = time.time() - epoch_start_time
        wandb.log({"epoch_runtime (seconds)": epoch_duration}, step=step)
        epoch_durations.append(epoch_duration)

        # Print epoch statistics
        logger.info('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log average epoch duration
    avg_epoch_runtime = sum(epoch_durations) / len(epoch_durations)
    wandb.log({"avg epoch runtime (seconds)": avg_epoch_runtime})



if __name__ == "__main__":
    # Get logger
    logger = Logger.get_logger(__name__)

    # Load configuration
    config = Configuration.load()

    # Initialize Weights & Biases
    wandb.login(key='***REMOVED***')
    wandb.init(project="mel-steg-cINN", entity="snikiel")
    wandb.config = config.unet_training.parameters

    train(config.unet_training)



