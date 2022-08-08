import time
import torch
import torchvision
from torchmetrics import Accuracy
import wandb
from Datasets.SpectrogramsDataset import SpectrogramsDataset 
import Configuration
from Models.UNET.unet_model import UNet
import Logger
import munch

loss_function = torch.nn.MSELoss()
accuracy = Accuracy()

def train_one_epoch(model, training_loader, optimizer, config, epoch, step):
    running_metrics = prepare_metrics()
    avg_metrics = prepare_metrics()

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, targets = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        # print(inputs.shape)
        outputs = model(inputs.float())

        # Compute the loss and its gradients
        loss, metrics = gather_batch_metrics(outputs, targets)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data
        running_metrics.loss += loss.item()
        running_metrics.accuracy += metrics.accuracy

        avg_metrics.loss += loss.item()
        avg_metrics.accuracy += metrics.accuracy

        # Report
        if i % config.batch_checkpoint == (config.batch_checkpoint - 1):
            step +=1

            # Calculate current checkpoint metrics
            current_loss = running_metrics.loss / config.batch_checkpoint
            current_accuracy = running_metrics.accuracy / config.batch_checkpoint

            # Log to stdout
            logger.info(f"      batch {i + 1} loss: {current_loss} accuracy: {current_accuracy * 100} %")

            # Log to Weights & Biases
            wandb.log({"train_loss": current_loss, "train_acc": current_accuracy, "epoch": epoch + ((i+1)/len(training_loader))}, step=step)

            # Reset batch metrics
            running_metrics = prepare_metrics()

        if (i + 1) == len(training_loader):
            avg_metrics.loss = avg_metrics.loss / len(training_loader)
            avg_metrics.accuracy = avg_metrics.accuracy / len(training_loader)


    return avg_metrics, step

def validate(model, validation_loader):
    avg_metrics = prepare_metrics()

    for i, vdata in enumerate(validation_loader):
        vinputs, vtargets = vdata
        voutputs = model(vinputs.float())

        vloss, metrics = gather_batch_metrics(voutputs, vtargets)

        avg_metrics.loss += vloss.item()
        avg_metrics.accuracy += metrics.accuracy


    avg_metrics.loss = avg_metrics.loss / len(validation_loader) 
    avg_metrics.accuracy = avg_metrics.accuracy / len(validation_loader) 

    return avg_metrics

def train(config):
    # Create datasets for training & validation
    training_set = SpectrogramsDataset(config.spectrogram_files_directory, train=True)
    validation_set = SpectrogramsDataset(config.spectrogram_files_directory, train=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=config.parameters.batch_size, shuffle=False, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config.parameters.batch_size, shuffle=False, num_workers=2)

    # Create model
    model = UNet(n_channels=1, n_classes=256).float()

    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.Adam(model.parameters(), lr=config.parameters.learning_rate)

    step = 0
    epoch_durations = []
    for epoch in range(config.parameters.epochs):
        logger.info('EPOCH {}:'.format(epoch + 1))
        epoch_start_time = time.time()

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_metrics, step = train_one_epoch(model, training_loader, optimizer, config, epoch, step)

        # We don't need gradients on to do reporting
        model.train(False)

        # Validate model
        validation_metrics = validate(model, validation_loader)

        # Log to Weights & Biases
        wandb.log({"avg_train_loss": train_metrics.loss, "avg_train_acc": train_metrics.accuracy}, step=step)
        wandb.log({"avg_val_loss": validation_metrics.loss, "avg_val_acc": validation_metrics.accuracy}, step=step)

        # Print epoch statistics
        logger.info(f"      AVG_LOSS train {train_metrics.loss} valid {validation_metrics.loss} %")
        logger.info(f"      AVG_ACCURACY train {train_metrics.accuracy * 100} % valid {validation_metrics.accuracy * 100} %")

        # Log epoch duration
        epoch_duration = time.time() - epoch_start_time
        wandb.log({"epoch_runtime (seconds)": epoch_duration}, step=step)
        epoch_durations.append(epoch_duration)

    # Log average epoch duration
    avg_epoch_runtime = sum(epoch_durations) / len(epoch_durations)
    wandb.log({"avg epoch runtime (seconds)": avg_epoch_runtime})

def gather_batch_metrics(outputs, targets):
    loss = loss_function(outputs, targets)

    metrics = munch.Munch()
    metrics.accuracy = accuracy(outputs, targets.int())

    return loss, metrics

def prepare_metrics():
    metrics = munch.Munch()
    metrics.loss = 0.0
    metrics.accuracy = 0.0

    return metrics


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



