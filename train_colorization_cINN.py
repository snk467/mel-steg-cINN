#!/usr/bin/env python
import sys
import os
from tkinter import S

import torch
import torch.nn
import torch.optim
from torch.nn.functional import avg_pool2d#, interpolate
from torch.autograd import Variable
import numpy as np
import tqdm
from Noise import GaussianNoise

# TODO: Support mojego configu
import colorization_cINN.config as c
import Configuration
config = Configuration.load()

import colorization_cINN.model as model

# TODO: Support datasetów
from datasets import SpectrogramsDataset

# TODO: Własny moduł wizualizacji
# import viz

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')  

if c.load_file:
    model.load(c.load_file)

class dummy_loss(object):
    def item(self):
        return 1.

def sample_outputs(sigma, out_shape):
    return [sigma * torch.FloatTensor(torch.Size((2, o))).normal_().to(device) for o in out_shape]

tot_output_size = 2 * c.img_dims[0] * c.img_dims[1]

training_set = SpectrogramsDataset(config.training.dataset_location,
                                    train=True,
                                    size=config.cinn_training.dataset_size,
                                    augmentor=GaussianNoise([0.0], [0.001, 0.001, 0.0]))

validation_set = SpectrogramsDataset(config.training.dataset_location,
                                    train=False,
                                    size=config.cinn_training.dataset_size,
                                    augmentor=GaussianNoise([0.0], [0.001, 0.001, 0.0]))

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=config.cinn_training.batch_size, shuffle=True, num_workers=2)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config.cinn_training.batch_size, shuffle=False, num_workers=2)


for i_epoch in range(-c.pre_low_lr, c.n_epochs):

    print("EPOCH:", i_epoch)

    loss_history = []

    # Ustawainie lr odpowiednio do epoki
    if i_epoch < 0:
        for param_group in model.optim.param_groups:
            param_group['lr'] = c.lr * 2e-2
    if i_epoch == 0:
        for param_group in model.optim.param_groups:
            param_group['lr'] = c.lr

    if c.end_to_end and i_epoch <= c.pretrain_epochs:
        for param_group in model.feature_optim.param_groups:
            param_group['lr'] = 0
        if i_epoch == c.pretrain_epochs:
            for param_group in model.feature_optim.param_groups:
                param_group['lr'] = 1e-4

    iterator = tqdm.tqdm(enumerate(iter(training_loader)),
                            total=min(len(training_loader), c.n_its_per_epoch),
                            leave=False,
                            mininterval=1.,
                            disable=(not c.progress_bar),
                            ncols=83)

    for i_batch , x in iterator:

        print("BATCH:", i_batch)

        L, ab, _, _ = x

        input = torch.cat((L, ab), dim=1)

        zz, jac = model.combined_model(input)

        neg_log_likeli = 0.5 * zz - jac

        l = torch.mean(neg_log_likeli) / tot_output_size
        l.backward()

        model.optim_step()
        loss_history.append([l.item(), 0.])

        if i_batch+1 >= c.n_its_per_epoch:
            iterator.close()
            break

    epoch_losses = np.mean(np.array(loss_history), axis=0)
    print("LOSS: ", epoch_losses)
    epoch_losses[1] = np.log10(model.optim.param_groups[0]['lr'])
    for i in range(len(epoch_losses)):
        epoch_losses[i] = min(epoch_losses[i], c.loss_display_cutoff)        

    with torch.no_grad():
        ims = []
        for x in validation_loader:
            x_l, x_ab, cond, ab_pred = model.prepare_batch(x)

            for i in range(3):
                z = sample_outputs(c.sampling_temperature, model.output_dimensions)
                x_ab_sampled = model.combined_model.module.reverse_sample(z, cond)
                # TODO: Prezentacja wyników
                # ims.extend(list(data.norm_lab_to_rgb(x_l, x_ab_sampled)))

            break

    if i_epoch >= c.pretrain_epochs * 2:
        model.weight_scheduler.step(epoch_losses[0])
        model.feature_scheduler.step(epoch_losses[0])

    

    # viz.show_imgs(*ims)
    # viz.show_loss(epoch_losses)

    if i_epoch > 0 and (i_epoch % c.checkpoint_save_interval) == 0:
        model.save(c.filename + '_checkpoint_%.4i' % (i_epoch * (1-c.checkpoint_save_overwrite)))

os.makedirs(os.path.dirname(c.filename), exist_ok=True)
model.save(c.filename)
