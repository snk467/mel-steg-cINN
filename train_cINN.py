#!/usr/bin/env python
import sys
import os

import torch
import torch.nn
import torch.optim
from torch.nn.functional import avg_pool2d#, interpolate
from torch.autograd import Variable
import numpy as np
import tqdm
import torchmetrics as torch_metrics

from noise import GaussianNoise
import torch.nn.functional as F

# TODO: Support mojego configu
import colorization_cINN.config as c
from config import config

import colorization_cINN.model as model

from datasets import SpectrogramsDataset

# TODO: Własny moduł wizualizacji
import visualization

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')  

if c.load_file:
    model.load(c.load_file)

class dummy_loss(object):
    def item(self):
        return 1.
    
metric = torch_metrics.MeanSquaredError().to(device)

def sample_outputs(sigma, out_shape, batch_size):
    return [sigma * torch.FloatTensor(torch.Size((batch_size, o))).normal_().to(device) for o in out_shape]

def run_cinn_training():
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

        model.combined_model.train()
        
        print(model.combined_model.feature_network.training)
        print(model.combined_model.fc_cond_network.training)
        print(model.combined_model.inn.training)

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
            if (i_batch + 1) % 10 == 0:
                print("\tBATCH:", i_batch + 1)
                print("\tLOSS:", np.mean(np.array(loss_history), axis=0))

            L, ab, _, _ = x            

            input = torch.cat((L, ab), dim=1).to(device)

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
        print("\tLOSS: ", epoch_losses)
        epoch_losses[1] = np.log10(model.optim.param_groups[0]['lr'])
        for i in range(len(epoch_losses)):
            epoch_losses[i] = min(epoch_losses[i], c.loss_display_cutoff)        

        with torch.no_grad():
            print("VALIDATION...")
            ims = []    
            mse = []
            for x in validation_loader:
                x_l, x_ab, cond, ab_pred = model.combined_model.prepare_batch(x)                

                for i in range(1):
                    z = sample_outputs(c.sampling_temperature, model.output_dimensions, x[0].shape[0])

                # i = 0
                # for z_i in z:
                #     print(f"\tz_{i}:", len(z_i))
                #     print(f"\tz_{i}:", type(z_i))
                #     print(f"\tz_{i}:", z_i.shape)
                #     for zz in z_i:
                        
                #         print("\t\t", zz.shape)

                #     i += 1

                   # print("cond.shape",cond[0].shape)

                    x_ab_sampled = model.combined_model.reverse_sample(z, cond)
                    
                    mse.append(metric(x_ab, x_ab_sampled[0]).item())

                # print("\tSampled ab:", x_ab_sampled[0].shape)
                # TODO: Prezentacja wyników
                # ims.extend(list(data.norm_lab_to_rgb(x_l, x_ab_sampled)))

                break
                
            print("MSE metric:", np.mean(np.array(mse)))

        if i_epoch >= c.pretrain_epochs * 2:
            model.weight_scheduler.step(epoch_losses[0])
            model.feature_scheduler.step(epoch_losses[0])

    

    # viz.show_imgs(*ims)
    # viz.show_loss(epoch_losses)

        if i_epoch > 0 and (i_epoch % c.checkpoint_save_interval) == 0:
            model.save(c.filename + '_checkpoint_%.4i' % (i_epoch * (1-c.checkpoint_save_overwrite)))

    os.makedirs(os.path.dirname(c.filename), exist_ok=True)
    model.save(c.filename)


    print("VALIDATION:")
    print("\tValidation batch:")
    validation_batch = validation_set[0]
    visualization.show_data(*validation_batch)
    validation_z = sample_outputs(c.sampling_temperature, model.output_dimensions, 1)
    x_l, x_ab, cond, ab_pred = model.combined_model.prepare_batch(validation_batch)
# cond[0] = cond[0][None, :]
    cond[1] = cond[1][None, :]
    print(cond[0].shape)
    print(cond[1].shape)
    print("===")
    x_ab_sampled, b = model.combined_model.reverse_sample(validation_z, cond)   
    print(x_ab_sampled.shape)
    print(b.shape)
    visualization.show_data(x_l[0], x_ab_sampled[0], validation_batch[2], validation_batch[3])

if __name__ == "__main__":
    run_cinn_training()


