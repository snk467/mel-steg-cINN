import functools
import os

import numpy as np
import torch.optim
import wandb
from FrEIA.framework import *
from FrEIA.modules import *

import models
import utils.logger
import utils.utilities as utilities
from config import config as main_config
from models.cinn.cinn_parts import *
from models.unet.unet_models import *
from utils.utilities import logger

# region Globals

feature_channels = 64
fc_cond_length = 512
n_blocks_fc = 1
outputs = []

sched_factor = 0.2
sched_patience = 8
sched_trehsh = 0.001
sched_cooldown = 2

logger = utils.logger.get_logger(__name__)
device = utilities.get_device(verbose=False)


# endregion

# region Functions

class cINN_builder:
    def __init__(self, config_training):
        self.config_training = config_training

        self.conditions = [
            ConditionNode(8, main_config.cinn_management.img_dims[0], main_config.cinn_management.img_dims[1]),
            ConditionNode(16, main_config.cinn_management.img_dims[0] // 2,
                          main_config.cinn_management.img_dims[1] // 2),
            ConditionNode(32, main_config.cinn_management.img_dims[0] // 4,
                          main_config.cinn_management.img_dims[1] // 4),
            ConditionNode(64, main_config.cinn_management.img_dims[0] // 8,
                          main_config.cinn_management.img_dims[1] // 8)]

        self.fc_cond_net = nn.Sequential(*[
            nn.LeakyReLU(),
            nn.Conv2d(feature_channels, fc_cond_length, 3, stride=16, padding=1),
            nn.AvgPool2d(4),
            nn.BatchNorm2d(fc_cond_length),
        ])

    def random_orthog(self, n):
        w = np.random.randn(n, n)
        w = w + w.T
        w, S, V = np.linalg.svd(w)
        return torch.FloatTensor(w)

    def _add_conditioned_section(self, nodes, depth, channels_in, channels, cond_level):
        for k in range(depth):
            subnet_kwargs = {'leaky_slope': 5e-2, 'channels_hidden': channels}
            nodes.append(Node([nodes[-1].out0], GLOWCouplingBlock,
                              {'clamp': self.config_training.clamping,
                               'subnet_constructor': functools.partial(F_conv, **subnet_kwargs)},
                              conditions=[self.conditions[cond_level]], name=F'conv_{k}'))
            nodes.append(Node([nodes[-1].out0], Fixed1x1Conv, {'M': self.random_orthog(channels_in)}))

    def _add_split_downsample(self, nodes, split, downsample, channels_in, channels):
        if downsample == 'haar':
            nodes.append(
                Node([nodes[-1].out0], HaarDownsampling, {'rebalance': 0.5, 'order_by_wavelet': True}, name='haar'))
        if downsample == 'reshape':
            nodes.append(Node([nodes[-1].out0], IRevNetDownsampling, {}, name='reshape'))

        for i in range(2):
            nodes.append(Node([nodes[-1].out0], Fixed1x1Conv, {'M': self.random_orthog(channels_in * 4)}))
            subnet_kwargs = {'kernel_size': 1, 'leaky_slope': 1e-2, 'channels_hidden': channels}
            nodes.append(Node([nodes[-1].out0], GLOWCouplingBlock,
                              {'clamp': self.config_training.clamping,
                               'subnet_constructor': functools.partial(F_conv, **subnet_kwargs)}))

        if split:
            nodes.append(Node([nodes[-1].out0], Split,
                              {'section_sizes': split, 'dim': 0}, name='split'))

            output = Node([nodes[-1].out1], Flatten, {}, name='flatten')
            nodes.insert(-2, output)
            nodes.insert(-2, OutputNode([output.out0], name='out'))

    def _add_fc_section(self, nodes):
        nodes.append(Node([nodes[-1].out0], Flatten, {}, name='flatten'))
        for k in range(n_blocks_fc):
            nodes.append(Node([nodes[-1].out0], PermuteRandom, {'seed': k}, name=F'permute_{k}'))
            subnet_kwargs = {'internal_size': fc_cond_length}
            nodes.append(Node([nodes[-1].out0], GLOWCouplingBlock,
                              {'clamp': self.config_training.clamping,
                               'subnet_constructor': functools.partial(F_fully_connected, **subnet_kwargs)},
                              conditions=[], name=F'fc_{k}'))

        nodes.append(OutputNode([nodes[-1].out0], name='out'))

    def build_cinn(self):
        nodes = [InputNode(2, *main_config.cinn_management.img_dims, name='inp')]
        # 2x64x64 px
        self._add_conditioned_section(nodes, depth=2, channels_in=2, channels=8, cond_level=0)
        self._add_split_downsample(nodes, split=False, downsample='reshape', channels_in=2, channels=8)

        # 8x32x32 px
        self._add_conditioned_section(nodes, depth=2, channels_in=8, channels=16, cond_level=1)
        self._add_split_downsample(nodes, split=(16, 16), downsample='reshape', channels_in=8, channels=16)

        # 16x16x16 px
        self._add_conditioned_section(nodes, depth=2, channels_in=16, channels=32, cond_level=2)
        self._add_split_downsample(nodes, split=(32, 32), downsample='reshape', channels_in=16, channels=32)

        # 32x8x8 px
        self._add_conditioned_section(nodes, depth=2, channels_in=32, channels=64, cond_level=3)
        self._add_split_downsample(nodes, split=(32, 3 * 32), downsample='haar', channels_in=32, channels=64)

        # 32x4x4 = 512 px
        self._add_fc_section(nodes)

        return ReversibleGraphNet(nodes + self.conditions, verbose=False), nodes

    def init_model(self, mod):
        for key, param in mod.named_parameters():
            split = key.split('.')
            if param.requires_grad:
                param.data = self.config_training.init_scale * torch.randn(param.data.shape)
                if len(split) > 3 and split[3][-1] == '3':  # last convolution in the coeff func
                    param.data.fill_(0.)

    def __calculate_output_dimenstions(self, nodes):
        output_dimensions = []
        for o in nodes:
            if type(o) is OutputNode:
                output_dimensions.append(o.input_dims[0][0])

        return output_dimensions

    def get_cinn(self):
        cinn, nodes = self.build_cinn()
        output_dimensions = self.__calculate_output_dimenstions(nodes)

        self.init_model(cinn)

        return cinn, output_dimensions

    def get_feature_net(self):
        feature_net = UNet(1)
        feature_net.load_state_dict(
            torch.load(main_config.cinn_management.feature_net_path, map_location=utilities.get_device(verbose=False)))
        feature_net.eval()
        return feature_net

    def get_fc_cond_net(self):
        return self.fc_cond_net


# endregion

class WrappedModel(nn.Module):
    def __init__(self, feature_network, fc_cond_network, inn):
        super().__init__()

        self.feature_network = feature_network
        self.fc_cond_network = fc_cond_network
        self.inn = inn

    def forward(self, x):

        x_l = x[:, 0:1, :, :]
        x_ab = x[:, 1:, :, :]

        logger.debug(f"x_L input shape: {x_l.shape}")
        logger.debug(f"x_ab input shape: {x_ab.shape}")

        x_ab = F.interpolate(x_ab, size=main_config.cinn_management.img_dims)

        logger.debug(f"x_ab shape after interpolate: {x_ab.shape}")

        if main_config.cinn_management.end_to_end:
            features, _ = self.feature_network.features(x_l)
        else:
            with torch.no_grad():
                features, _ = self.feature_network.features(x_l)

        cond = [*features]

        for i, c in enumerate(cond):
            logger.debug(f"cond[{i}].shape: {c.shape}")

        z, jac = self.inn(x_ab, cond)
        zz = sum(torch.sum(o ** 2) for o in z)

        return z, zz, jac

    def reverse_sample(self, z, cond):
        return self.inn(z, cond, rev=True)

    def train(self, mode: bool = True):
        self.feature_network.train(main_config.cinn_management.end_to_end and mode)
        self.fc_cond_network.train(mode)
        self.inn.train(mode)

    def eval(self):
        self.feature_network.eval()
        self.fc_cond_network.eval()
        self.inn.eval()

    def istraining(self):
        return self.inn.training, self.feature_network.training

    def prepare_batch(self, x):
        self.eval()

        net_feat = self.feature_network
        net_inn = self.inn
        net_cond = self.fc_cond_network

        x_l, x_ab, _, _ = x
        x_l = x_l.to(device)

        if x_l.ndim == 3:
            x_l = x_l[None, :]

        if x_ab.ndim == 3:
            x_ab = x_ab[None, :]

        x_ab = F.interpolate(x_ab, size=main_config.cinn_management.img_dims)

        features, from_features = net_feat.features(x_l)
        ab_pred = net_feat.forward_from_features(from_features)
        cond = [*features]

        self.train()

        return x_l.detach().to('cpu'), x_ab.detach().to('cpu'), cond, ab_pred.detach().to('cpu')


class cINNTrainingUtilities:
    def __init__(self, model: WrappedModel, config: main_config.cinn_training) -> None:
        self.model = model
        self.config_training = config
        self.feature_optimizer = None
        self.feature_scheduler = None
        self.optimizer = None
        self.scheduler = None

        if config is not None:
            params_trainable = (list(filter(lambda p: p.requires_grad, model.inn.parameters()))
                                + list(model.fc_cond_network.parameters()))

            self.optimizer = torch.optim.Adam(params_trainable, lr=self.config_training.lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        factor=sched_factor,
                                                                        patience=sched_patience,
                                                                        threshold=sched_trehsh,
                                                                        min_lr=0, eps=1e-08,
                                                                        cooldown=sched_cooldown,
                                                                        verbose=True)
            if main_config.cinn_management.end_to_end:
                self.__load_feature_optimizer_and_scheduler()

    def load(self, path, device):
        state_dicts = torch.load(path, map_location=device)

        self.model.load_state_dict(state_dicts['net'])
        self.model.to(device)

        if not self.optimizer:
            return

        try:
            self.optimizer.load_state_dict(state_dicts['opt'])

            if not main_config.cinn_management.end_to_end:
                self.feature_optimizer = None
            elif state_dicts['opt_f'] is None:
                self.__load_feature_optimizer_and_scheduler()
            else:
                self.feature_optimizer.load_state_dict(state_dicts['opt_f'])
        except:
            logger.error('Cannot load optimizer for some reason or other')

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.feature_optimizer is not None:
            logger.debug("Feature_optimizer step.")
            self.feature_optimizer.step()
            self.feature_optimizer.zero_grad()

    def scheduler_step(self, value):
        self.scheduler.step(value)

        if self.feature_scheduler is not None:
            self.feature_scheduler.step(value)

    def save(self, path):
        torch.save({'opt': self.optimizer.state_dict(),
                    'opt_f': None if self.feature_optimizer is None else self.feature_optimizer.state_dict(),
                    'net': self.model.state_dict()}, path)

    def __load_feature_optimizer_and_scheduler(self):
        self.feature_optimizer = torch.optim.Adam(self.model.feature_network.parameters(),
                                                  lr=self.config_training.lr_feature_net,
                                                  betas=self.config_training.betas,
                                                  eps=1e-4)
        self.feature_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.feature_optimizer,
                                                                            factor=sched_factor,
                                                                            patience=sched_patience,
                                                                            threshold=sched_trehsh,
                                                                            min_lr=0, eps=1e-08,
                                                                            cooldown=sched_cooldown,
                                                                            verbose=True)


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_metric = np.inf

    def early_stop(self, validation_metric):
        if validation_metric < self.min_validation_metric:
            self.min_validation_metric = validation_metric
            self.counter = 0
            logger.info(f"New min_validation_metric: {self.min_validation_metric}")
        elif validation_metric > (self.min_validation_metric + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(
                    f"Early stopping - validation_metric: {validation_metric} and min_validation_metric: {self.min_validation_metric} + {self.min_delta}(min_delta)")
                return True
        return False


def get_cinn_model(cinn_training_config, filename=None, run_path=None, device='cpu'):
    if run_path is not None:
        try:
            restored_model = wandb.restore(filename, run_path=run_path, root=os.path.join(os.getcwd(),"tmp",run_path))
        except ValueError:
            restored_model = wandb.restore(os.path.join("tmp", filename), run_path=run_path, root=os.path.join(os.getcwd(),"tmp",run_path))

        logger.info(f"Downloaded {filename}: {run_path}")

    cinn_builder = models.cinn.cinn_model.cINN_builder(cinn_training_config)

    feature_net = cinn_builder.get_feature_net()
    fc_cond_net = cinn_builder.get_fc_cond_net()
    cinn_net, cinn_output_dimensions = cinn_builder.get_cinn()

    cinn_model = models.cinn.cinn_model.WrappedModel(feature_net, fc_cond_net, cinn_net).to(device)
    cinn_utilities = models.cinn.cinn_model.cINNTrainingUtilities(cinn_model.float(), cinn_training_config)

    if run_path is not None:
        cinn_utilities.load(restored_model.name, device=device)
        logger.info(f"Loaded {filename}: {run_path}")

    return cinn_utilities, cinn_output_dimensions
