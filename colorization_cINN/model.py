import functools
import warnings

import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from FrEIA.framework import *
from FrEIA.modules import *
from cINN_components.coeff_functs import *
from colorization_cINN.subnet_coupling import *
import colorization_cINN.config as c

import configuration

config = configuration.load()

feature_channels = 256
fc_cond_length = 512
n_blocks_fc = 8
outputs = []

conditions = [ConditionNode(feature_channels, c.img_dims[0], c.img_dims[1]),
              ConditionNode(fc_cond_length)]

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')   

def random_orthog(n):
    w = np.random.randn(n, n)
    w = w + w.T
    w, S, V = np.linalg.svd(w)
    return torch.FloatTensor(w)

def cond_subnet(level, c_out, extra_conv=False):
    c_intern = [feature_channels, 128, 128, 256]
    modules = []

    for i in range(level):
        modules.extend([nn.Conv2d(c_intern[i], c_intern[i+1], 3, stride=2, padding=1),
                        nn.LeakyReLU() ])

    if extra_conv:
        modules.extend([
            nn.Conv2d(c_intern[level], 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 2*c_out, 3, padding=1),
        ])
    else:
        modules.append(nn.Conv2d(c_intern[level], 2*c_out, 3, padding=1))

    modules.append(nn.BatchNorm2d(2*c_out))

    return nn.Sequential(*modules)

fc_cond_net = nn.Sequential(*[
                              nn.Conv2d(feature_channels, 128, 3, stride=2, padding=1), # 32 x 32
                              nn.LeakyReLU(),
                              nn.Conv2d(128, 256, 3, stride=2, padding=1), # 16 x 16
                              nn.LeakyReLU(),
                              nn.Conv2d(256, 256, 3, stride=2, padding=1), # 8 x 8
                              nn.LeakyReLU(),
                              nn.Conv2d(256, fc_cond_length, 3, stride=2, padding=1), # 4 x 4
                              nn.AvgPool2d(4),
                              nn.BatchNorm2d(fc_cond_length),
                            ])

def _add_conditioned_section(nodes, depth, channels_in, channels, cond_level):

    for k in range(depth):
        nodes.append(Node([nodes[-1].out0],
                            subnet_coupling_layer,
                            {'clamp':c.clamping, 'F_class':F_conv,
                            'subnet':cond_subnet(cond_level, channels//2), 'sub_len':channels,
                            'F_args':{'leaky_slope': 5e-2, 'channels_hidden':channels}},
                            conditions=[conditions[0]], name=F'conv_{k}'))
        nodes.append(Node([nodes[-1].out0], Fixed1x1Conv, {'M':random_orthog(channels_in)}))


def _add_split_downsample(nodes, split, downsample, channels_in, channels):
    if downsample=='haar':
        nodes.append(Node([nodes[-1].out0], HaarDownsampling, {'rebalance':0.5, 'order_by_wavelet':True}, name='haar'))
    if downsample=='reshape':
        nodes.append(Node([nodes[-1].out0], IRevNetDownsampling, {}, name='reshape'))

    for i in range(2):
        nodes.append(Node([nodes[-1].out0], Fixed1x1Conv, {'M':random_orthog(channels_in*4)}))
        subnet_kwargs = {'kernel_size':1, 'leaky_slope': 1e-2, 'channels_hidden':channels}
        nodes.append(Node([nodes[-1].out0],GLOWCouplingBlock,
                {'clamp':c.clamping, 'subnet_constructor':functools.partial(F_conv, **subnet_kwargs)}))

    if split:
        nodes.append(Node([nodes[-1].out0], Split,
                        {'section_sizes': split, 'dim':0}, name='split'))

        output = Node([nodes[-1].out1], Flatten, {}, name='flatten')
        nodes.insert(-2, output)
        nodes.insert(-2, OutputNode([output.out0], name='out'))

def _add_fc_section(nodes):
    nodes.append(Node([nodes[-1].out0], Flatten, {}, name='flatten'))
    for k in range(n_blocks_fc):
        nodes.append(Node([nodes[-1].out0], PermuteRandom, {'seed':k}, name=F'permute_{k}'))
        subnet_kwargs = {'internal_size':512}
        nodes.append(Node([nodes[-1].out0], GLOWCouplingBlock,
                {'clamp':c.clamping, 'subnet_constructor':functools.partial(F_fully_connected, **subnet_kwargs)},
                conditions=[conditions[1]], name=F'fc_{k}'))

    nodes.append(OutputNode([nodes[-1].out0], name='out'))

nodes = [InputNode(2, *c.img_dims, name='inp')]
# 2x64x64 px
_add_conditioned_section(nodes, depth=4, channels_in=2, channels=32, cond_level=0)
_add_split_downsample(nodes, split=False, downsample='reshape', channels_in=2, channels=64)

# 8x32x32 px
_add_conditioned_section(nodes, depth=6, channels_in=8, channels=64, cond_level=1)
_add_split_downsample(nodes, split=(16, 16), downsample='reshape', channels_in=8, channels=128)

# 16x16x16 px
_add_conditioned_section(nodes, depth=6, channels_in=16, channels=128, cond_level=2)
_add_split_downsample(nodes, split=(32, 32), downsample='reshape', channels_in=16, channels=256)

# 32x8x8 px
_add_conditioned_section(nodes, depth=6, channels_in=32, channels=256, cond_level=3)
_add_split_downsample(nodes, split=(32, 3*32), downsample='haar', channels_in=32, channels=256)

# 32x4x4 = 512 px
_add_fc_section(nodes)

def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).to(device)
            if len(split) > 3 and split[3][-1] == '3': # last convolution in the coeff func
                param.data.fill_(0.)


cinn = ReversibleGraphNet(nodes + conditions, verbose=False)
output_dimensions = []
for o in nodes:
    if type(o) is OutputNode:
        output_dimensions.append(o.input_dims[0][0])

cinn.to(device)
init_model(cinn)
#init_model(fc_cond_net)

if c.load_inn_only:
    cinn.load_state_dict(torch.load(c.load_inn_only)['net'])

# Load feature net
from Models.UNET.unet_models import UNet_256
feature_net = torch.load(config.cinn_training.feature_net_path, map_location=device)
feature_net.to(device)
feature_net.eval()


# try:
#     pretrained_dict = torch.load('./features_pretrained.pt')
#     pretrained_dict = {k:v for k,v in pretrained_dict.items() if 'num_batches_tracked' not in k}
#     feature_net.load_state_dict(pretrained_dict)
# except FileNotFoundError:
#     warnings.warn("No loading pretrained weights for conditioning network (./features_pretrained.pt)")

# feature_net.to(device)

# TODO: Ładowanie zapisanej sieci
# feature_net.class8_ab.state_dict()['weight'].copy_(torch.from_numpy(np.load('./pts_in_hull.npy').T).view(2, 313, 1, 1))



class WrappedModel(nn.Module):
    def __init__(self, feature_network, fc_cond_network, inn):
        super().__init__()

        self.feature_network = feature_network
        self.fc_cond_network = fc_cond_network
        self.inn = inn

    def forward(self, x):

        # print(x.shape)

        x_l = x[:, 0:1, :, :]
        x_ab = x[:, 1:, :, :]

        # print(x_l.shape)
        # print(x_ab.shape)

        x_ab = F.interpolate(x_ab, size=c.img_dims)
        # x_ab += 5e-2 * torch.cuda.FloatTensor(x_ab.shape).normal_()

        if c.end_to_end:
            _,_,_,features = self.feature_network.features(x_l)
            # features = features[:, :, 1:-1, 1:-1]
        else:
            with torch.no_grad():
                _,_,_,features = self.feature_network.features(x_l)
                # features = features[:, :, 1:-1, 1:-1]

        cond = [features, self.fc_cond_network(features).squeeze()]

        z, jac = self.inn(x_ab, cond)

        # for item in z:
        #     print(type(item))
        #     if isinstance(item, tuple):
        #         print(len(item))
        #         for inner_item in item:
        #             print(type(inner_item), inner_item.shape)
                
        # print(z[0].shape)
        zz = sum(torch.sum(o**2) for o in z)
        # jac = self.inn.jacobian(run_forward=False)

        return zz, jac

    def reverse_sample(self, z, cond):
        return self.inn(z, cond, rev=True)

    def train(self, mode: bool = True, feature_mode: bool = False):
        self.feature_network.train(feature_mode)
        self.fc_cond_network.train(mode)
        self.inn.train(mode)

    def eval(self):
        self.feature_network.eval()
        self.fc_cond_network.eval()
        self.inn.eval()

    def istraining(self):
        return self.inn.training, self.feature_network.training

    def prepare_batch(self, x):
        mode = self.istraining()
        self.eval()

        net_feat = self.feature_network
        net_inn  = self.inn
        net_cond = self.fc_cond_network

    
        x_l, x_ab, _, _ = x

        # print("BEFORE")
        # print(x_l.shape)
        # print(x_ab.shape)

        if x_l.ndim == 3:
            x_l = x_l[None, :]

        if x_ab.ndim == 3:
            x_ab = x_ab[None, :]

        # print("AFTER")
        # print(x_l.shape)
        # print(x_ab.shape)

        # Na razie tego nie używamy
        x_ab = F.interpolate(x_ab, size=c.img_dims)
        # x_ab += 5e-2 * torch.cuda.FloatTensor(x_ab.shape).normal_()

        features = net_feat.features(x_l)
        # features = features[:, :, 1:-1, 1:-1]

        ab_pred = net_feat.forward_from_features(*features)

        # print(net_cond.training)
        cond = [features[-1], net_cond(features[-1]).squeeze()]

        self.train(*mode)

        return x_l.detach(), x_ab.detach(), cond, ab_pred

combined_model = WrappedModel(feature_net, fc_cond_net, cinn)
combined_model.to(device)
# combined_model = nn.DataParallel(combined_model, device_ids=c.device_ids)

params_trainable = (list(filter(lambda p: p.requires_grad, combined_model.inn.parameters()))
                  + list(combined_model.fc_cond_network.parameters()))

optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

sched_factor = 0.2
sched_patience = 8
sched_trehsh = 0.001
sched_cooldown = 2

weight_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                            factor=sched_factor,
                                                            patience=sched_patience,
                                                            threshold=sched_trehsh,
                                                            min_lr=0, eps=1e-08,
                                                            cooldown=sched_cooldown,
                                                            verbose = True)

weight_scheduler_fixed = torch.optim.lr_scheduler.StepLR(optim, 120, gamma=0.2)

class DummyOptim:
    def __init__(self):
        self.param_groups = []
    def state_dict(self):
        return {}
    def load_state_dict(self, *args, **kwargs):
        pass
    def step(self, *args, **kwargs):
        pass
    def zero_grad(self):
        pass

feature_net.train()

if c.end_to_end:
    feature_optim = torch.optim.Adam(combined_model.module.feature_network.parameters(), lr=c.lr_feature_net, betas=c.betas, eps=1e-4)
    feature_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(feature_optim,
                                                            factor=sched_factor,
                                                            patience=sched_patience,
                                                            threshold=sched_trehsh,
                                                            min_lr=0, eps=1e-08,
                                                            cooldown=sched_cooldown,
                                                            verbose = True)
else:
    feature_optim = DummyOptim()
    feature_scheduler = DummyOptim()

def optim_step():
    optim.step()
    optim.zero_grad()

    feature_optim.step()
    feature_optim.zero_grad()

# TODO: Może to trzeba przerobić? - na razie jest OK
def save(name):
    torch.save({'opt':optim.state_dict(),
                'opt_f':feature_optim.state_dict(),
                'net':combined_model.state_dict()}, name)

def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    combined_model.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
        feature_optim.load_state_dict(state_dicts['opt_f'])
    except:
        print('Cannot load optimizer for some reason or other')
