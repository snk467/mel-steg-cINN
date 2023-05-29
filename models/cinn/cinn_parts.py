from math import exp
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class F_conv(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=None,
                 stride=None, kernel_size=3, leaky_slope=0.1,
                 batch_norm=False):
        super(F_conv, self).__init__()

        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)
        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.conv1 = nn.Conv2d(in_channels, channels_hidden,
                               kernel_size=kernel_size, padding=pad,
                               bias=not batch_norm)
        self.conv2 = nn.Conv2d(channels_hidden, channels_hidden,
                               kernel_size=kernel_size, padding=pad,
                               bias=not batch_norm)
        self.conv3 = nn.Conv2d(channels_hidden, channels,
                               kernel_size=kernel_size, padding=pad,
                               bias=not batch_norm)

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(channels_hidden)
            self.bn1.weight.data.fill_(1)
            self.bn2 = nn.BatchNorm2d(channels_hidden)
            self.bn2.weight.data.fill_(1)
            self.bn3 = nn.BatchNorm2d(channels)
            self.bn3.weight.data.fill_(1)
        self.batch_norm = batch_norm

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = F.leaky_relu(out, self.leaky_slope)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = F.leaky_relu(out, self.leaky_slope)

        out = self.conv3(out)
        if self.batch_norm:
            out = self.bn3(out)
        return out


class F_fully_connected(nn.Module):
    '''Fully connected tranformation, not reversible, but used below.'''

    def __init__(self, size_in, size, internal_size=None, dropout=0.0):
        super(F_fully_connected, self).__init__()
        if not internal_size:
            internal_size = 2 * size

        self.d1 = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(size_in, internal_size)
        self.fc3 = nn.Linear(internal_size, size)

        self.nl1 = nn.ReLU()

    def forward(self, x):
        out = self.nl1(self.d1(self.fc1(x)))
        out = self.fc3(out)
        return out


class F_fully_convolutional(nn.Module):

    def __init__(self, in_channels, out_channels, internal_size=256, kernel_size=3, leaky_slope=0.02):
        super().__init__()

        pad = kernel_size // 2

        self.leaky_slope = leaky_slope
        self.conv1 = nn.Conv2d(in_channels, internal_size, kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(in_channels + internal_size, internal_size, kernel_size=kernel_size, padding=pad)
        self.conv3 = nn.Conv2d(in_channels + 2 * internal_size, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), self.leaky_slope)
        x2 = F.leaky_relu(self.conv2(torch.cat([x, x1], 1)), self.leaky_slope)
        return self.conv3(torch.cat([x, x1, x2], 1))


class subnet_coupling_layer(nn.Module):
    def __init__(self, dims_in, dims_c, F_class, subnet, sub_len, F_args={}, clamp=5.):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.conditional = True
        condition_length = sub_len
        self.subnet = subnet

        self.s1 = F_class(self.split_len1 + condition_length, self.split_len2 * 2, **F_args)
        self.s2 = F_class(self.split_len2 + condition_length, self.split_len1 * 2, **F_args)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, c=[], rev=False, jac=True):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))
        c_star = self.subnet(torch.cat(c, 1))

        if not rev:
            r2 = self.s2(torch.cat([x2, c_star], 1) if self.conditional else x2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(torch.cat([y1, c_star], 1) if self.conditional else y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1
            self.last_jac = self.log_e(s1) + self.log_e(s2)

        else:  # names of x and y are swapped!
            r1 = self.s1(torch.cat([x1, c_star], 1) if self.conditional else x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(torch.cat([y2, c_star], 1) if self.conditional else y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)
            self.last_jac = - self.log_e(s1) - self.log_e(s2)

        return (torch.cat((y1, y2), 1),), self.jacobian(x)

    def jacobian(self, x, c=[], rev=False):
        return torch.sum(self.last_jac, dim=tuple(range(1, self.ndims + 1)))

    def output_dims(self, input_dims):
        return input_dims
