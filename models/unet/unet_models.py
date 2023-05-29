""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch.nn as nn


class Up1(nn.Module):
    def __init__(self):
        super(Up1, self).__init__()
        self.up1 = Up.from_residual_half_channels(64, 32)

    def forward(self, x3, x4):
        return self.up1(x4, x3)


class Up2(nn.Module):
    def __init__(self):
        super(Up2, self).__init__()
        self.up1 = Up1()
        self.up2 = Up.from_residual_half_channels(32, 16)

    def forward(self, x2, x3, x4):
        x = self.up1(x3, x4)
        return self.up2(x, x2)


class Up3(nn.Module):
    def __init__(self):
        super(Up3, self).__init__()
        self.up2 = Up2()
        self.up3 = Up.from_residual_half_channels(16, 8)

    def forward(self, x1, x2, x3, x4):
        x = self.up2(x2, x3, x4)
        return self.up3(x, x1)


class UNet(nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.up1 = Up.from_residual_half_channels(64, 32)
        self.up2 = Up.from_residual_half_channels(32, 16)
        self.up3 = Up.from_residual_half_channels(16, 8)
        self.out = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 2))

    def forward(self, x):
        _, x = self.features(x)
        return self.forward_from_features(x)

    def features(self, x):
        features = []
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        features.insert(0, x4)  # level_3

        x_up1 = self.up1(x4, x3)
        features.insert(0, x_up1)  # level_2

        x_up2 = self.up2(x_up1, x2)
        features.insert(0, x_up2)  # level_1

        x_up3 = self.up3(x_up2, x1)
        features.insert(0, x_up3)  # level_0

        return features, x_up3

    def forward_from_features(self, x):
        x = x.permute((0, 2, 3, 1))
        x = self.out(x)
        ab = x.permute((0, 3, 1, 2))
        return ab


class custom_UNet(nn.Module):
    def __init__(self, n_channels):
        super(custom_UNet, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 32)
        self.down2 = Down(32, 32)
        self.down3 = Down(32, 64)
        self.down4 = Down(64, 64)
        # 128x16x16_
        self.down5 = Down(64, 128)
        self.up1 = Up.from_custom_residual_channels(128, 64, 64)
        self.up2 = Up.from_custom_residual_channels(64, 64, 64)
        self.up3 = Up.from_custom_residual_channels(64, 32, 32)
        self.up4 = Up.from_custom_residual_channels(32, 32, 32)
        self.up5 = Up.from_custom_residual_channels(32, 32, 32)
        self.out = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 2))

    def features(self, x):
        """
            Features for cINN.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        return x1, x2, x3, x4, x5, x6

    def forward_from_features(self, x1, x2, x3, x4, x5, x6):
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = x.permute((0, 2, 3, 1))
        x = self.out(x)
        ab = x.permute((0, 3, 1, 2))

        return ab

    def forward(self, x):
        x1, x2, x3, x4, x5, x6 = self.features(x)
        ab = self.forward_from_features(x1, x2, x3, x4, x5, x6)

        return ab


class UNet_32(nn.Module):
    def __init__(self, n_channels):
        super(UNet_32, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        # 256x64x64
        self.down3 = Down(128, 256)
        self.up1 = Up.from_residual_half_channels(256, 128)
        self.up2 = Up.from_residual_half_channels(128, 64)
        self.up3 = Up.from_residual_half_channels(64, 32)
        self.out = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 2))

    def features(self, x):
        """
            Features for cINN.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        features = x

        return features, x

    def forward_from_features(self, x):
        x = x.permute((0, 2, 3, 1))
        x = self.out(x)
        ab = x.permute((0, 3, 1, 2))

        return ab

    def forward(self, x):
        _, from_features = self.features(x)
        ab = self.forward_from_features(*from_features)
        return ab


class UNet_256(nn.Module):
    def __init__(self, n_channels):
        super(UNet_256, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        # 256x64x64
        self.down3 = Down(128, 256)
        self.up1 = Up.from_residual_half_channels(256, 128)
        self.up2 = Up.from_residual_half_channels(128, 64)
        self.up3 = Up.from_residual_half_channels(64, 32)
        self.out = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 2))

    def features(self, x):
        """
            Features for cINN.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        features = x4

        return features, (x1, x2, x3, x4)

    def forward_from_features(self, x1, x2, x3, x4):
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = x.permute((0, 2, 3, 1))
        x = self.out(x)
        ab = x.permute((0, 3, 1, 2))

        return ab

    def forward(self, x):
        _, from_features = self.features(x)
        ab = self.forward_from_features(*from_features)
        return ab


class UNet_128(nn.Module):
    def __init__(self, n_channels):
        super(UNet_128, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        # 256x64x64
        self.down3 = Down(128, 256)
        self.up1 = Up.from_residual_half_channels(256, 128)
        self.up2 = Up.from_residual_half_channels(128, 64)
        self.up3 = Up.from_residual_half_channels(64, 32)
        self.out = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 2))

    def features(self, x):
        """
            Features for cINN.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)

        features = x

        return features, (x, x1, x2)

    def forward_from_features(self, _x, x1, x2):
        x = self.up2(_x, x2)
        x = self.up3(x, x1)
        x = x.permute((0, 2, 3, 1))
        x = self.out(x)
        ab = x.permute((0, 3, 1, 2))

        return ab

    def forward(self, x):
        _, from_features = self.features(x)
        ab = self.forward_from_features(*from_features)
        return ab
