""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        
        self.inc = DoubleConv(n_channels, 4)
        self.down1 = Down(4, 8)
        self.down2 = Down(8, 16)
        self.down3 = Down(16, 32)
        self.down4 = Down(32, 64)
        # 128x16x16
        self.down5 = Down(64, 128)
        self.up1 = Up.from_residual_half_channels(128, 64)
        self.up2 = Up.from_residual_half_channels(64, 32)
        self.up3 = Up.from_residual_half_channels(32, 16)
        self.up4 = Up.from_residual_half_channels(16, 8)
        self.up5 = Up.from_residual_half_channels(8, 4)
        self.out = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = x.permute((0,2,3,1))
        x = self.out(x)
        ab = x.permute((0,3,1,2))
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
        # 128x16x16
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
        x = x.permute((0,2,3,1))
        x = self.out(x)
        ab = x.permute((0,3,1,2))

        return ab

    def forward(self, x):

        x1, x2, x3, x4, x5, x6 = self.features(x)
        ab = self.forward_from_features(x1, x2, x3, x4, x5, x6 )
        
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
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return x

    def forward_from_features(self, x):        
        x = x.permute((0,2,3,1))
        x = self.out(x)
        ab = x.permute((0,3,1,2))

        return ab

    def forward(self, x):
        x = self.features(x)
        ab = self.forward_from_features(x)
        return ab


class UNet_256_2(nn.Module):
    def __init__(self, n_channels):
        super(UNet_256_2, self).__init__()
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

        return x1, x2, x3, x4

    def forward_from_features(self, x1, x2, x3, x4):
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = x.permute((0,2,3,1))
        x = self.out(x)
        ab = x.permute((0,3,1,2))

        return ab

    def forward(self, x):
        x1, x2, x3, x4 = self.features(x)
        ab = self.forward_from_features(x1, x2, x3, x4)
        return ab



