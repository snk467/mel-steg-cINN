""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        #TODO: Kończymy 128x16x16
        self.down4 = Down(128, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = Up(32, 16)
        #TODO: Wyjście na warstwy gęste(liniowe)
        self.out = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 2), nn.Sigmoid())
        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = x.permute((0,2,3,1))
        x = self.out(x)
        ab = x.permute((0,3,1,2))
        return ab
    

