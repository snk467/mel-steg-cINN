import Models.UNET.unet_model as m
from torchinfo import summary

model = m.UNet(1)
batch_size = 5
summary(model, input_size=(batch_size, 1, 128, 128))