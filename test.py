import Models.UNET.unet_models as m
from torchinfo import summary
import torch

model = m.custom_UNet(1)
batch_size = 5
summary(model, input_size=(batch_size, 1, 512, 512))


torch.save(model, "/notebooks/mel-steg-cINN/model.pt")