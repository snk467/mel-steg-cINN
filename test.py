import Models.UNET.unet_models as m
from torchinfo import summary
import torch


import colorization_cINN.model as cinn_model

# model = m.UNet_256(1)
# batch_size = 8
# summary(model, input_size=(batch_size, 1, 512, 512))


batch_size = 8
summary(cinn_model.combined_model, input_size=(batch_size, 3, 512, 512))

# torch.save(model, "/notebooks/mel-steg-cINN/model.pt")