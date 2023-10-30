import context
from torch.utils.data import DataLoader
from models.diffusion import DiffusionModel, ForwardDiffusion
from models.unet import UNet
import torch
import torchvision
import os
import wandb

device = torch.device("cpu")

batch = torch.randn((16, 3, 32, 32))

model = UNet(4, attention=True)

out = model(batch)
print(out.shape)