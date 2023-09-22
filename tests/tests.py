import context
from utils.datasets import UnconditionedCifar10Dataset
from torch.utils.data import DataLoader
from models.diffusion import DiffusionModel, ForwardDiffusion
from models.unet import UNet
import torch
import torchvision
import os
import wandb

ds = UnconditionedCifar10Dataset("/itet-stor/peerli/net_scratch")
dl = DataLoader(ds, batch_size=10)

k = next(iter(dl))

device = torch.device("cpu")

model_path = "/itet-stor/peerli/net_scratch/cifar10_checkpoints/checkpoint90.pt"
model = DiffusionModel(UNet(4), ForwardDiffusion(1000))
model = model.to(device)

wandb.init(project="dummy")

#samples = model.sample(25, 32)
samples = torch.randn((25, 32, 32))
samples = torchvision.utils.make_grid(samples, nrow=5)
print(samples.shape)
images = wandb.Image(
    samples, 
    caption="dummy caption"
)
wandb.log({"examples": images})
path = os.path.join(".", f"samples_epoch{0}.png")
torchvision.utils.save_image(samples, path)
print(f"Epoch {0} | Samples saved at {path}")