"""Incomplete."""
import context
import torch
from diffusion_models.models.vae import VariationalAutoencoder, ResNet18Encoder, ResNetDecoderBlock

model = VariationalAutoencoder(3, 256)

sample = torch.randn((16, 3, 224, 224))

x = model(sample)

print(x.shape)