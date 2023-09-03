import context
import torch
from diffusion_models.models.vae import VariationalAutoencoder, ResNet18Encoder, ResNetDecoderBlock

model = ResNet18Encoder(3)

#print(model)

sample = torch.randn((16, 3, 224, 224))

#print(model(sample).shape)

out = model(sample)

model2 = ResNetDecoderBlock(512, 512)
model3 = ResNetDecoderBlock(512, 256)

print(model2(out).shape)
print(model3(out).shape)