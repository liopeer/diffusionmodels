import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
import torch
from torchvision.io import read_image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import context
from diffusion_models.models.forward_diffusion import ForwardDiffusion

img = "/home/pel1yh/Pictures/output7.png"
img2 = "/home/pel1yh/Pictures/output8.png"
img = read_image(img) / 255
img2 = read_image(img2) / 255

print("image shape:", img.shape)

batch = torch.stack([img, img], dim=0)

print("batch shape:", batch.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("using device:", device)

noiser = ForwardDiffusion(timesteps=5000).to(device)
batch = batch.to(device)

noisies = [noiser.forward(batch[0], i*500).permute(1,2,0) for i in range(7)]

blub = [r"$x_{} \sim q(x_{})$".format("{"+str(0)+"}", "{"+str(0)+"}")]
titles = [
    r"$x_{} \sim q(x_{}\mid x_{})$".format("{"+str(i)+"}", "{"+str(i)+"}", "{"+str(i-1)+"}") for i in [j*500 for j in range(1, 7)]
]
blub.extend(titles)

fig, ax = plt.subplots(1,6,figsize=(25,5))
for i, (elem, title) in enumerate(zip(noisies[:-1], blub[:-1])):
    ax[i].imshow(elem.cpu())
    ax[i].axis("off")
    ax[i].set_title(title)
fig.show()

fig.savefig("img/forward_naoshima.png")