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
from models.diffusion import ForwardDiffusion
from torchvision.transforms import Compose, Normalize, Resize
import torchvision

mode = "linear"
timesteps = 1200
every = 200

#############################################################################

img = "/Users/lionelpeer/Pictures/2020/Japan/darktable_exported/DSC_1808.jpg"
img = read_image(img) / 255
transform = transform = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), Resize((120, 180))])
img = transform(img)

print("image shape:", img.shape)

batch = torch.stack([img, img], dim=0)

print("batch shape:", batch.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("using device:", device)

noiser = ForwardDiffusion(timesteps=timesteps, type=mode).to(device)
batch = batch.to(device)

noisies, noise = noiser.forward(batch[0], torch.tensor([i*every for i in range(0,timesteps//every)]))
noisies = [noisies[i].permute(1,2,0) for i in range(noisies.shape[0])]

blub = [r"$x_{} \sim q(x_{})$".format("{"+str(0)+"}", "{"+str(0)+"}")]
titles = [
    r"$x_{} \sim q(x_{}\mid x_{})$".format("{"+str(i)+"}", "{"+str(i)+"}", "{"+str(i-1)+"}") for i in [j*every for j in range(1, timesteps//every+1)]
]
blub.extend(titles)

fig, ax = plt.subplots(1,timesteps//every,figsize=(25,5))
for i, (elem, title) in enumerate(zip(noisies[:], blub[:])):
    ax[i].imshow(elem.cpu())
    ax[i].axis("off")
    ax[i].set_title(title)
fig.show()

fig.savefig(f"img/forward_naoshima_{mode}.png")