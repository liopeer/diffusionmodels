from jaxtyping import Float
from torch import Tensor
from torch.fft import ifftn, fftn, fftshift

filename = "/home/peerli/net_scratch/shortRes_samples.h5"

def to_imgspace(kspace: Float[Tensor, "batch 2 height width"]) -> Float[Tensor, "batch 1 height width"]:
    kspace = torch.view_as_complex(kspace.permute(0,2,3,1).contiguous())
    img = ifftn(kspace, norm="ortho", dim=(1,2))
    img = torch.view_as_real(img).permute(0,3,1,2)
    return torch.norm(img, dim=1, keepdim=True)

def to_kspace(img: Float[Tensor, "batch 1 height width"]) -> Float[Tensor, "batch 2 height width"]:
    img = img.squeeze(1)
    kspace = fftshift(fftn(img, norm="ortho", dim=(1,2)), dim=(1,2)) # batch height width
    kspace = torch.view_as_real(kspace).permute(0,3,1,2)
    return kspace

import h5py
import torch
import context
from utils.datasets import QuarterFastMRI, FastMRIBrainTrain
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from sampling_utils import get_kspace_mask

ds = FastMRIBrainTrain("/itet-stor/peerli/bmicdatasets-originals/Originals/fastMRI/brain/multicoil_train")
samples = [ds.__getitem__(i*600)[0] for i in range(100)]
print(99*600, ds.__len__())
samples = torch.stack(samples)

f = h5py.File(filename, "w")
f["samples"] = samples

center_fracs = [0.1, 0.07, 0.04]
masks = []
accs = [4, 8, 16]
eff_accs = []
for center_frac, acc in zip(center_fracs, accs):
    mask = get_kspace_mask((128,128), center_frac, acc)
    masks.append(mask.unsqueeze(0))
    eff_accs.append(1 / torch.mean(mask.view(-1)))

corr_samples = []
for mask in masks:
    kspace = to_kspace(samples)
    kspace = kspace * mask
    corr_samples.append(kspace)

for i in range(3):
    f.create_group(f"mask{i}")
    f[f"mask{i}"]["mask"] = masks[i]
    f[f"mask{i}"]["center_fraction"] = torch.tensor(center_fracs[i])
    f[f"mask{i}"]["acceleration"] = torch.tensor(accs[i])
    f[f"mask{i}"]["effective_acceleration"] = torch.tensor(eff_accs[i])
    f[f"mask{i}"]["corrupted_kspace"] = corr_samples[i]

f.close()