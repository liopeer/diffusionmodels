from sampler_dutifulpond10 import setup_sampler, get_samples, get_kspace_mask
import torch
import torchvision
from torchvision.utils import save_image, make_grid
from torch.fft import fftn, ifftn, fftshift
import numpy as np
from itertools import permutations

def _2d_gaussian(size: int = 128, normalized_sigma: float = 1):
    x, y = torch.arange(-size//2, size//2), torch.arange(-size//2, size//2)
    x, y = torch.meshgrid([x,y])
    sigma = normalized_sigma*size/2
    filter = torch.exp(-(x**2)/(2*sigma**2) - (y**2)/(2*sigma**2))
    return filter / filter.max()

def masked_kspace_resampling(sampler, samples, acceleration_factor, center_frac):
    # create mask
    mask = get_kspace_mask((samples.shape[-2],samples.shape[-1]), center_frac=center_frac, acc_fact=acceleration_factor)
    mask = mask.unsqueeze(0).unsqueeze(0).to(samples.device)
    save_image(mask[0].to(torch.float), "samples/kspace_mask.png")

    # prepare k space
    samples = samples.squeeze(1)
    kspace = fftshift(fftn(samples, norm="ortho", dim=(1,2)), dim=(1,2))
    kspace = torch.view_as_real(kspace).permute(0, 3, 1, 2)
    kspace = kspace * ~mask

    # mask = torch.ones(128, dtype=torch.bool)
    # mask[64-7:64+7] = 0
    # mask = mask.unsqueeze(0).repeat(128, 1)
    # save_image(mask.to(torch.float32), "samples/boxfilter.png")
    # mask = mask.unsqueeze(0).unsqueeze(0).to(kspace.device)
    # kspace = kspace * ~mask
    
    # save corrupted images
    corrupted = torch.view_as_complex(kspace.permute(0,2,3,1).contiguous())
    corrupted = torch.view_as_real(ifftn(corrupted, dim=(1,2), norm="ortho")).permute(0,3,1,2)
    corrupted = torch.norm(corrupted, dim=1, keepdim=True)
    corrupted = torchvision.utils.make_grid(corrupted, nrow=int(np.sqrt(samples.shape[0])))
    save_image(corrupted, "samples/samples_dutifulpond10_corrupted.png")

    # do filtering
    for sigma in [0.5,0.4,0.3,0.2,0.15,0.1,0.08,0.07,0.06,0.05,0.04,0.03,0.02]:
        filter = _2d_gaussian(normalized_sigma=sigma).unsqueeze(0).unsqueeze(0)
        filtered_kspace = kspace * filter.to(kspace.device)

        # run inference
        out = sampler.masked_sampling_kspace(filtered_kspace, mask, gaussian_scheduling=False)
        out = torchvision.utils.make_grid(out, nrow=int(np.sqrt(samples.shape[0])))
        # save_image(out, f"samples/samples_dutifulpond10_sampleGaussian_0.2BoxFilter.png")
        save_image(out, f"samples/samples_dutifulpond10_sampleGaussian_{sigma:.3f}std.png")

if __name__ == "__main__":
    device = torch.device("cuda")
    sampler = setup_sampler("/itet-stor/peerli/net_scratch/dutiful-pond-10/checkpoint60.pt").to(device)
    samples = get_samples(16).to(device)

    masked_kspace_resampling(sampler, samples, 4, 0.2)