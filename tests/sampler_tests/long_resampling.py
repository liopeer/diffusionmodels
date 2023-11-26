from sampler_dutifulpond10 import setup_sampler, get_samples, get_kspace_mask
import torch
import torchvision
from torchvision.utils import save_image, make_grid
from torch.fft import fftn, ifftn, fftshift
import numpy as np
from itertools import permutations

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
    
    # save corrupted images
    corrupted = torch.view_as_complex(kspace.permute(0,2,3,1).contiguous())
    corrupted = torch.view_as_real(ifftn(corrupted, dim=(1,2), norm="ortho")).permute(0,3,1,2)
    corrupted = torch.norm(corrupted, dim=1, keepdim=True)
    corrupted = torchvision.utils.make_grid(corrupted, nrow=int(np.sqrt(samples.shape[0])))
    save_image(corrupted, "samples/samples_dutifulpond10_corrupted.png")

    jump_lengths = [10, 20, 40, 50, 100, 125, 200, 250]
    resamplings = [1, 2, 4, 6, 8, 10, 15, 30]

    unique_combs = []
    for r in resamplings:
        for j in jump_lengths:
            unique_combs.append((r,j))

    # run inference
    for comb in unique_combs:
        out = sampler.masked_sampling_with_resampling_kspace(kspace, mask, *comb)
        out = torchvision.utils.make_grid(out, nrow=int(np.sqrt(samples.shape[0])))
        save_image(out, f"samples/samples_dutifulpond10_resampledkspace_{comb[0]}res_{comb[1]}jump.png")

if __name__ == "__main__":
    device = torch.device("cuda")
    sampler = setup_sampler("/itet-stor/peerli/net_scratch/dutiful-pond-10/checkpoint60.pt").to(device)
    samples = get_samples(16).to(device)

    masked_kspace_resampling(sampler, samples, 4, 0.2)