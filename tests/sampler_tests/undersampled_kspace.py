from sampler_dutifulpond10 import setup_sampler, get_samples, get_kspace_mask
import torch
import torchvision
from torchvision.utils import save_image, make_grid
from torch.fft import fftn, ifftn, fftshift
import numpy as np
from itertools import permutations
from tqdm import tqdm
from torchvision.transforms import Resize, Compose
from typing import Tuple
import random

def _2d_gaussian(size: int = 128, normalized_sigma: float = 1):
    x, y = torch.arange(-size//2, size//2), torch.arange(-size//2, size//2)
    x, y = torch.meshgrid([x,y])
    sigma = normalized_sigma*size/2
    filter = 1/(2*torch.pi*sigma**2) * torch.exp(- ((x**2)+(y**2))/(2*sigma**2))
    return filter

def gaussian_filter(x, std: int=0.1):
    filter = _2d_gaussian(size=x.shape[-1], normalized_sigma=std).to(x.device)
    x = sampler._to_kspace(x) * filter.unsqueeze(0).unsqueeze(0)
    return sampler._to_imgspace(x)

def get_kspace_mask(img_res: Tuple[int], center_frac: float, acc_fact: int):
    img_size = img_res
    offset = img_size[1]*center_frac//2
    middle = img_size[1]//2
    mask = torch.zeros(img_size, dtype=torch.float32)

    # create middle strip
    mask[:, int(middle-offset):int(middle+offset)] = 1

    # create random sampling
    remaining = [i for i in range(int(middle-offset))]
    remaining2 = [i for i in range(int(middle+offset),img_size[1])]
    remaining.extend(remaining2)

    idx = random.sample(remaining, len(remaining)//acc_fact)
    mask[:, torch.tensor(idx)] = 1
    return mask

def apply_mask(x, center_frac, acc_fact):
    mask = get_kspace_mask(x.shape[-2:], center_frac=center_frac, acc_fact=acc_fact).to(x.device)
    x = sampler._to_kspace(x) * mask.unsqueeze(0).unsqueeze(0)
    return sampler._to_imgspace(x)

def linear_transform(x, center_frac: float=0.2, acc_fact: int=4, std: float=3.0, only_masking: bool=False):
    x = apply_mask(x, center_frac=center_frac, acc_fact=acc_fact)
    if only_masking:
        return x
    x = gaussian_filter(x, std=std)
    return x

def reconstruction(sampler, samples, guidance_factor=10e2):
    corrupted = linear_transform(samples, only_masking=True)
    save_image(make_grid(corrupted, nrow=4, normalize=True), "corrupted.png")
    corrupted = linear_transform(samples)
    save_image(make_grid(corrupted, nrow=4, normalize=True), "info_passed.png")

    # init seed 
    t_x = sampler.model.init_noise(samples.shape[0]) * sampler.model.fwd_diff.sqrt_one_minus_alphas_dash[-1]

    for i in tqdm(reversed(range(1, sampler.model.fwd_diff.timesteps))):
        t = i * torch.ones((samples.shape[0]), dtype=torch.long, device=samples.device)

        sqrt_alpha_dash = sampler.model.fwd_diff.sqrt_alphas_dash[i]
        one_minus_sqrt_alpha_dash = sampler.model.fwd_diff.sqrt_one_minus_alphas_dash[i]

        noise = torch.randn_like(samples, device=samples.device)
        t_x = guidance_factor * (sqrt_alpha_dash * corrupted + one_minus_sqrt_alpha_dash * linear_transform(noise) - linear_transform(t_x)) + t_x

        t_x = sampler.model.denoise_singlestep(t_x, t)

    return t_x

if __name__ == "__main__":
    device = torch.device("cuda")
    sampler = setup_sampler("/itet-stor/peerli/net_scratch/dutiful-pond-10/checkpoint60.pt").to(device)
    samples = get_samples(16).to(device)

    guidance_factors = [(i+100)**1.5 for i in range(50)]

    res = reconstruction(sampler, samples)
    save_image(make_grid(res, nrow=4, normalize=True), "reconstructed.png")