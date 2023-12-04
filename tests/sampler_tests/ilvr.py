from sampler_dutifulpond10 import setup_sampler, get_samples, get_kspace_mask
import torch
import torchvision
from torchvision.utils import save_image, make_grid
from torch.fft import fftn, ifftn, fftshift
import numpy as np
from itertools import permutations
from tqdm import tqdm
from torchvision.transforms import Resize, Compose

def _2d_gaussian(size: int = 128, normalized_sigma: float = 1):
    x, y = torch.arange(-size//2, size//2), torch.arange(-size//2, size//2)
    x, y = torch.meshgrid([x,y])
    sigma = normalized_sigma*size/2
    filter = 1/(2*torch.pi*sigma**2) * torch.exp(- ((x**2)+(y**2))/(2*sigma**2))
    return filter

def standard_ilvr(sampler, samples):
    """Samples need to be originals, filtering happens in latent space"""
    filter = Compose([Resize((16,16), antialias=True), Resize((128,128), antialias=True)])
    downsampled = filter(samples)
    save_image(make_grid(downsampled, nrow=4), "downsampled_ilvr.png")

    # init seed 
    t_x = sampler.model.init_noise(samples.shape[0]) * sampler.model.fwd_diff.sqrt_one_minus_alphas_dash[-1]

    for i in tqdm(reversed(range(1, sampler.model.fwd_diff.timesteps))):
        t = i * torch.ones((samples.shape[0]), dtype=torch.long, device=samples.device)
        img_t_y, _ = sampler.model.fwd_diff(samples, t)
        t_x = filter(img_t_y) + t_x - filter(t_x)
        t_x = sampler.model.denoise_singlestep(t_x, t)

    return t_x

def gaussian_filter(x, std: int=0.1):
    filter = _2d_gaussian(normalized_sigma=std).to(x.device)
    x = sampler._to_kspace(x) * filter.unsqueeze(0).unsqueeze(0)
    return sampler._to_imgspace(x)

def gaussian_ilvr(sampler, samples):
    downsampled = gaussian_filter(samples)
    save_image(make_grid(downsampled, nrow=4, normalize=True), "filtered_ilvr_gaussian.png")

    # init seed 
    t_x = sampler.model.init_noise(samples.shape[0]) * sampler.model.fwd_diff.sqrt_one_minus_alphas_dash[-1]

    for i in tqdm(reversed(range(1, sampler.model.fwd_diff.timesteps))):
        t = i * torch.ones((samples.shape[0]), dtype=torch.long, device=samples.device)

        sqrt_alpha_dash = sampler.model.fwd_diff.sqrt_alphas_dash[i]
        one_minus_sqrt_alpha_dash = sampler.model.fwd_diff.sqrt_one_minus_alphas_dash[i]

        t_x = sqrt_alpha_dash * downsampled + one_minus_sqrt_alpha_dash * gaussian_filter(torch.randn_like(samples, device=samples.device)) + t_x - gaussian_filter(t_x)

        t_x = sampler.model.denoise_singlestep(t_x, t)

    return t_x

if __name__ == "__main__":
    device = torch.device("cuda")
    sampler = setup_sampler("/itet-stor/peerli/net_scratch/dutiful-pond-10/checkpoint60.pt").to(device)
    samples = get_samples(16).to(device)

    res = gaussian_ilvr(sampler, samples)
    save_image(make_grid(res, nrow=4), "ilvr_gaussian.png")

    # res = standard_ilvr(sampler, samples)
    # save_image(make_grid(res, nrow=4), "standard_ilvr.png")