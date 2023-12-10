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
from torch.nn.functional import mse_loss

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

def linear_transform(x, center_frac: float=0.1, acc_fact: int=4):
    x = apply_mask(x, center_frac=center_frac, acc_fact=acc_fact)
    return x

def loss_grad(pred, target):
    pred = pred.requires_grad_(requires_grad=True)
    loss = mse_loss(linear_transform(pred), linear_transform(target))
    grads = torch.autograd.grad(loss, pred)[0]
    # print(len(grads), [elem.shape for elem in grads])
    pred = pred.requires_grad_(requires_grad=False)
    return grads

def reconstruction(sampler, samples, guidance_factor=1.0):
    corrupted = linear_transform(samples)
    save_image(make_grid(corrupted, nrow=4, normalize=True), "samples/corrupted_lossguidance.png")

    # init seed 
    t_x = sampler.model.init_noise(samples.shape[0]) * sampler.model.fwd_diff.sqrt_one_minus_alphas_dash[-1]

    for i in tqdm(reversed(range(1, sampler.model.fwd_diff.timesteps))):
        t = i * torch.ones((samples.shape[0]), dtype=torch.long, device=samples.device)

        # t_y, _ = sampler.model.fwd_diff(corrupted, t)
        t_y = corrupted
        t_x = t_x - guidance_factor * loss_grad(t_x, t_y)
        t_x = sampler.model.denoise_singlestep(t_x, t)

    return t_x

def reconstruction_long(sampler, samples, guidance_factor):
    corrupted = linear_transform(samples)
    save_image(make_grid(corrupted, nrow=4, normalize=True), "samples/corrupted_lossguidance.png")

    t_x = sampler.model.init_noise(samples.shape[0]) * sampler.model.fwd_diff.sqrt_one_minus_alphas_dash[-1]
    for j in range(10):
        for i in tqdm(reversed(range(1, sampler.model.fwd_diff.timesteps//(j+1)))):
            t = i * torch.ones((samples.shape[0]), dtype=torch.long, device=samples.device)
            t_y = corrupted
            t_x = t_x - guidance_factor * loss_grad(t_x, t_y)
            t_x = sampler.model.denoise_singlestep(t_x, t)
        if j!=9:
            t = sampler.model.fwd_diff.timesteps//(j+2) * torch.ones((samples.shape[0]), dtype=torch.long, device=samples.device)
            t_x, _ = sampler.model.fwd_diff(t_x, t)
    return t_x

if __name__ == "__main__":
    device = torch.device("cuda")
    sampler = setup_sampler("/itet-stor/peerli/net_scratch/dutiful-pond-10/checkpoint60.pt").to(device)
    samples = get_samples(16).to(device)

    # guidance_factors = [(i+10)**2 for i in range(50)]

    # for factor in guidance_factors:
    factor = 500
    # res = reconstruction(sampler, samples, guidance_factor=factor)
    res = reconstruction_long(sampler, samples, guidance_factor=factor)
    save_image(make_grid(res, nrow=4, normalize=True), f"samples/reconstructed_lossguidanceLONG_factor{factor:.2f}.png")