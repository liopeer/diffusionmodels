from sampling_utils import setup_sampler, get_samples, get_kspace_mask, apply_mask
from torch.nn.functional import mse_loss
import torch
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import numpy as np
from h5_utils import write_results_lossguidance, load_corrupted_kspace_and_mask
from jaxtyping import Float, Int32
from torch import Tensor
from typing import Literal

def freq_replacement(
        sampler, 
        pred_img: Float[Tensor, "batch channel height width"], 
        target_kspace: Float[Tensor, "batch channel height width"], 
        mask: Float[Tensor, "1 1 height width"], 
        t: Int32[Tensor, "batch"], 
        noising: Literal["kspace", "imgspace"]="kspace"
    ):
    pred_kspace = sampler._to_kspace(pred_img)
    if noising == "kspace":
        noise = torch.randn_like(target_kspace)
        target_kspace = target_kspace * sampler.model.fwd_diff.sqrt_alphas_dash[t].view(-1,1,1,1) + noise * torch.sqrt((sampler.model.fwd_diff.sqrt_one_minus_alphas_dash[t].view(-1,1,1,1)**2 / 2))
        # target_kspace = target_kspace * sampler.model.fwd_diff.sqrt_alphas_dash[t].view(-1,1,1,1) + noise * sampler.model.fwd_diff.sqrt_one_minus_alphas_dash[t].view(-1,1,1,1) / 4
        pred_kspace = pred_kspace + target_kspace * mask - pred_kspace * mask
        return sampler._to_imgspace(pred_kspace)
    elif noising == "imgspace":
        target_img = sampler._to_imgspace(target_kspace)
        target_img, _ = sampler.model.fwd_diff(target_img, t)
        target_kspace = sampler._to_kspace(target_img)
        pred_img = pred_img + sampler._to_imgspace(target_kspace * mask) - sampler._to_imgspace(pred_kspace * mask)
        return pred_img
    else:
        raise ValueError("no such noising process")

def reconstruction(sampler, corrupted_kspace, mask, process=Literal["kspace","imgspace"]):
    t_x = sampler.model.init_noise(corrupted_kspace.shape[0]) * sampler.model.fwd_diff.sqrt_one_minus_alphas_dash[-1]
    for i in tqdm(reversed(range(1, sampler.model.fwd_diff.timesteps))):
        t = i * torch.ones((corrupted_kspace.shape[0]), dtype=torch.long, device=corrupted_kspace.device)
        t_x = freq_replacement(sampler, t_x, corrupted_kspace, mask, t, noising=process)
        t_x = sampler.model.denoise_singlestep(t_x, t)
    return t_x

def reconstruction2(sampler, corrupted_kspace, mask, process=Literal["kspace","imgspace"]):
    t_x = sampler.model.init_noise(corrupted_kspace.shape[0]) * sampler.model.fwd_diff.sqrt_one_minus_alphas_dash[-1]
    starts = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
    for start in starts:
        for i in tqdm(reversed(range(1, start))):
            t = i * torch.ones((corrupted_kspace.shape[0]), dtype=torch.long, device=corrupted_kspace.device)
            t_x = freq_replacement(sampler, t_x, corrupted_kspace, mask, t, noising=process)
            t_x = sampler.model.denoise_singlestep(t_x, t)
        if start != 100:
            t_x, _ = sampler.model.fwd_diff(t_x, (start-100)*torch.ones((corrupted_kspace.shape[0]), dtype=torch.long, device=corrupted_kspace.device))
    return t_x

if __name__ == "__main__":
    device = torch.device("cuda")
    sampler = setup_sampler("/itet-stor/peerli/net_scratch/dutiful-pond-10/checkpoint60.pt").to(device)

    mask = get_kspace_mask((128,128), 0.1, 6).unsqueeze(1)
    samples = get_samples(16)
    kspace = sampler._to_kspace(samples)
    kspace = kspace * mask

    processes = ["imgspace"]
    for process in processes:
        kspace, mask = kspace.to(device), mask.to(device)
        res = reconstruction(sampler, kspace, mask, process)
        save_image(make_grid(res, nrow=4), f"reconstruction_{process}.png")

    save_image(make_grid(samples, nrow=4), "samples.png")
    save_image(make_grid(sampler._to_imgspace(kspace), nrow=4), "corrupted_samples.png")