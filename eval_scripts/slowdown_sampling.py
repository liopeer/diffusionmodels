from sampling_utils import setup_sampler, get_samples, get_kspace_mask, apply_mask
from torch.nn.functional import mse_loss
import torch
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import numpy as np
from h5_utils import write_results_lossguidance, load_corrupted_kspace_and_mask
import context
from models.positional_encoding import PositionalEncoding
import math

def mse_grad(sampler, pred, corrupted_kspace, mask):
    pred = pred.requires_grad_(requires_grad=True)

    pred2 = sampler._to_kspace(pred) * mask
    loss = mse_loss(pred2, corrupted_kspace)

    # loss scaling for different masks
    eff_acc = torch.mean(mask.view(-1))
    loss = loss / eff_acc

    grads = torch.autograd.grad(loss, pred)[0]
    pred = pred.requires_grad_(requires_grad=False)

    return grads, loss.item()

def reconstruction(sampler, corrupted_kspace, guidance_factor, mask):
    t_x = sampler.model.init_noise(corrupted_kspace.shape[0]) * sampler.model.fwd_diff.sqrt_one_minus_alphas_dash[-1]
    losses = []
    for i in tqdm(reversed(range(1, sampler.model.fwd_diff.timesteps))):
        t = i * torch.ones((corrupted_kspace.shape[0]), dtype=torch.long, device=corrupted_kspace.device)
        t_y = corrupted_kspace
        grads, loss = mse_grad(sampler, t_x, t_y, mask)
        losses.append(loss)
        t_x = t_x - guidance_factor * grads
        t_x = sampler.model.denoise_singlestep(t_x, t)
    return t_x, losses

def update_positional_encoding(timesteps):
    d_model = 512
    original_timesteps = 1000
    new_timesteps = timesteps

    position = torch.linspace(0, original_timesteps, new_timesteps).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(new_timesteps, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)

    return pe

if __name__ == "__main__":
    # slowdown_factors = [5,3,2,1.5,1.0,0.75,0.5]
    slowdown_factors = [5,3,2,1.5]
    slowdown_factors = list(reversed(slowdown_factors))

    timesteps = [int(1000*elem) for elem in slowdown_factors]

    debug = False

    for index in [2]:
        for slowdown, timestep in zip(slowdown_factors,timesteps):
            guidance_factors = [int(1000*i//slowdown) for i in [1, 2, 3, 5, 10, 20, 30, 50, 100]]

            device = torch.device("cuda")
            sampler = setup_sampler("/itet-stor/peerli/net_scratch/dutiful-pond-10/checkpoint60.pt")
            sampler.model.fwd_diff.__init__(timesteps=timestep, type="cosine")
            sampler.model.time_encoder.pe = update_positional_encoding(timestep).to(device)
            sampler = sampler.to(device)

            kspace, mask = load_corrupted_kspace_and_mask(index, debug=debug)
            # save_image(make_grid(sampler._to_imgspace(kspace), nrow=10), "corrupted_kspace.jpg")
            kspace, mask = kspace.to(device), mask.to(device)
            for factor in guidance_factors:
                res, losses = reconstruction(sampler, kspace, guidance_factor=factor, mask=mask)
                # save_image(make_grid(res, nrow=10), "recon.jpg")
                write_results_lossguidance(index, res, "slowingDown", factor, torch.tensor(losses), slowdown_factor=slowdown, debug=debug)