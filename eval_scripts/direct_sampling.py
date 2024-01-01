from sampling_utils import setup_sampler, get_samples, get_kspace_mask, apply_mask
from torch.nn.functional import mse_loss
import torch
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import numpy as np
from h5_utils import write_results_lossguidance, load_corrupted_kspace_and_mask

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

if __name__ == "__main__":
    device = torch.device("cuda")
    sampler = setup_sampler("/itet-stor/peerli/net_scratch/dutiful-pond-10/checkpoint60.pt").to(device)

    factors = [1000*i for i in [1, 2, 3, 5, 10, 20, 30, 50, 100, 200]]

    debug = False

    for index in [0,1,2]:
        kspace, mask = load_corrupted_kspace_and_mask(index, debug=debug)
        kspace, mask = kspace.to(device), mask.to(device)
        for factor in factors:
            res, losses = reconstruction(sampler, kspace, guidance_factor=factor, mask=mask)
            write_results_lossguidance(index, res, "direct", factor, torch.tensor(losses), debug=debug)