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

def reconstruction(sampler, corrupted_samples, guidance_factor, mask, resample_every: int):
    assert sampler.model.fwd_diff.timesteps % resample_every == 0

    t_x = sampler.model.init_noise(corrupted_samples.shape[0]) * sampler.model.fwd_diff.sqrt_one_minus_alphas_dash[-1]
    losses = []
    starts = [sampler.model.fwd_diff.timesteps-i*resample_every for i in range(sampler.model.fwd_diff.timesteps//resample_every)]
    print(starts)
    for j,start in enumerate(starts):
        for i in tqdm(reversed(range(1, start))):
            t = i * torch.ones((corrupted_samples.shape[0]), dtype=torch.long, device=corrupted_samples.device)
            t_y = corrupted_samples
            grads, loss = mse_grad(sampler, t_x, t_y, mask)
            losses.append(loss)
            t_x = t_x - guidance_factor * grads
            t_x = sampler.model.denoise_singlestep(t_x, t)
        if start != starts[-1]:
            t = starts[j+1] * torch.ones((corrupted_samples.shape[0]), dtype=torch.long, device=corrupted_samples.device)
            sampler.model.fwd_diff(t_x, t)
    return t_x, losses

if __name__ == "__main__":
    device = torch.device("cuda")
    sampler = setup_sampler("/itet-stor/peerli/net_scratch/dutiful-pond-10/checkpoint60.pt").to(device)

    guidance_factors = [1000*i for i in [1, 2, 3, 5, 10, 20]]
    jump_lengths = [100, 200, 500]
    jump_lengths = list(reversed(jump_lengths))

    debug = False

    for index in [2]:
        samples, mask = load_corrupted_kspace_and_mask(index, debug=debug)
        samples, mask = samples.to(device), mask.to(device)
        for guidance in guidance_factors:
            for jump_length in jump_lengths:
                res, losses = reconstruction(sampler, samples, guidance_factor=guidance, mask=mask, resample_every=jump_length)
                # save_image(make_grid(res, nrow=10), "recon.png")
                write_results_lossguidance(index, res, "globalResampling", guidance, torch.tensor(losses), jump_length=jump_length, debug=debug)