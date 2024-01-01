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

def reconstruction(sampler, corrupted_kspace, guidance_factor, mask, jump_length, num_resamplings):
    assert sampler.model.fwd_diff.timesteps % jump_length == 0

    t_x = sampler.model.init_noise(corrupted_kspace.shape[0]) * sampler.model.fwd_diff.sqrt_one_minus_alphas_dash[-1]
    losses = []
    for i in tqdm(reversed(range(1, sampler.model.fwd_diff.timesteps))):
        t = i * torch.ones((corrupted_kspace.shape[0]), dtype=torch.long, device=corrupted_kspace.device)
        grads, loss = mse_grad(sampler, t_x, corrupted_kspace, mask)
        losses.append(loss)
        t_x = t_x - guidance_factor * grads
        t_x = sampler.model.denoise_singlestep(t_x, t)

        if i % jump_length == 0:
            for _ in range(num_resamplings):
                t_x, _ = sampler.model.fwd_diff.forward_flexible(t_x, t-1, t-1+jump_length)
                for j in reversed(range(i, i+jump_length)):
                    tj = j * torch.ones((corrupted_kspace.shape[0]), dtype=torch.long, device=corrupted_kspace.device)
                    grads, loss = mse_grad(sampler, t_x, corrupted_kspace, mask)
                    losses.append(loss)
                    t_x = t_x - guidance_factor * grads
                    t_x = sampler.model.denoise_singlestep(t_x, tj)
    return t_x, losses

if __name__ == "__main__":
    device = torch.device("cuda")
    sampler = setup_sampler("/itet-stor/peerli/net_scratch/dutiful-pond-10/checkpoint60.pt").to(device)

    guidance_factors = [1000*i for i in [10, 30, 100]]
    jump_lengths = [100, 200, 500]
    jump_lengths = list(reversed(jump_lengths))
    num_resamplings = [1, 2, 5, 10]

    debug = False

    for index in [0,1,2]:
        samples, mask = load_corrupted_kspace_and_mask(index, debug=debug)
        samples, mask = samples.to(device), mask.to(device)
        for guidance in guidance_factors:
            for jump_length in jump_lengths:
                for num_res in num_resamplings:
                    res, losses = reconstruction(sampler, samples, guidance_factor=guidance, mask=mask, jump_length=jump_length, num_resamplings=num_res)
                    write_results_lossguidance(index, res, "localResampling", guidance, torch.tensor(losses), jump_length=jump_length, num_resamplings=num_res, debug=debug)