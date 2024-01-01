import sys
sys.path.append("../../diffusion_models")
from models.sampler import DiffusionSampler
from models.unet import UNet
from models.diffusion import DiffusionModel, ForwardDiffusion
import torch
import torch.nn as nn
import os
import json
import torchvision
from torchvision.utils import save_image
import os
from torch import Tensor
from jaxtyping import Float, Bool
from utils.datasets import FastMRIBrainTrain
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from torch.fft import fftn, ifftn, fftshift, ifftshift
from typing import Tuple
from random import sample
from torch.nn.functional import mse_loss
from tqdm import tqdm

def setup_sampler(ckpt: str):
    config = None
    with open("config_dutifulpond10.json", "r") as f:
        config = json.load(f)
    ckpt_path = ckpt
    backbone = UNet(
        num_encoding_blocks = config["backbone_enc_depth"]["value"],
        in_channels = config["in_channels"]["value"],
        kernel_size = config["kernel_size"]["value"],
        time_emb_size = config["time_enc_dim"]["value"],
        dropout = 0.0,
        activation = nn.SiLU,
        verbose = False,
        init_channels = config["unet_init_channels"]["value"],
        attention = False
    )
    fwd_diff = ForwardDiffusion(
        timesteps = config["max_timesteps"]["value"],
        type = config["schedule_type"]["value"]
    )
    model = DiffusionModel(
        backbone = backbone,
        fwd_diff = fwd_diff,
        img_size = config["img_size"]["value"],
        time_enc_dim = config["time_enc_dim"]["value"]
    )
    sampler = DiffusionSampler(
        model,
        ckpt_path,
        "cuda",
        mixed_precision = False
    )
    return sampler

def get_samples(num_samples):
    ds = FastMRIBrainTrain("/itet-stor/peerli/bmicdatasets-originals/Originals/fastMRI/brain/multicoil_train", size=128)
    dl = DataLoader(ds, batch_size=num_samples, shuffle=True)

    path = "samples_dutifulpond10.obj"

    if (os.path.exists(path)) and (torch.load(path).shape[0]==num_samples):
        out = torch.load(path)
    else:
        out = next(iter(dl))[0]
        torch.save(out, path)
    samples = torchvision.utils.make_grid(out, nrow=int(np.sqrt(num_samples)))
    save_image(samples, path.split(".")[0] + ".png")
    return out

def masked_sampling(sampler, samples):
    mask = torch.zeros(*samples.shape, dtype=torch.bool, device=samples.device)
    mask[:, :, 0:80, 0:80] = 1
    masked_samples = torchvision.utils.make_grid(samples*~mask, nrow=4)
    save_image(masked_samples, "samples_dutifulpond10_masked.png")
    # out2 = sampler.masked_sampling(samples, mask)
    out2 = sampler.masked_sampling(samples*~mask, mask)
    recon_samples = torchvision.utils.make_grid(out2, nrow=int(np.sqrt(samples.shape[0])))
    save_image(recon_samples, "samples_dutifulpond10_reconstructed.png")

def mse_grad(sampler, pred, corrupted_samples, mask):
    pred = pred.requires_grad_(requires_grad=True)

    pred2 = pred * ~mask
    loss = mse_loss(pred2, corrupted_samples)

    grads = torch.autograd.grad(loss, pred)[0]
    pred = pred.requires_grad_(requires_grad=False)

    return grads, loss.item()

def masked_sampling2(sampler, samples):
    mask = torch.zeros(*samples.shape, dtype=torch.bool, device=samples.device)
    mask[:, :, 0:80, 0:80] = 1
    masked_samples = torchvision.utils.make_grid(samples*~mask, nrow=4)
    save_image(masked_samples, "samples_dutifulpond10_masked.png")
    samples = samples * ~mask
    t_x = sampler.model.init_noise(samples.shape[0]) * sampler.model.fwd_diff.sqrt_one_minus_alphas_dash[-1]
    guidance_factor = 100000
    for i in tqdm(reversed(range(1, sampler.model.fwd_diff.timesteps))):
        t = i * torch.ones((samples.shape[0]), dtype=torch.long, device=samples.device)
        t_y = samples
        grads, loss = mse_grad(sampler, t_x, t_y, mask)
        t_x = t_x - guidance_factor * grads
        t_x = sampler.model.denoise_singlestep(t_x, t)
    save_image(torchvision.utils.make_grid(t_x, nrow=4), "samples_masked_reconloss.png")

def masked_resampling(sampler, samples):
    mask = torch.zeros(*samples.shape, dtype=torch.bool, device=samples.device)
    mask[:, :, 0:80, 0:80] = 1
    masked_samples = torchvision.utils.make_grid(samples*~mask, nrow=4)
    save_image(masked_samples, "samples_dutifulpond10_masked.png")
    # resampled, steps = sampler.masked_sampling_with_resampling(samples, mask, 10, 10, True)
    resampled, steps = sampler.masked_sampling_with_resampling(samples*~mask, mask, 10, 10, True)
    resampled = torchvision.utils.make_grid(resampled, nrow=int(np.sqrt(samples.shape[0])))
    save_image(resampled, "samples_dutifulpond10_resampled.png")

    fig, ax = plt.subplots(1, 3, figsize=(30,10))
    ax[0].plot(steps)
    ax[0].set_title("Complete Schedule")
    ax[1].plot(steps[:300])
    ax[1].set_title("First 300 Steps")
    ax[2].plot(steps[-300:])
    ax[2].set_title("Last 300 Steps")
    fig.savefig("stepsplot.png")

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

    idx = sample(remaining, len(remaining)//acc_fact)
    mask[:, torch.tensor(idx)] = 1
    return (1-mask)

def masked_kspace_sampling(sampler, samples, acceleration_factor, center_frac, gaussian_scheduling):
    # create mask
    mask = get_kspace_mask((samples.shape[-2],samples.shape[-1]), center_frac=center_frac, acc_fact=acceleration_factor)
    mask = mask.unsqueeze(0).unsqueeze(0).to(samples.device)
    save_image(mask[0].to(torch.float), "kspace_mask.png")

    # prepare k space
    samples = samples.squeeze(1)
    kspace = fftshift(fftn(samples, norm="ortho", dim=(1,2)), dim=(1,2))
    kspace = torch.view_as_real(kspace).permute(0, 3, 1, 2)
    # kspace = kspace * (1-mask)
    
    # save corrupted images
    corrupted = torch.view_as_complex(kspace.permute(0,2,3,1).contiguous())
    corrupted = torch.view_as_real(ifftn(corrupted, dim=(1,2), norm="ortho")).permute(0,3,1,2)
    corrupted = torch.norm(corrupted, dim=1, keepdim=True)
    corrupted = torchvision.utils.make_grid(corrupted, nrow=int(np.sqrt(samples.shape[0])))
    save_image(corrupted, "samples_dutifulpond10_corrupted.png")

    # run inference
    out = sampler.masked_sampling_kspace(kspace, mask, gaussian_scheduling=gaussian_scheduling)
    out = torchvision.utils.make_grid(out, nrow=int(np.sqrt(samples.shape[0])))
    save_image(out, "samples_dutifulpond10_reconstructedkspace.png")

def masked_kspace_resampling(sampler, samples, acceleration_factor, center_frac, gaussian_scheduling):
    # create mask
    mask = get_kspace_mask((samples.shape[-2],samples.shape[-1]), center_frac=center_frac, acc_fact=acceleration_factor)
    mask = mask.unsqueeze(0).unsqueeze(0).to(samples.device)
    save_image(mask[0].to(torch.float), "kspace_mask.png")

    # prepare k space
    samples = samples.squeeze(1)
    kspace = fftshift(fftn(samples, norm="ortho", dim=(1,2)), dim=(1,2))
    kspace = torch.view_as_real(kspace).permute(0, 3, 1, 2)
    # kspace = kspace * (1-mask)
    
    # save corrupted images
    corrupted = torch.view_as_complex(kspace.permute(0,2,3,1).contiguous())
    corrupted = torch.view_as_real(ifftn(corrupted, dim=(1,2), norm="ortho")).permute(0,3,1,2)
    corrupted = torch.norm(corrupted, dim=1, keepdim=True)
    corrupted = torchvision.utils.make_grid(corrupted, nrow=int(np.sqrt(samples.shape[0])))
    save_image(corrupted, "samples_dutifulpond10_corrupted.png")

    # run inference
    out = sampler.masked_sampling_with_resampling_kspace(kspace, mask, 10, 10, gaussian_scheduling)
    out = torchvision.utils.make_grid(out, nrow=int(np.sqrt(samples.shape[0])))
    save_image(out, "samples_dutifulpond10_resampledkspace.png")

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="DutifulPond10Sampler"
    )
    parser.add_argument("-c","--checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("-n", "--num_samples", type=int)
    parser.add_argument("-a", "--acceleration_factor", type=int)
    parser.add_argument("-f", "--center_fraction", type=float)
    parser.add_argument("-s", "--gaussian_scheduling", action="store_true")
    parser.add_argument("--masked_sampling", action="store_true")
    parser.add_argument("--masked_resampling", action="store_true")
    parser.add_argument("--masked_kspace_sampling", action="store_true")
    parser.add_argument("--masked_kspace_resampling", action="store_true")
    args = parser.parse_args()

    # example
    # python sampler_dutifulpond10.py -c "/itet-stor/peerli/net_scratch/dutiful-pond-10/checkpoint60.pt" -n 16 -a 4 -f 0.1 --gaussian_scheduling --masked_kspace_sampling

    device = torch.device("cuda")
    sampler = setup_sampler(args.checkpoint).to(device)
    samples = get_samples(args.num_samples).to(device)
    if args.masked_sampling:
        masked_sampling2(sampler, samples)
    if args.masked_resampling:
        masked_resampling(sampler, samples)
    if args.masked_kspace_sampling:
        assert args.acceleration_factor is not None
        masked_kspace_sampling(sampler, samples, args.acceleration_factor, args.center_fraction, args.gaussian_scheduling)
    if args.masked_kspace_resampling:
        assert args.acceleration_factor is not None
        masked_kspace_resampling(sampler, samples, args.acceleration_factor, args.center_fraction, args.gaussian_scheduling)