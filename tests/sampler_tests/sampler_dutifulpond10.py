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
        out = torch.load(path).to(device)
    else:
        out = next(iter(dl))[0].to(device)
        torch.save(out, path)
    samples = torchvision.utils.make_grid(out, nrow=int(np.sqrt(num_samples)))
    save_image(samples, path.split(".")[0] + ".png")
    return out

def masked_sampling(sampler, samples):
    mask = torch.zeros(*samples.shape, dtype=torch.bool, device=device)
    mask[:, :, 0:80, 0:80] = 1
    masked_samples = torchvision.utils.make_grid(samples*~mask, nrow=4)
    save_image(masked_samples, "samples_dutifulpond10_masked.png")
    # out2 = sampler.masked_sampling(samples, mask)
    out2 = sampler.masked_sampling(samples*~mask, mask)
    recon_samples = torchvision.utils.make_grid(out2, nrow=int(np.sqrt(samples.shape[0])))
    save_image(recon_samples, "samples_dutifulpond10_reconstructed.png")

def masked_resampling(sampler, samples):
    mask = torch.zeros(*samples.shape, dtype=torch.bool, device=device)
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

def masked_kspace_sampling(sampler, samples, acceleration_factor):
    # create mask
    mask = torch.ones((samples.shape[0], 1, samples.shape[2], samples.shape[3]), dtype=torch.bool, device=device)
    center_fraction = 1 / (acceleration_factor * 2)
    width = samples.shape[-1]
    center_width = int(width * center_fraction)
    middle = width // 2
    mask[:, :, :, middle-center_width//2 : middle+center_width//2] = 0
    mask[:, :, :, : middle-center_width//2 : acceleration_factor] = 0
    mask[:, :, :, middle+center_width//2 : : acceleration_factor] = 0
    save_image(mask[0].to(torch.float), "kspace_mask.png")

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
    save_image(corrupted, "samples_dutifulpond10_corrupted.png")

    # run inference
    out = sampler.masked_sampling_kspace(kspace, mask)
    out = torchvision.utils.make_grid(out, nrow=int(np.sqrt(samples.shape[0])))
    save_image(out, "samples_dutifulpond10_reconstructedkspace.png")

def masked_kspace_resampling(sampler, samples, acceleration_factor):
    # create mask
    mask = torch.ones((samples.shape[0], 1, samples.shape[2], samples.shape[3]), dtype=torch.bool, device=device)
    center_fraction = 1 / (acceleration_factor * 2)
    width = samples.shape[-1]
    center_width = int(width * center_fraction)
    middle = width // 2
    mask[:, :, :, middle-center_width//2 : middle+center_width//2] = 0
    mask[:, :, :, : middle-center_width//2 : acceleration_factor] = 0
    mask[:, :, :, middle+center_width//2 : : acceleration_factor] = 0
    save_image(mask[0].to(torch.float), "kspace_mask.png")

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
    save_image(corrupted, "samples_dutifulpond10_corrupted.png")

    # run inference
    out = sampler.masked_sampling_with_resampling_kspace(kspace, mask)
    out = torchvision.utils.make_grid(out, nrow=int(np.sqrt(samples.shape[0])))
    save_image(out, "samples_dutifulpond10_reconstructedkspace.png")

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="DutifulPond10Sampler"
    )
    parser.add_argument("-c","--checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("-n", "--num_samples", type=int)
    parser.add_argument("-a", "--acceleration_factor", type=int)
    parser.add_argument("--masked_sampling", action="store_true")
    parser.add_argument("--masked_resampling", action="store_true")
    parser.add_argument("--masked_kspace_sampling", action="store_true")
    parser.add_argument("--masked_kspace_resampling", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    sampler = setup_sampler(args.checkpoint).to(device)
    samples = get_samples(args.num_samples).to(device)
    if args.masked_sampling:
        masked_sampling(sampler, samples)
    if args.masked_resampling:
        masked_resampling(sampler, samples)
    if args.masked_kspace_sampling:
        assert args.acceleration_factor is not None
        masked_kspace_sampling(sampler, samples, args.acceleration_factor)
    if args.masked_kspace_resampling:
        assert args.acceleration_factor is not None
        masked_kspace_resampling(sampler, samples, args.acceleration_factor)