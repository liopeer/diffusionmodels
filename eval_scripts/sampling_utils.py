import context
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
    return mask.unsqueeze(0)

def apply_mask(sampler, x: Float[Tensor, "batch channel height width"], mask: Float[Tensor, "height width"]):
    x = sampler._to_kspace(x)
    x = x * mask.unsqueeze(0).unsqueeze(0)
    return sampler._to_imgspace(x)