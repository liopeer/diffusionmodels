import sys
sys.path.append("../..")
from diffusion_models.models.sampler import DiffusionSampler
from diffusion_models.models.unet import UNet
from diffusion_models.models.diffusion import DiffusionModel, ForwardDiffusion
import torch
import torch.nn as nn
import os
import json
import torchvision
from torchvision.utils import save_image
import os
from torch import Tensor
from jaxtyping import Float, Bool
from diffusion_models.utils.datasets import FastMRIBrainTrain
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
    with open("config_stellarfire1.json", "r") as f:
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