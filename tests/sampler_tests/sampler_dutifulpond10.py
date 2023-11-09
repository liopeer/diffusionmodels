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

def run():
    config = None
    with open("config_dutifulpond10.json", "r") as f:
        config = json.load(f)
    ckp_num = 40
    ckpt_path = f"/itet-stor/peerli/net_scratch/dutiful-pond-10/checkpoint{ckp_num}.pt"
    device = torch.device("cuda")
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
    ).to(device)
    out = sampler.sample(16)
    out = torchvision.utils.make_grid(out, nrow=4)
    torchvision.utils.save_image(out, "samples_dutifulpond10.png")

if __name__ == "__main__":
    run()