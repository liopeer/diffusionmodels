import torch
from train_generative import config
import torchvision
import torch.nn as nn
from math import sqrt

model = config.architecture(
        backbone = config.backbone(
            num_encoding_blocks = 4,
            in_channels = 1,
            kernel_size = 3,
            dropout = config.dropout,
            activation = nn.SiLU,
            time_emb_size = config.time_enc_dim,
            init_channels = 128,
            attention = False,
            attention_heads = 0,
            attention_ff_dim = 0
        ),
        fwd_diff = config.forward_diff(
            timesteps = config.max_timesteps,
            start = config.t_start,
            end = config.t_end,
            offset = config.offset,
            max_beta = config.max_beta,
            type = "linear"
        ),
        img_size = config.img_size,
        time_enc_dim = config.time_enc_dim,
        dropout = config.dropout
    )

model = model.to("cuda")
model.load_state_dict(torch.load("/itet-stor/peerli/net_scratch/ghoulish-goosebump-9/checkpoint90.pt"))

samples = model.sample(9)
samples = torchvision.utils.make_grid(samples, nrow=int(sqrt(9)))
torchvision.utils.save_image(samples, "/home/peerli/Downloads/sample2.png")