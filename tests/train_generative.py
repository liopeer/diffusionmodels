import context
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from models.mnist_enc import MNISTEncoder
from models.unet import UNet
from models.diffusion import DiffusionModel, ForwardDiffusion
import numpy as np
from time import time
from utils.trainer import DiscriminativeTrainer, GenerativeTrainer
import torch.multiprocessing as mp
import os
from utils.mp_setup import DDP_Proc_Group
from utils.datasets import MNISTTrainDataset, UnconditionedCifar10Dataset
from utils.helpers import dotdict
import wandb
import torch.nn.functional as F

config = dotdict(
    total_epochs = 2,
    batch_size = 1000,
    learning_rate = 0.001,
    device_type = "cuda",
    dataset = UnconditionedCifar10Dataset,
    architecture = DiffusionModel,
    backbone = UNet,
    in_channels = 3,
    backbone_enc_depth = 4,
    kernel_size = 3,
    dropout = 0.5,
    forward_diff = ForwardDiffusion,
    max_timesteps = 1000,
    t_start = 0.0001, 
    t_end = 0.02,
    schedule_type = "linear",
    time_enc_dim = 256,
    optimizer = torch.optim.Adam,
    #data_path = os.path.abspath("./data"),
    #checkpoint_folder = os.path.abspath(os.path.join("./data/checkpoints")),
    data_path = "/itet-stor/peerli/net_scratch",
    checkpoint_folder = "/itet-stor/peerli/net_scratch/cifar10_checkpoints",
    save_every = 1,
    loss_func = F.mse_loss,
    log_wandb = True
)

def load_train_objs(config):
    train_set = config.dataset(config.data_path)
    model = config.architecture(
        config.backbone(
            num_encoding_blocks = config.backbone_enc_depth,
            in_channels = config.in_channels,
            kernel_size = config.kernel_size,
            dropout = config.dropout,
            time_emb_size = config.time_enc_dim
        ),
        config.forward_diff(
            config.max_timesteps,
            config.t_start,
            config.t_end,
            config.schedule_type
        ),
        config.time_enc_dim
    )
    optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)
    return train_set, model, optimizer

def training(rank, world_size, config):
    if (rank == 0) and (config.log_wandb):
        wandb.init(project="cifar_gen_trials", config=config, save_code=True)
    dataset, model, optimizer = load_train_objs(config)
    trainer = GenerativeTrainer(
        model, 
        dataset, 
        config.loss_func, 
        optimizer, 
        rank, 
        config.batch_size, 
        config.save_every, 
        config.checkpoint_folder, 
        config.device_type,
        config.log_wandb
    )
    trainer.train(config.total_epochs)

if __name__ == "__main__":
    if config.device_type == "cuda":
        world_size = torch.cuda.device_count()
        print("Device Count:", world_size)
        mp.spawn(DDP_Proc_Group(training), args=(world_size, config), nprocs=world_size)
    else:
        training(0, 0, config)