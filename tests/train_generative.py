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
from utils.datasets import MNISTTrainDataset, Cifar10Dataset, MNISTDebugDataset, Cifar10DebugDataset, FastMRIDebug, FastMRIBrainTrain, QuarterFastMRI
from utils.helpers import dotdict
import wandb
import torch.nn.functional as F

config = dotdict(
    total_epochs = 1000,
    log_wandb = True,
    project = "fastMRI_gen_trials",
    data_path = "/itet-stor/peerli/bmicdatasets-originals/Originals/fastMRI/brain/multicoil_train",
    #data_path = "/itet-stor/peerli/net_scratch",
    checkpoint_folder = "/itet-stor/peerli/net_scratch/run_name", # append wandb run name to this path
    wandb_dir = "/itet-stor/peerli/net_scratch",
    from_checkpoint = "/itet-stor/peerli/net_scratch/super-rain-7/checkpoint490.pt", #"/itet-stor/peerli/net_scratch/ghoulish-goosebump-9/checkpoint30.pt",
    loss_func = F.mse_loss,
    save_every = 10,
    num_samples = 4,
    show_denoising_history = False,
    show_history_every = 50,
    batch_size = 32,
    learning_rate = 0.0001,
    img_size = 64,
    device_type = "cuda",
    in_channels = 1,
    dataset = QuarterFastMRI,
    architecture = DiffusionModel,
    backbone = UNet,
    attention = False,
    attention_heads = 4,
    attention_ff_dim = None,
    unet_init_channels = 64,
    activation = nn.SiLU,
    backbone_enc_depth = 5,
    kernel_size = 3,
    dropout = 0.0,
    forward_diff = ForwardDiffusion,
    max_timesteps = 1000,
    t_start = 0.0001,
    t_end = 0.02,
    offset = 0.008,
    max_beta = 0.999,
    schedule_type = "cosine",
    time_enc_dim = 512,
    optimizer = torch.optim.Adam
)

def load_train_objs(config):
    train_set = config.dataset(config.data_path)
    model = config.architecture(
        backbone = config.backbone(
            num_encoding_blocks = config.backbone_enc_depth,
            in_channels = config.in_channels,
            kernel_size = config.kernel_size,
            dropout = config.dropout,
            activation = config.activation,
            time_emb_size = config.time_enc_dim,
            init_channels = config.unet_init_channels,
            attention = config.attention,
            attention_heads = config.attention_heads,
            attention_ff_dim = config.attention_ff_dim
        ),
        fwd_diff = config.forward_diff(
            timesteps = config.max_timesteps,
            start = config.t_start,
            end = config.t_end,
            offset = config.offset,
            max_beta = config.max_beta,
            type = config.schedule_type
        ),
        img_size = config.img_size,
        time_enc_dim = config.time_enc_dim,
        dropout = config.dropout
    )
    optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)
    return train_set, model, optimizer

def training(rank, world_size, config):
    if (rank == 0) and (config.log_wandb):
        wandb.init(project=config.project, config=config, save_code=True, dir=config.wandb_dir)
    dataset, model, optimizer = load_train_objs(config)
    if (config.device_type == "cuda") and (world_size > 1):
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if (rank == 0) and config.log_wandb and ("run_name" in config.checkpoint_folder):
        base_folder = os.path.dirname(config.checkpoint_folder)
        config.checkpoint_folder = os.path.join(base_folder, wandb.run.name)
    trainer = GenerativeTrainer(
        model = model, 
        train_data = dataset, 
        loss_func = config.loss_func, 
        optimizer = optimizer, 
        gpu_id = rank, 
        batch_size = config.batch_size, 
        save_every = config.save_every, 
        checkpoint_folder = config.checkpoint_folder, 
        device_type = config.device_type,
        log_wandb = config.log_wandb,
        num_samples = config.num_samples,
        show_denoising_process = config.show_denoising_history,
        show_denoising_every = config.show_history_every
    )
    if config.from_checkpoint:
        trainer.load_checkpoint(config.from_checkpoint)
    trainer.train(config.total_epochs)

if __name__ == "__main__":
    if config.device_type == "cuda":
        world_size = torch.cuda.device_count()
        print("Device Count:", world_size)
        mp.spawn(DDP_Proc_Group(training), args=(world_size, config), nprocs=world_size)
    else:
        training(0, 0, config)
