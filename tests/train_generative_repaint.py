import context
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from models.mnist_enc import MNISTEncoder
from models.unet import UNet
from models.repaint_unet.unet import UNetModel
from models.diffusion import DiffusionModel, ForwardDiffusion
import numpy as np
from time import time
from utils.trainer import DiscriminativeTrainer, GenerativeTrainer
import torch.multiprocessing as mp
import os
from utils.mp_setup import DDP_Proc_Group
from utils.datasets import MNISTTrainDataset, Cifar10Dataset, MNISTDebugDataset, Cifar10DebugDataset, FastMRIDebug, FastMRIBrainTrain, QuarterFastMRI, MNISTKSpace, MNISTKSpaceDebug
from utils.helpers import dotdict
import wandb
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

config = dotdict(
    world_size = 2,
    total_epochs = 100,
    log_wandb = True,
    project = "mnist_gen_trials",
    #data_path = "/itet-stor/peerli/bmicdatasets-originals/Originals/fastMRI/brain/multicoil_train",
    data_path = "/itet-stor/peerli/net_scratch",
    checkpoint_folder = "/itet-stor/peerli/net_scratch/run_name", # append wandb run name to this path
    wandb_dir = "/itet-stor/peerli/net_scratch",
    from_checkpoint = False,
    loss_func = F.mse_loss,
    mixed_precision = False,
    optimizer = torch.optim.AdamW,
    lr_scheduler = "cosine_ann_warm",
    cosine_ann_T_0 = 3,
    cosine_ann_T_mult = 2,
    save_every = 1,
    num_samples = 9,
    k_space = True,
    channels_per_att_head = 128,
    batch_size = 64,
    gradient_accumulation_rate = 8,
    learning_rate = 0.0005,
    img_size = 32,
    device_type = "cuda",
    in_channels = 2,
    dataset = MNISTKSpace,
    architecture = DiffusionModel,
    backbone = UNetModel,
    unet_init_channels = 128,
    activation = nn.SiLU,
    backbone_enc_depth = 4,
    num_res_blocks = 2,
    kernel_size = 3,
    attention_downsampling_factors = (4, 8), # at resolutions 8, 4
    dropout = 0.0,
    forward_diff = ForwardDiffusion,
    max_timesteps = 1000,
    t_start = 0.0001,
    t_end = 0.02,
    offset = 0.008,
    max_beta = 0.999,
    schedule_type = "cosine",
    time_enc_dim = 512
)

def load_train_objs(config):
    train_set = config.dataset(config.data_path)
    model = config.architecture(
        backbone = config.backbone(
            image_size = config.img_size,
            in_channels = config.in_channels,
            num_encoding_blocks = config.backbone_enc_depth,
            model_channels = config.unet_init_channels,
            out_channels = config.in_channels,
            num_res_blocks = config.num_res_blocks,
            use_fp16 = False,
            attention_resolutions = config.attention_downsampling_factors,
            time_embed_dim = config.time_enc_dim,
            dropout = config.dropout,
            num_head_channels = config.channels_per_att_head
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
    if config.lr_scheduler == "cosine_ann_warm":
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.cosine_ann_T_0, T_mult=config.cosine_ann_T_0)
        return train_set, model, optimizer, lr_scheduler
    else:
        return train_set, model, optimizer, lr_scheduler
    
def training(rank, world_size, config):
    if (rank == 0) and (config.log_wandb):
        wandb.init(project=config.project, config=config, save_code=True, dir=config.wandb_dir)
    if config.lr_scheduler == "cosine_ann_warm":
        dataset, model, optimizer, lr_scheduler = load_train_objs(config)
    else:
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
        num_gpus = world_size,
        batch_size = config.batch_size, 
        save_every = config.save_every, 
        checkpoint_folder = config.checkpoint_folder, 
        device_type = config.device_type,
        log_wandb = config.log_wandb,
        num_samples = config.num_samples,
        mixed_precision = config.mixed_precision,
        gradient_accumulation_rate = config.gradient_accumulation_rate,
        lr_scheduler = lr_scheduler,
        k_space = config.k_space
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
