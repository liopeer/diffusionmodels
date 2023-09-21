import context
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from models.mnist_enc import MNISTEncoder
import numpy as np
from time import time
from utils.trainer import DiscriminativeTrainer
import torch.multiprocessing as mp
import os
from utils.mp_setup import DDP_Proc_Group
from utils.dataloaders import MNISTTrainLoader
from utils.helpers import dotdict
import wandb
import torch.nn.functional as F

config = dotdict(
    total_epochs = 5,
    batch_size = 1000,
    learning_rate = 0.001,
    device_type = "cpu",
    dataloader = MNISTTrainLoader,
    architecture = MNISTEncoder,
    out_classes = 10,
    optimizer = torch.optim.Adam,
    kernel_size = 3,
    #data_path = os.path.abspath("./data"),
    #checkpoint_folder = os.path.abspath(os.path.join("./data/checkpoints")),
    data_path = "/itet-stor/peerli/net_scratch",
    checkpoint_folder = "/itet-stor/peerli/net_scratch/mnist_checkpoints",
    save_every = 10,
    loss_func = F.cross_entropy,
    log_wandb = False
)

def load_train_objs(config):
    train_set = config.dataloader(config.data_path)
    model = config.architecture(config.out_classes, config.kernel_size)
    optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)
    return train_set, model, optimizer

def training(rank, world_size, config):
    if rank == 0:
        wandb.init(project="mnist_trials", config=config, save_code=True)
    dataset, model, optimizer = load_train_objs(config)
    trainer = DiscriminativeTrainer(
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
    with wandb.init(project="mnist_trials", config=config, save_code=True):
        if config.device_type == "cuda":
            world_size = torch.cuda.device_count()
            print("Device Count:", world_size)
            mp.spawn(DDP_Proc_Group(training), args=(world_size, config), nprocs=world_size)
        else:
            training(0, 0, config)
