import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer
import numpy as np
from time import time
import wandb
from typing import Callable, Literal
import wandb

class DiscriminativeTrainer:
    """Trainer Class that trains 1 model instance on 1 device."""
    def __init__(
        self,
        model: nn.Module,
        train_data: Dataset,
        loss_func: Callable,
        optimizer: Optimizer,
        gpu_id: int,
        batch_size: int,
        save_every: int,
        checkpoint_folder: str,
        device_type: Literal["cuda","mps","cpu"],
        log_wandb: bool=True
    ) -> None:
        """Constructor of Trainer Class.
        
        Parameters
        ----------
        model
            instance of nn.Module to be copied to a GPU
        train_data
            Dataset instance
        loss_func
            criterion to determine the loss
        optimizer
            torch.optim instance with model.parameters and learning rate passed
        gpu_id
            int in range [0, num_GPUs]
        save_every
            how often (epochs) to save model checkpoint
        checkpoint_folder
            where to save checkpoint to
        device_type
            specify in case not training no CUDA capable device
        log_wandb
            whether to log to wandb; requires that initialization has been done
        """
        self.device_type = device_type
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        if device_type != "cuda":
            self.gpu_id = 0
            self.model = model.to(torch.device(device_type))
            self.train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        else:
            self.model = self._setup_model(model)
            self.train_data = self._setup_dataloader(train_data)
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.save_every = save_every
        self.checkpoint_folder = checkpoint_folder
        self.log_wandb = log_wandb
        if log_wandb:
            wandb.watch(self.model, log="all", log_freq=save_every)

    def _setup_model(self, model):
        model = model.to(self.gpu_id)
        return DDP(model, device_ids=[self.gpu_id])
    
    def _setup_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(dataset))

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        pred = self.model(source)
        loss = self.loss_func(pred, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _run_epoch_nonCuda(self, epoch):
        epoch_losses = []
        time1 = time()
        for source, targets in self.train_data:
            source, targets = source.to(self.device_type), targets.to(self.device_type)
            batch_loss = self._run_batch(source, targets)
            epoch_losses.append(batch_loss)
        if self.log_wandb:
            wandb.log({"epoch": epoch, "loss": np.mean(epoch_losses)})
        print(f"[{self.device_type}{self.gpu_id}] Epoch {epoch} | Batchsize: {self.batch_size} | Steps: {len(self.train_data)} | Loss: {np.mean(epoch_losses)} | Time: {time()-time1:.2f}s")

    def _run_epoch(self, epoch):
        epoch_losses = []
        time1 = time()
        for source, targets in self.train_data:
            source, targets = source.to(self.gpu_id), targets.to(self.gpu_id)
            batch_loss = self._run_batch(source, targets)
            epoch_losses.append(batch_loss)
        if self.log_wandb:
            wandb.log({"epoch": epoch, "loss": np.mean(epoch_losses)})
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {self.batch_size} | Steps: {len(self.train_data)} | Loss: {np.mean(epoch_losses)} | Time: {time()-time1:.2f}s")

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        if not os.path.isdir(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)
        path = os.path.join(self.checkpoint_folder, f"checkpoint{epoch}.pt")
        torch.save(ckp, path)
        print(f"Epoch {epoch} | Training checkpoint saved at {path}")

    def train(self, max_epochs: int):
        """Train method of Trainer class.
        
        Parameters
        ----------
        max_epochs
            how many epochs to train the model
        """
        for epoch in range(max_epochs):
            if self.device_type != "cuda":
                self._run_epoch_nonCuda(epoch)
            else:
                self._run_epoch(epoch)
            if (self.gpu_id == 0) and (epoch % self.save_every == 0) and (epoch != 0):
                self._save_checkpoint(epoch)