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
from typing import Callable, Literal, Any, Tuple
import wandb
from torch.nn import Module
import torchvision

class Trainer:
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
        self.log_wandb = log_wandb and (self.gpu_id==0)
        if self.log_wandb:
            wandb.watch(self.model, log="all", log_freq=save_every)

    def _setup_model(self, model):
        model = model.to(self.gpu_id)
        return DDP(model, device_ids=[self.gpu_id])
    
    def _setup_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(dataset))

    def _run_batch(self, data):
        raise NotImplementedError("use dedicated subclass")

    def _run_epoch(self, epoch):
        epoch_losses = []
        time1 = time()
        for data in self.train_data:
            if self.device_type == "cuda":
                data = tuple(map(lambda x: x.to(self.gpu_id), data))
            else:
                data = tuple(map(lambda x: x.to(self.device_type), data))
            batch_loss = self._run_batch(data)
            epoch_losses.append(batch_loss)
        if self.log_wandb:
            wandb.log({"epoch": epoch, "loss": np.mean(epoch_losses), "epoch_time": time()-time1})
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {self.batch_size} | Steps: {len(self.train_data)} | Loss: {np.mean(epoch_losses)} | Time: {time()-time1:.2f}s")

    def _save_checkpoint(self, epoch):
        if self.device_type == "cuda":
            ckp = self.model.module.state_dict()
        else:
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
            self._run_epoch(epoch)
            if (self.gpu_id == 0) and (epoch % self.save_every == 0) and (epoch != 0):
                self._save_checkpoint(epoch)

class DiscriminativeTrainer(Trainer):
    def __init__(self, model: Module, train_data: Dataset, loss_func: Callable[..., Any], optimizer: Optimizer, gpu_id: int, batch_size: int, save_every: int, checkpoint_folder: str, device_type: Literal['cuda', 'mps', 'cpu'], log_wandb: bool = True) -> None:
        super().__init__(model, train_data, loss_func, optimizer, gpu_id, batch_size, save_every, checkpoint_folder, device_type, log_wandb)

    def _run_batch(self, data):
        """Run a data batch.

        Parameters
        ----------
        data
            tuple of training batch and targets
        """
        source, targets = data
        self.optimizer.zero_grad()
        pred = self.model(source)
        loss = self.loss_func(pred, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
class GenerativeTrainer(Trainer):
    def __init__(self, model: Module, train_data: Dataset, loss_func: Callable[..., Any], optimizer: Optimizer, gpu_id: int, batch_size: int, save_every: int, checkpoint_folder: str, device_type: Literal['cuda', 'mps', 'cpu'], log_wandb: bool = True) -> None:
        super().__init__(model, train_data, loss_func, optimizer, gpu_id, batch_size, save_every, checkpoint_folder, device_type, log_wandb)

    def _run_batch(self, data):
        """Run a data batch.

        Parameters
        ----------
        data
            single item tuple of training batch
        """
        self.optimizer.zero_grad()
        ### to be changed!
        pred = self.model(data[0])
        loss = self.loss_func(*pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _save_checkpoint(self, epoch):
        """Overwriting original method."""
        super()._save_checkpoint(epoch)
        if self.device_type == "cuda":
            samples = self.model.module.sample(25, 32)
        else:
            sample = self.model.sample(25, 32)
        samples = torchvision.utils.make_grid(samples, nrow=5)
        if self.log_wandb:
            images = wandb.Image(
                samples, 
                caption=f"Samples Epoch {epoch}"
            )
            wandb.log({"examples": images})
        path = os.path.join(self.checkpoint_folder, f"samples_epoch{epoch}.png")
        torchvision.utils.save_image(samples, path)
        print(f"Epoch {epoch} | Samples saved at {path}")
