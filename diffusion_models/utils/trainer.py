import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer
import numpy as np
from time import time
import wandb
from typing import Callable, Literal, Any, Tuple
from torch import Tensor
import wandb
from torch.nn import Module
import torchvision
from math import isqrt
from jaxtyping import Float

class Trainer:
    """Trainer Class that trains 1 model instance on 1 device, suited for distributed training."""
    def __init__(
        self,
        model: nn.Module,
        train_data: Dataset,
        loss_func: Callable,
        optimizer: Optimizer,
        gpu_id: int,
        num_gpus: int,
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
            int in range [0, num_GPUs], value does not matter if `device_type!="cuda"`
        num_gpus
            does not matter if `device_type!="cuda"`
        save_every
            checkpoint model & upload data to wandb every `save_every` epoch
        checkpoint_folder
            where to save checkpoints to
        device_type
            specify in case not training no CUDA capable device
        log_wandb
            whether to log to wandb; requires that initialization of wandb process has been done on GPU 0 (and on this GPU only!)
        """
        self.device_type = device_type
        self.gpu_id = gpu_id
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        if device_type != "cuda":
            # distributed training not supported for devices other than CUDA
            self.gpu_id = 0
            self.model = model.to(torch.device(device_type))
            self.train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        else:
            # this works for single and multi-GPU setups
            self.model = self._setup_model(model) # self.model will be DistributedDataParallel-wrapped model
            self.train_data = self._setup_dataloader(train_data) # self.train_data will be DataLoader with DistributedSampler
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.save_every = save_every
        self.checkpoint_folder = checkpoint_folder
        self.log_wandb = log_wandb and (self.gpu_id==0) # only log if in process for GPU 0
        if self.log_wandb:
            wandb.watch(self.model, log="all", log_freq=save_every)
        self.loss_history = []

    def _setup_model(self, model: nn.Module):
        model = model.to(self.gpu_id)
        return DDP(model, device_ids=[self.gpu_id])
    
    def _setup_dataloader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(dataset))

    def _run_batch(self, data: Tuple):
        raise NotImplementedError("Use dedicated subclass (generative/discriminative) of Trainer to run a mini-batch of data.")

    def _run_epoch(self, epoch: int):
        epoch_losses = []
        epoch_time1 = time()
        for data in self.train_data:
            batch_time1 = time()
            if self.device_type == "cuda":
                # move all data inputs onto GPU
                data = tuple(map(lambda x: x.to(self.gpu_id), data))
            else:
                data = tuple(map(lambda x: x.to(self.device_type), data))
            batch_loss = self._run_batch(data)
            epoch_losses.append(batch_loss)
            if self.log_wandb:
                wandb.log({"epoch": epoch, "loss": batch_loss, "batch_time": time()-batch_time1})
        if self.log_wandb:
            wandb.log({"epoch_loss": np.mean(epoch_losses), "epoch_time": time()-epoch_time1})
        self.loss_history.append(np.mean(epoch_losses))
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {self.batch_size} | Steps: {len(self.train_data)} | Loss: {np.mean(epoch_losses)} | Time: {time()-epoch_time1:.2f}s")

    def _save_checkpoint(self, epoch: int):
        if self.device_type == "cuda":
            # for DistributedDataParallel-wrapped model (nn.Module)
            ckp = self.model.module.state_dict()
        else:
            ckp = self.model.state_dict()
        if not os.path.isdir(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)
        path = os.path.join(self.checkpoint_folder, f"checkpoint{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": ckp,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss_history[-1],
                "device_type": self.device_type
            },
            path
        )
        print(f"Epoch {epoch} | Training checkpoint saved at {path}")

    def load_checkpoint(self, checkpoint_path: str):
        map_location = None
        if ckp["device_type"] != self.device_type:
            map_location = torch.device(self.device_type)
        ckp = torch.load(checkpoint_path, map_location=map_location)
        if self.device_type == "cuda":
            self.model.module.load_state_dict(ckp["model_state_dict"])
        else:
            self.model.load_state_dict(ckp["model_state_dict"])
        self.optimizer.load_state_dict(ckp["optimizer_state_dict"])
        self.loss_history.append(ckp["loss"])

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
    def __init__(self, model: Module, train_data: Dataset, loss_func: Callable[..., Any], optimizer: Optimizer, gpu_id: int, num_gpus: int, batch_size: int, save_every: int, checkpoint_folder: str, device_type: Literal['cuda', 'mps', 'cpu'], log_wandb: bool = True) -> None:
        super().__init__(model, train_data, loss_func, optimizer, gpu_id, num_gpus, batch_size, save_every, checkpoint_folder, device_type, log_wandb)

    def _run_batch(self, data):
        """Run a data batch.

        Parameters
        ----------
        data
            tuple of input data and targets, several inputs are possible: (input1, input2, ..., inputN, target)
        """
        *source, targets = data
        self.optimizer.zero_grad()
        pred = self.model(*source)
        loss = self.loss_func(pred, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
class GenerativeTrainer(Trainer):
    def __init__(
            self, 
            model: Module, 
            train_data: Dataset, 
            loss_func: Callable[..., Any], 
            optimizer: Optimizer, 
            gpu_id: int,
            num_gpus: int,
            batch_size: int, 
            save_every: int, 
            checkpoint_folder: str, 
            device_type: Literal['cuda', 'mps', 'cpu'], 
            log_wandb: bool,
            num_samples: int
        ) -> None:
        """Constructor of GenerativeTrainer class.

        Parameters
        ----------
        model
            instance of nn.Module, must implement a `sample(num_samples: int)` method
        """
        super().__init__(model, train_data, loss_func, optimizer, gpu_id, num_gpus, batch_size, save_every, checkpoint_folder, device_type, log_wandb)

        def is_square(i: int) -> bool:
            return i == isqrt(i) ** 2
            
        def closest_square_divisible_by(num_samples: int, div: int):
            counter = 1
            while (counter**2 % div != 0) and (counter**2 < num_samples):
                counter += 1
            return counter**2
        
        if (num_samples % self.num_gpus != 0) or (not is_square(num_samples)):
            num_samples = closest_square_divisible_by(num_samples, self.num_gpus)
        self.num_samples = num_samples

    def _run_batch(self, data):
        """Run a data batch.

        Parameters
        ----------
        data
            tuple containing training batch
        """
        self.optimizer.zero_grad()
        pred = self.model(*data)
        loss = self.loss_func(*pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _wandb_log_sample(self, sample: Float[Tensor, "channels height width"], epoch: int):
        images = wandb.Image(sample, caption=f"Samples Epoch {epoch}")
        wandb.log({"examples": images}, commit=False)
    
    def _save_samples(self, samples: Float[Tensor, "samples channels height width"], storage_folder: str, epoch: int):
        samples = torchvision.utils.make_grid(samples, nrow=int(np.sqrt(self.num_samples)))
        path = os.path.join(self.checkpoint_folder, f"samples_epoch{epoch}.png")
        torchvision.utils.save_image(samples, path)
        print(f"Epoch {epoch} | Samples saved at {path}")
        if self.log_wandb:
            self._wandb_log_sample(samples, epoch)

    def get_samples(self, num_samples: int):
        if (self.device_type == "cuda") and (self.num_gpus == 1):
            samples = self.model.module.sample(self.num_samples)
        if (self.device_type == "cuda") and (self.num_gpus > 1):
            samples = self.model.module.sample(int(self.num_samples//self.num_gpus))
            total_samples = torch.zeros(samples.shape[0]*self.num_gpus, device=samples.device)
            dist.all_gather_into_tensor(total_samples, samples)
            samples = total_samples
        else:
            samples = self.model.sample(self.num_samples)
        return samples
    
    def _save_checkpoint(self, epoch: int):
        """Overwriting original method - Checkpoint model and generate samples."""
        super()._save_checkpoint(epoch)
        samples = self.get_samples(self.num_samples)
        self._save_samples(self.num_samples, self.checkpoint_folder, epoch)