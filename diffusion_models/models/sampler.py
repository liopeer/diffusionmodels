import torch
import os
from time import time
from torch import Tensor
from torch import nn
from jaxtyping import Float
from typing import Callable, Literal, Any, Tuple
from .diffusion import DiffusionModel
from utils.helpers import bytes_to_gb

class DiffusionSampler(nn.Module):
    def __init__(
            self,
            model: DiffusionModel,
            checkpoint: str,
            device_type: Literal["cuda","mps","cpu"]="cuda",
            mixed_precision: bool=False
        ) -> None:
        """Constructor of DiffusionSampler.

        Parameters
        ----------
        model
            instance of DiffusionModel
        checkpoint
            path to checkpoint to be loaded
        device_type
            what device inference should be executed on
        mixed_precision
            whether mixed precision should be used during inference
        """
        super().__init__()
        self.model = model
        state_dict = self._load_model_statedict(checkpoint, device_type)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.checkpoint = checkpoint
        self.device_type = device_type
        self.mixed_precision = mixed_precision

    def _load_model_statedict(self, ckpt_path: str, device_type: str):
        map_location = None
        if device_type != "cuda":
            map_location = torch.device(device_type)
        ckp = torch.load(ckpt_path, map_location=map_location)
        if "model_state_dict" in ckp.keys():
            return ckp["model_state_dict"]
        return ckp
    
    def masked_sampling(mask: Float[Tensor, "batch channels height width"]):
        "Mask should "
    
    @torch.no_grad()
    def sample(self, num_samples: int):
        if self.mixed_precision:
            with torch.autocast(self.device_type, dtype=torch.float16):
                samples = self.model.sample(num_samples)
        else:
            samples =  self.model.sample(num_samples)
        max_mem = bytes_to_gb(torch.cuda.max_memory_allocated())
        print(f"Max Memory Allocated: {max_mem:.2f}")
        return samples
    
    def sample_schedule(self, num_samples: int):
        pass