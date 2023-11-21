import torch
import os
from time import time
from torch import Tensor
from torch import nn
from jaxtyping import Float, Bool
from typing import Callable, Literal, Any, Tuple, Union
from .diffusion import DiffusionModel
from utils.helpers import bytes_to_gb
from torch.fft import fftn, ifftn, fftshift, ifftshift
from tqdm import tqdm
import torchvision
from torchvision.utils import save_image

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
    
    @torch.no_grad()
    def masked_sampling(
            self, 
            partial_img: Float[Tensor, "batch channels height width"], 
            mask: Bool[Tensor, "batch 1 height width"]
        ) -> Float[Tensor, "batch channels height width"]:
        "Mask should be True where we did not sample, False where we sampled."
        assert (mask.shape[2]==self.model.img_size) and (mask.shape[3]==self.model.img_size), "mask size must match image size"
        assert mask.shape[0]==partial_img.shape[0], "batch sizes must be equal"
        beta = self.model.fwd_diff.betas[-1].view(-1,1,1,1)
        noise = self.model.init_noise(partial_img.shape[0]) * torch.sqrt(beta)
        x = noise

        for i in tqdm(reversed(range(1, self.model.fwd_diff.timesteps))):
            t = i * torch.ones((partial_img.shape[0]), dtype=torch.long, device=beta.device)
            x = x * mask

            img_t, _ = self.model.fwd_diff(partial_img, t)
            img_t = img_t * ~mask

            x = x + img_t

            x = self.model.denoise_singlestep(x, t)
        return x
    
    @torch.no_grad()
    def masked_sampling_with_resampling_kspace(
            self,
            partial_kspace: Float[Tensor, "batch 2 height width"],
            mask: Bool[Tensor, "batch 1 height width"],
            num_resamplings: int,
            jump_length: int,
            return_steps: bool=False
        ) -> Float[Tensor, "batch 1 height width"]:
        beta = self.model.fwd_diff.betas[-1].view(-1,1,1,1)
        noise = self.model.init_noise(partial_kspace.shape[0]) * torch.sqrt(beta)
        x = noise

        partial_img = self._to_imgspace(partial_kspace)

        steps = []
        for global_t in tqdm(reversed(range(1, self.model.fwd_diff.timesteps))):
            t = global_t * torch.ones((partial_img.shape[0]), dtype=torch.long, device=beta.device)
            img_t, _ = self.model.fwd_diff(partial_img, t)

            # switching to kspace
            kspace_t = self._to_kspace(img_t)
            x_k = self._to_kspace(x)
            x_k = x_k * mask + kspace_t * ~mask

            # switching to img space
            x = self._to_imgspace(x_k)
            x = torch.norm(x, dim=1, keepdim=True)

            x = self.model.denoise_singlestep(x, t)

            # RESAMPLING
            if (((global_t+1) % jump_length) == 0) and (global_t != self.model.fwd_diff.timesteps-1):
                for _ in range(num_resamplings):
                    # in img space
                    x, _ = self.model.fwd_diff.forward_flexible(x, t, t + jump_length)
                    for local_t in reversed(range(1, jump_length+1)):
                        img_t, _ = self.model.fwd_diff(partial_img, t + local_t)

                        # switching to kspace
                        kspace_t = self._to_kspace(img_t)
                        x_k = self._to_kspace(x)
                        x_k = x_k * mask + kspace_t * ~mask

                        # switching to img space
                        x = self._to_imgspace(x_k)
                        x = torch.norm(x, dim=1, keepdim=True)

                        x = self.model.denoise_singlestep(x, t + local_t)
                        steps.append(global_t + local_t)
        if return_steps:
            return x, torch.tensor(steps)
        return x
    
    @torch.no_grad()
    def masked_sampling_kspace(
            self,
            partial_kspace: Float[Tensor, "batch 2 height width"],
            mask: Bool[Tensor, "batch 1 height width"],
            gaussian_scheduling: bool = True
        ) -> Float[Tensor, "batch 2 height width"]:
        "Mask should be True where we did not sample, False where we sampled."
        beta = self.model.fwd_diff.betas[-1].view(-1,1,1,1)
        noise = self.model.init_noise(partial_kspace.shape[0]) * torch.sqrt(beta)
        x = noise

        partial_img = self._to_imgspace(partial_kspace)

        if gaussian_scheduling:
            sigmas = torch.linspace(0, 1, self.model.fwd_diff.timesteps)
            sigmas = torch.exp(10*sigmas)
            sigmas = sigmas / torch.max(sigmas) * 3 + 0.1
            gaussian_mask = self._2d_gaussian(partial_kspace.shape[-1], normalized_sigma=sigmas[0])
            gaussian_mask = gaussian_mask.unsqueeze(0).unsqueeze(0).to(partial_img.device)
            gaussian_mask = mask.to(torch.float32) * gaussian_mask
            partial_img = self._to_imgspace(partial_kspace * gaussian_mask)

        for i, global_t in tqdm(enumerate(reversed(range(1, self.model.fwd_diff.timesteps)))):
            t = global_t * torch.ones((partial_kspace.shape[0]), dtype=torch.long, device=beta.device)
            img_t, _ = self.model.fwd_diff(partial_img, t)

            # switching to kspace
            kspace_t = self._to_kspace(img_t)
            x_k = self._to_kspace(x)
            if gaussian_scheduling:
                x_k = x_k * gaussian_mask + kspace_t * (1-gaussian_mask)
            else:
                x_k = x_k * mask + kspace_t * ~mask

            # switching to img space
            x = self._to_imgspace(x_k)
            x = torch.norm(x, dim=1, keepdim=True)

            x = self.model.denoise_singlestep(x, t)

            if gaussian_scheduling:
                #gaussian_mask = self._2d_gaussian(partial_kspace.shape[-1], normalized_sigma=sigmas[i])
                gaussian_mask = self._2d_gaussian(partial_kspace.shape[-1], normalized_sigma=sigmas[i])
                gaussian_mask = gaussian_mask.unsqueeze(0).unsqueeze(0).to(partial_img.device)
                gaussian_mask = mask.to(torch.float32) * gaussian_mask
                partial_img = self._to_imgspace(partial_kspace * gaussian_mask)

        return x

    def _to_kspace(self, img: Float[Tensor, "batch 1 height width"]) -> Float[Tensor, "batch 2 height width"]:
        img = img.squeeze(1)
        kspace = fftshift(fftn(img, norm="ortho", dim=(1,2)), dim=(1,2)) # batch height width
        kspace = torch.view_as_real(kspace).permute(0,3,1,2)
        return kspace

    def _2d_gaussian(self, size: Tuple[int], normalized_sigma: float):
        x, y = torch.arange(-size//2, size//2), torch.arange(-size//2, size//2)
        x, y = torch.meshgrid([x,y])
        sigma = normalized_sigma*size/2
        return torch.exp(-(x**2)/(2*sigma**2) - (y**2)/(2*sigma**2))
    
    def _to_imgspace(self, kspace: Float[Tensor, "batch 2 height width"]) -> Float[Tensor, "batch 1 height width"]:
        kspace = torch.view_as_complex(kspace.permute(0,2,3,1).contiguous())
        img = ifftn(kspace, norm="ortho", dim=(1,2))
        img = torch.view_as_real(img).permute(0,3,1,2)
        return torch.norm(img, dim=1, keepdim=True)

    @torch.no_grad()
    def masked_sampling_with_resampling(
            self,
            partial_img: Float[Tensor, "batch channels height width"],
            mask: Bool[Tensor, "batch 1 height width"],
            num_resamplings: int,
            jump_length: int,
            return_steps: bool=False,
        ) -> Union[Float[Tensor, "batch channels height width"], Tuple]:
        """Mask should be True where we did not sample, False where we sampled."""
        beta = self.model.fwd_diff.betas[-1].view(-1,1,1,1)
        noise = self.model.init_noise(partial_img.shape[0]) * torch.sqrt(beta)
        x = noise

        steps = []
        for global_t in reversed(range(1, self.model.fwd_diff.timesteps)):
            t = global_t * torch.ones((partial_img.shape[0]), dtype=torch.long, device=beta.device)
            img_t, _ = self.model.fwd_diff(partial_img, t)
            x = x * mask + img_t * ~mask
            x = self.model.denoise_singlestep(x, t)
            if (((global_t+1) % jump_length) == 0) and (global_t != self.model.fwd_diff.timesteps-1):
                for _ in range(num_resamplings):
                    x, _ = self.model.fwd_diff.forward_flexible(x, t, t + jump_length)
                    for local_t in reversed(range(1, jump_length+1)):
                        img_t, _ = self.model.fwd_diff(partial_img, t + local_t)
                        x = x * mask + img_t * ~mask
                        x = self.model.denoise_singlestep(x, t + local_t)
                        steps.append(global_t + local_t)
        if return_steps:
            return x, torch.tensor(steps)
        return x
    
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