import torch
from torch import nn, Tensor
from jaxtyping import Float, Int64
from typing import Literal

class DiffusionModel(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            timesteps: int,
            t_start: float=0.0001,
            t_end: float=0.02,
            schedule_type: Literal["linear", "cosine"]="linear"
        ) -> None:
        super().__init__()
        self.model = backbone
        self.fwd_diff = ForwardDiffusion(timesteps, t_start, t_end, schedule_type)

    def forward(self, x):
        t = self._sample_timestep(x.shape[0])
        t = t.unsqueeze(-1).type(torch.float)
        t = self._pos_encoding(t, self.time_dim)
        x_t, noise = self.fwd_diff(x, t)
        noise_pred = self.model(x_t, t)
        return noise_pred, noise
    
    def _pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def _sample_timestep(self, batch_size: int) -> Int64[Tensor, "batch"]:
        return torch.randint(low=1, high=self.fwd_diff.noise_steps, size=(batch_size,))


class ForwardDiffusion(nn.Module):
    """Class for forward diffusion process in DDPMs (denoising diffusion probabilistic models).
    
    Attributes
    ----------
    timesteps
        max number of supported timesteps of the schedule
    start
        start value of scheduler
    end
        end value of scheduler
    type
        type of scheduler, currently linear and cosine supported
    """
    def __init__(self, timesteps: int, start: float=0.0001, end: float=0.02, type: Literal["linear", "cosine"]="linear") -> None:
        """Constructor of ForwardDiffusion class.
        
        Parameters
        ----------
        timesteps
            timesteps
        start
            start
        end
            end
        type
            type
        """
        super().__init__()
        self.timesteps = timesteps
        self.start = start
        self.end = end
        self.type = type
        if self.type == "linear":
            self.betas = self._linear_scheduler(timesteps=self.timesteps, start=self.start, end=self.end)
        elif self.type == "cosine":
            self.betas = self._cosine_scheduler(timesteps=self.timesteps, start=self.start, end=self.end)
        else:
            raise NotImplementedError("Invalid scheduler option:", type)
        self.alphas = 1. - self.betas

        self.register_buffer("alphas_dash", torch.cumprod(self.alphas, axis=0), persistent=False)
        self.register_buffer("sqrt_alphas_dash", torch.sqrt(self.alphas_dash), persistent=False)
        self.register_buffer("sqrt_one_minus_alpha_dash", 1. - self.alphas_dash, persistent=False)

        self.register_buffer("noise_normal", torch.empty((1)), persistent=False)

    def forward(self, x_0: Float[Tensor, "batch channels height width"], t: int) -> Float[Tensor, "batch channels height width"]:
        """Forward method of ForwardDiffusion class.
        
        Parameters
        ----------
        x_0
            input tensor where noise should be applied to
        t
            timestep of the noise scheduler from which noise should be chosen

        Returns
        -------
        Float[Tensor, "batch channels height width"]
            tensor with applied noise according to schedule and chosen timestep
        """
        self.noise_normal = torch.randn_like(x_0)
        if t > self.timesteps-1:
            raise IndexError("t ({}) chosen larger than max. available t ({})".format(t, self.timesteps-1))
        sqrt_alpha_dash_t = self.sqrt_alphas_dash[t]
        sqrt_one_minus_alpha_dash_t = self.sqrt_one_minus_alpha_dash[t]
        x_t = sqrt_alpha_dash_t * x_0 + sqrt_one_minus_alpha_dash_t * self.noise_normal
        return x_t

    def _linear_scheduler(self, timesteps, start, end):
        return torch.linspace(start, end, timesteps)
    
    def _cosine_scheduler(self, timesteps, start, end):
        raise NotImplementedError("Cosine scheduler not implemented yet.")