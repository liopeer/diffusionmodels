import torch
from torch import nn, Tensor
from jaxtyping import Float, Int64, Int
from typing import Literal, Tuple, Union, List
from models.positional_encoding import PositionalEncoding
import math

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
    def __init__(self, timesteps: int, start: float=0.0001, end: float=0.02, offset: float=0.008, max_beta: float=0.999, type: Literal["linear", "cosine"]="linear") -> None:
        """Constructor of ForwardDiffusion class.
        
        Parameters
        ----------
        timesteps
            total number of timesteps in diffusion process
        start
            start beta for linear scheduler
        end
            end beta for linear scheduler
        offset
            offset parameter for cosine scheduler
        max_beta
            maximal value to clip betas for cosine scheduler
        type
            type of scheduler, either linear or cosine
        """
        super().__init__()
        self.timesteps = timesteps
        self.start = start
        self.end = end
        self.offset = offset
        self.type = type
        self.max_beta = max_beta
        if self.type == "linear":
            self.init_betas = self._linear_scheduler(timesteps=self.timesteps, start=self.start, end=self.end)
        elif self.type == "cosine":
            self.init_betas = self._cosine_scheduler(timesteps=self.timesteps, offset=self.offset, max_beta=self.max_beta)
        else:
            raise NotImplementedError("Invalid scheduler option:", type)
        self.init_alphas = 1. - self.init_betas

        self.register_buffer("alphas", self.init_alphas, persistent=False)
        self.register_buffer("betas", self.init_betas, persistent=False)
        self.register_buffer("alphas_dash", torch.cumprod(self.alphas, axis=0), persistent=False)
        self.register_buffer("alphas_dash_prev", torch.cat([torch.tensor([1.0]), self.alphas_dash[:-1]]), persistent=False)
        self.register_buffer("alphas_dash_next", torch.cat([self.alphas_dash[1:], torch.tensor([0.0])]), persistent=False)
        self.register_buffer("sqrt_alphas_dash", torch.sqrt(self.alphas_dash), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_dash", torch.sqrt(1. - self.alphas_dash), persistent=False)
        self.register_buffer("log_one_minus_alphas_dash", torch.log(1. - self.alphas_dash), persistent=False)
        self.register_buffer("sqrt_recip_alphas_dash", torch.sqrt(1.0 / self.alphas_dash))
        self.register_buffer("sqrt_recipminus1_alphas_dash", torch.sqrt(1.0 / self.alphas_dash - 1.0))

    def forward(
            self, 
            x_0: Float[Tensor, "batch channels height width"], 
            t: Int[Tensor, "batch"]
        ) -> Float[Tensor, "batch channels height width"]:
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
        noise_normal = torch.randn_like(x_0, device=x_0.device)
        if True in torch.gt(t, self.timesteps-1):
            raise IndexError("t ({}) chosen larger than max. available t ({})".format(t, self.timesteps-1))
        sqrt_alphas_dash_t = self.sqrt_alphas_dash[t]
        sqrt_one_minus_alphas_dash_t = self.sqrt_one_minus_alphas_dash[t]
        x_t = sqrt_alphas_dash_t.view(-1, 1, 1, 1) * x_0
        x_t += sqrt_one_minus_alphas_dash_t.view(-1, 1, 1, 1) * noise_normal
        return x_t, noise_normal
    
    def forward_flexible(
            self,
            x_t1: Float[Tensor, "batch channels height width"],
            t_1: Int64[Tensor, "batch"],
            t_2: Int64[Tensor, "batch"]
        ) -> Float[Tensor, "batch channels height width"]:
        """Flexible method that enables jumping from/to any timestep in the forward diffusion process.
        
        Parameters
        ----------
        x_t1
            batch of (partially noisy) inputs of different stages
        t_1
            initial timesteps of forward process (that above x_t1 are in at the moment)
        t_2
            timesteps that we would x_t1 transport to (elements must be larger than corresponding elements in t_1)
        """
        diff = t_2 - t_1
        if diff[diff<0].shape[0] != 0:
            raise ValueError("Timesteps in forward process must increase.")
        noise_normal = torch.randn_like(x_t1, device=x_t1.device)
        if (True in torch.gt(t_1, self.timesteps-1)) or (True in torch.gt(t_2, self.timesteps-1)):
            raise IndexError("t ({}, {}) chosen larger than max. available t ({})".format(t_1, t_2, self.timesteps-1))
        batch_sqrt_alphas_dash = torch.zeros((t_1.shape[0]))
        batch_sqrt_one_minus_alpha_dash = torch.zeros((t_1.shape[0]))
        for sample in range(x_t1.shape[0]):
            alphas_interval = self.alphas[t_1[sample]:t_2[sample]+1]
            alphas_dash_interval = torch.cumprod(alphas_interval, axis=0)
            sqrt_alphas_dash_interval = torch.sqrt(alphas_dash_interval)
            sqrt_one_minus_alphas_dash_interval = torch.sqrt(1. - alphas_dash_interval)
            batch_sqrt_alphas_dash[sample] = sqrt_alphas_dash_interval
            batch_sqrt_one_minus_alpha_dash[sample] = sqrt_one_minus_alphas_dash_interval
        mean = batch_sqrt_alphas_dash.view(-1, 1, 1, 1) * x_t1
        out = mean + batch_sqrt_one_minus_alpha_dash.view(-1, 1, 1, 1) * noise_normal
        return out, noise_normal

    def _linear_scheduler(self, timesteps, start, end):
        return torch.linspace(start, end, timesteps)
    
    def _cosine_scheduler(self, timesteps, offset, max_beta):
        """t is actually t/T from the paper"""
        return self._betas_for_alpha_bar(timesteps, lambda t: math.cos((t + offset) / (1.0 + offset) * math.pi / 2) ** 2, max_beta)

    def _betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta):
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps # t -> t/T
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas)
    
class DiffusionModel(nn.Module):
    """DiffusionModel class that implements a DDPM (denoising diffusion probabilistic model)."""
    def __init__(
            self,
            backbone: nn.Module,
            fwd_diff: ForwardDiffusion,
            img_size: int,
            time_enc_dim: int=256,
            dropout: float=0,
        ) -> None:
        """Constructor of DiffusionModel class.

        Parameters
        ----------
        backbone
            backbone module (instance) for noise estimation
        fwd_diff
            forward diffusion module (instance)
        img_size
            size of (quadratic) images
        time_enc_dim
            feature dimension that should be used for time embedding/encoding
        dropout
            value of dropout layers
        """
        super().__init__()
        self.model = backbone
        self.fwd_diff = fwd_diff
        self.img_size = img_size
        self.time_enc_dim = time_enc_dim
        self.dropout = dropout

        self.time_encoder = PositionalEncoding(d_model=time_enc_dim, dropout=dropout)

    def forward(
            self, 
            x: Float[Tensor, "batch channels height width"]
        ) -> Tuple[Float[Tensor, "batch channels height width"], Float[Tensor, "batch channels height width"]]:
        """Predict noise for single denoising step.

        Parameters
        ----------
        x
            batch of original images
        
        Returns
        -------
        out
            tuple of noise predictions and noise for random timesteps in the denoising process
        """
        timesteps = self._sample_timesteps(x.shape[0], device=x.device)
        if timesteps.dim() != 1:
            raise ValueError("Timesteps should only have batch dimension.", timesteps.shape)
        time_enc = self.time_encoder.get_pos_encoding(timesteps)
        if time_enc.dim() != 2:
            raise ValueError("Time Encoding should be 2 dimensional.", time_enc.shape)
        # make (partially) noisy versions of batch, returns noisy version + applied noise
        x_t, noise = self.fwd_diff(x, timesteps)
        # predict the applied noise from the noisy version
        noise_pred = self.model(x_t, time_enc)
        return noise_pred, noise
    
    def init_noise(self, num_samples: int):
        return torch.randn((num_samples, self.model.in_channels, self.img_size, self.img_size), device=list(self.parameters())[0].device)
    
    def denoise_singlestep(
            self, 
            x: Float[Tensor, "batch channels height width"],
            t: Int64[Tensor, "batch"]
        ) -> Float[Tensor, "batch channels height width"]:
        """Denoise single timestep in reverse direction.

        Parameters
        ----------
        x
            tensor representing a batch of noisy pictures (may be of different timesteps)
        t
            tensor representing the t timesteps for the batch (where the batch now is)

        Returns
        -------
        out
            less noisy version (by one timestep, now at t-1 from the arguments)
        """
        self.model.eval()
        with torch.no_grad():
            t_enc = self.time_encoder.get_pos_encoding(t)
            noise_pred = self.model(x, t_enc)
            alpha = self.fwd_diff.alphas[t][:, None, None, None]
            alpha_hat = self.fwd_diff.alphas_dash[t][:, None, None, None]
            beta = self.fwd_diff.betas[t][:, None, None, None]
            noise = torch.randn_like(x, device=noise_pred.device)
            # noise where t = 1 should be zero
            (t_one_idx, ) = torch.where(t==1)
            noise[t_one_idx] = 0
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * noise_pred) + torch.sqrt(beta) * noise
        self.model.train()
        return x
    
    def sample(
            self,
            num_samples: int
        ) -> Float[Tensor, "batch channel height width"]:
        beta = self.fwd_diff.betas[-1].view(-1,1,1,1)
        x = self.init_noise(num_samples) * torch.sqrt(beta)
        for i in reversed(range(1, self.fwd_diff.timesteps)):
            t = i * torch.ones((num_samples), dtype=torch.long, device=list(self.model.parameters())[0].device)
            x = self.denoise_singlestep(x, t)
        return x

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> Int64[Tensor, "batch"]:
        return torch.randint(low=1, high=self.fwd_diff.timesteps, size=(batch_size,), device=device)