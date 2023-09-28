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
            self.init_betas = self._linear_scheduler(timesteps=self.timesteps, start=self.start, end=self.end)
        elif self.type == "cosine":
            self.init_betas = self._cosine_scheduler(timesteps=self.timesteps, start=self.start, end=self.end)
        else:
            raise NotImplementedError("Invalid scheduler option:", type)
        self.init_alphas = 1. - self.init_betas

        self.register_buffer("alphas", self.init_alphas, persistent=False)
        self.register_buffer("betas", self.init_betas, persistent=False)
        self.register_buffer("alphas_dash", torch.cumprod(self.alphas, axis=0), persistent=False)
        self.register_buffer("sqrt_alphas_dash", torch.sqrt(self.alphas_dash), persistent=False)
        self.register_buffer("sqrt_one_minus_alpha_dash", 1. - self.alphas_dash, persistent=False)

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
        sqrt_alpha_dash_t = self.sqrt_alphas_dash[t]
        sqrt_one_minus_alpha_dash_t = self.sqrt_one_minus_alpha_dash[t]
        x_t = sqrt_alpha_dash_t.view(-1, 1, 1, 1) * x_0
        x_t += sqrt_one_minus_alpha_dash_t.view(-1, 1, 1, 1) * noise_normal
        return x_t, noise_normal

    def _linear_scheduler(self, timesteps, start, end):
        return torch.linspace(start, end, timesteps)
    
    def _cosine_scheduler(self, timesteps, start, end):
        return self._betas_for_alpha_bar(
            timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    def _betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                        produces the cumulative product of (1-beta) up to that
                        part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                        prevent singularities.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
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
        time_enc = self.time_encoder.get_pos_encoding(timesteps)
        x_t, noise = self.fwd_diff(x, timesteps)
        noise_pred = self.model(x_t, time_enc)
        return noise_pred, noise
    
    def sample(
            self, 
            num_samples: int, 
            debugging: bool=False,
            save_every: int=20
        ) -> Union[Float[Tensor, "batch channel height width"], List[Float[Tensor, "batch channel height width"]]]:
        """Sample a batch of images.

        Parameters
        ----------
        num_samples
            how big the batch should be
        debugging
            if true, returns list that shows the sampling process
        save_every
            defines how often the tensors should be saved in the denoising process

        Returns
        -------
        out
            either a list of tensors if debugging is true, else a single tensor with final images
        """
        self.model.eval()
        device = list(self.parameters())[0].device
        with torch.no_grad():
            x = torch.randn((num_samples, self.model.in_channels, self.img_size, self.img_size), device=device)
            x_list = []
            for i in reversed(range(1, self.fwd_diff.timesteps)):
                t_step = i * torch.ones((num_samples), dtype=torch.long, device=device)
                t_enc = self.time_encoder.get_pos_encoding(t_step)
                noise_pred = self.model(x, t_enc)

                alpha = self.fwd_diff.alphas[t_step][:, None, None, None]
                alpha_hat = self.fwd_diff.alphas_dash[t_step][:, None, None, None]
                beta = self.fwd_diff.betas[t_step][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x, device=device)
                else:
                    noise = torch.zeros_like(x, device=device)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * noise_pred) + torch.sqrt(beta) * noise
                if debugging and (i % save_every == 0):
                    x_list.append(x)
        self.model.train()
        if debugging:
            return x_list
        return x

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> Int64[Tensor, "batch"]:
        return torch.randint(low=1, high=self.fwd_diff.timesteps, size=(batch_size,), device=device)