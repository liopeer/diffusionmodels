import torch
from torch import nn, Tensor
from jaxtyping import Float, Int64, Int
from typing import Literal, Tuple, Union, List
from diffusion_models.models.positional_encoding import PositionalEncoding
import math
from diffusion_models.models.unet import UNet
from diffusion_models.models.openai_unet import UNetModel
from diffusion_models.models.diffusion import ForwardDiffusion
    
class DiffusionModelOpenAI(nn.Module):
    """DiffusionModel class that implements a DDPM (denoising diffusion probabilistic model)."""
    def __init__(
            self,
            backbone: UNet,
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
        x_t, noise = self.fwd_diff(x, timesteps)
        # predict the applied noise from the noisy version
        noise_pred = self.model(x_t, timesteps/self.fwd_diff.timesteps)
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
            # t_enc = self.time_encoder.get_pos_encoding(t)
            noise_pred = self.model(x, t/self.fwd_diff.timesteps)
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
        ) -> Union[Float[Tensor, "batch channel height width"], Tuple]:
        beta = self.fwd_diff.betas[-1].view(-1,1,1,1)
        x = self.init_noise(num_samples) * torch.sqrt(beta)
        intermediates = {}
        for i in reversed(range(1, self.fwd_diff.timesteps)):
            t = i * torch.ones((num_samples), dtype=torch.long, device=list(self.model.parameters())[0].device)
            x = self.denoise_singlestep(x, t)
        return x

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> Float[Tensor, "batch"]:
        return torch.randint(low=1, high=self.fwd_diff.timesteps, size=(batch_size,), device=device)