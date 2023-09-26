import torch
from torch import nn, Tensor
from jaxtyping import Float, Int64, Int
from typing import Literal, Tuple
from positional_encoding import PositionalEncoding

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
        self.noise_normal = torch.randn_like(x_0)
        if True in torch.gt(t, self.timesteps-1):
            raise IndexError("t ({}) chosen larger than max. available t ({})".format(t, self.timesteps-1))
        sqrt_alpha_dash_t = self.sqrt_alphas_dash[t]
        sqrt_one_minus_alpha_dash_t = self.sqrt_one_minus_alpha_dash[t]
        x_t = sqrt_alpha_dash_t.view(-1, 1, 1, 1) * x_0
        x_t += sqrt_one_minus_alpha_dash_t.view(-1, 1, 1, 1) * self.noise_normal
        return x_t, self.noise_normal

    def _linear_scheduler(self, timesteps, start, end):
        return torch.linspace(start, end, timesteps)
    
    def _cosine_scheduler(self, timesteps, start, end):
        raise NotImplementedError("Cosine scheduler not implemented yet.")
    
class DiffusionModel(nn.Module):
    """DiffusionModel class that implements a DDPM (denoising diffusion probabilistic model)."""
    def __init__(
            self,
            backbone: nn.Module,
            fwd_diff: ForwardDiffusion,
            time_enc_dim: int=256,
            dropout: float=0
        ) -> None:
        """"""
        super().__init__()
        self.model = backbone
        self.fwd_diff = fwd_diff
        self.time_enc_dim = time_enc_dim
        self.dropout = dropout

        self.time_enc = PositionalEncoding(d_model=time_enc_dim, dropout=dropout)

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
        self.timesteps = self._sample_timesteps(x.shape[0]).to(x.device)
        self.time_enc = self.time_enc.get_pos_encoding(self.timesteps)
        x_t, noise = self.fwd_diff(x, self.timesteps)
        noise_pred = self.model(x_t, self.time_enc)
        return noise_pred, noise
    
    def sample(self, n: int, img_size: Tuple[int, int]):
        """Sample a batch of images."""
        self.model.eval()
        device = list(self.parameters())[0].device
        with torch.no_grad():
            x = torch.randn((n, self.model.in_channels, img_size, img_size), device=device)
            for i in reversed(range(1, self.fwd_diff.timesteps)):
                t_steps = torch.ones((n), dtype=torch.long, device=device)
                t_enc = self._time_encoding(t_steps, self.time_enc_dim)
                predicted_noise = self.model(x, t_enc)

                self.fwd_diff.alphas = self.fwd_diff.alphas.to(device)
                self.fwd_diff.alphas_dash = self.fwd_diff.alphas_dash.to(device)
                self.fwd_diff.betas = self.fwd_diff.betas.to(device)

                alpha = self.fwd_diff.alphas[t_steps][:, None, None, None]
                alpha_hat = self.fwd_diff.alphas_dash[t_steps][:, None, None, None]
                beta = self.fwd_diff.betas[t_steps][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x, device=device)
                else:
                    noise = torch.zeros_like(x, device=device)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        self.model.train()
        return x
    
    def _time_encoding(
            self, 
            t: Int[Tensor, "batch"], 
            channels: int
        ) -> Float[Tensor, "batch time_enc_dim"]:
        t = t.unsqueeze(-1).type(torch.float)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        inv_freq = inv_freq.to(t.device)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def _sample_timesteps(self, batch_size: int) -> Int64[Tensor, "batch"]:
        return torch.randint(low=1, high=self.fwd_diff.timesteps, size=(batch_size,))