import torch
from torch import nn

class ForwardDiffusion(nn.Module):
    def __init__(self, timesteps: int, start: float=0.0001, end: float=0.02, random_seed: int=42, type="linear", device=None) -> None:
        self.timesteps = timesteps
        self.start = start
        self.end = end
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        self.type = type
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if self.type == "linear":
            self.betas = self.linear_schedule(timesteps=self.timesteps, start=self.start, end=self.end)
        elif self.type == "cosine":
            raise NotImplementedError("Cosine scheduler not implemented yet.")
        else:
            raise NotImplementedError("Invalid scheduler option:", type)
        self.alphas = 1. - self.betas
        self.alphas_dash = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_dash = torch.sqrt(self.alphas_dash)
        self.sqrt_one_minus_alpha_dash = torch.sqrt(1. - self.alphas_dash)

    def forward(self, x_0: torch.Tensor, t: int):
        noise_normal = torch.randn_like(x_0).to(self.device)
        sqrt_alpha_dash_t = self.sqrt_alphas_dash[t].to(self.device)
        sqrt_one_minus_alpha_dash_t = self.sqrt_one_minus_alpha_dash[t].to(self.device)
        if x_0.device != self.device:
            x_0 = x_0.to(self.device)
        x_t = sqrt_alpha_dash_t * x_0 + sqrt_one_minus_alpha_dash_t * noise_normal
        return x_t

    def linear_schedule(self, timesteps, start, end):
        return torch.linspace(start, end, timesteps)
    
    def cosine_scheduler(self, timesteps, start, end):
        pass