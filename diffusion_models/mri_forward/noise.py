from typing import Tuple
import torch

def apply_add_gauss_noise(tensor: torch.tensor, sigma: float):
    """Applies additive noise."""
    return tensor + independent_gauss_noise(tensor.shape, sigma)

def apply_mult_gauss_noise(tensor: torch.tensor, sigma: float):
    """Applies multiplicative noise."""
    return tensor * independent_gauss_noise(tensor.shape, sigma)

def independent_gauss_noise(size: Tuple[int], sigma: float):
    """Create independent Gaussian noise tensor."""
    return sigma * torch.randn(size)