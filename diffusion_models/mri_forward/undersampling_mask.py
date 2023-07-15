import torch
from torch import nn

class UndersamplingMask(nn.Module):
    def __init__(self, mask_type: str, undersampling_ratio: int, device=None) -> None:
        super().__init__()
        self.mask_type = mask_type
        if self.mask_type == "naive_1d_v":
            self.gen = naive_undersampling1d_v
        elif self.mask_type == "naive_1d_h":
            self.gen = naive_undersampling1d_h
        elif self.mask_type == "naive_2d":
            self.gen = naive_undersampling2d
        self.undersampling_ratio = undersampling_ratio
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def forward(self, x):
        mask = self.gen(x.shape, self.undersampling_ratio)
        mask = normalize_mask(mask)
        x = x * mask.expand(*x.size()).to(self.device)
        return x, mask

class StochasticUndersamplingMask(nn.Module):
    def __init__(self, mask_type: str, rel_sigma: float, undersampling_rate: float, random_seed: int=42, device=None) -> None:
        super().__init__()
        torch.manual_seed(random_seed)
        self.mask_type = mask_type
        self.rel_sigma = rel_sigma
        self.undersampling_rate = undersampling_rate
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if self.mask_type == "gauss_1d_h":
            self.gen = gaussian_kernel1d_h
        elif self.mask_type == "gauss_1d_v":
            self.gen = gaussian_kernel1d_v
        elif self.mask_type == "gauss_2d":
            self.gen = gaussian_kernel2d

    def forward(self, x):
        mask = self.gen(x.shape, self.rel_sigma)
        mask = normalize_mask(mask)
        mean = torch.mean(mask)
        noise = torch.randn_like(mask)
        mask = mask + noise
        mask = mask.ge(mean)
        return x, mask
    
def normalize_mask(mask: torch.Tensor):
    max = torch.max(mask)
    factor = 1. / max
    return mask * factor

def naive_undersampling2d(size: torch.tensor, undersampling_ratio: int) -> torch.Tensor:
    """2D regular subsampling with given undersampling ratio.

    Parameters
    ----------
    size
        shape tensor
    undersampling_ratio
        integer representing the acceleration in both dimensions

    Returns
    -------
    Tensor
        2D tensor with shape of last 2 dim of size
    """
    mask = torch.zeros((size[-2],size[-1]))
    mask[::undersampling_ratio,::undersampling_ratio] = 1
    return mask

def naive_undersampling1d_v(size: torch.tensor, undersampling_ratio: int) -> torch.Tensor:
    """Regular subsampling of vertical lines with given undersampling ratio.

    Parameters
    ----------
    size
        shape tensor
    undersampling_ratio
        integer representing the acceleration in both dimensions

    Returns
    -------
    Tensor
        2D tensor with shape of last 2 dim of size
    """
    mask = torch.zeros((size[-1]))
    mask[::undersampling_ratio] = 1
    return mask.repeat(size[-2], 1)

def naive_undersampling1d_h(size: torch.tensor, undersampling_ratio: int) -> torch.Tensor:
    """Regular subsampling of horizontal lines with given undersampling ratio.

    Parameters
    ----------
    size
        shape tensor
    undersampling_ratio
        integer representing the acceleration in both dimensions

    Returns
    -------
    Tensor
        2D tensor with shape of last 2 dim of size
    """
    mask = torch.zeros((size[-2]))
    mask[::undersampling_ratio] = 1
    return mask.repeat(size[-1], 1).transpose(1,0)

def gaussian_kernel1d_h(size: torch.Tensor, rel_sigma: float) -> torch.Tensor:
    """1D Gaussian kernel repeated along horizontal dimension.

    Parameters
    ----------
    size
        shape tensor
    rel_sigma
        std dev of kernel, if image width/height was 1

    Returns
    -------
    Tensor
        2D tensor with shape of last 2 dim of size
    """
    sig = rel_sigma * size[-2]
    length = torch.linspace(-(size[-2] - 1) / 2, (size[-2] - 1) / 2, size[-2])
    gauss = torch.exp(-0.5 * torch.square(length) / sig**2)
    kernel = gauss / torch.sum(gauss)
    return kernel.repeat(size[-1], 1).transpose(1,0)

def gaussian_kernel1d_v(size: torch.Tensor, rel_sigma: float) -> torch.Tensor:
    """1D Gaussian kernel repeated along vertical dimension.

    Parameters
    ----------
    size
        shape tensor
    rel_sigma
        std dev of kernel, if image width/height was 1

    Returns
    -------
    Tensor
        2D tensor with shape of last 2 dim of size
    """
    sig = rel_sigma * size[-1]
    length = torch.linspace(-(size[-1] - 1) / 2, (size[-1] - 1) / 2, size[-1])
    gauss = torch.exp(-0.5 * torch.square(length) / sig**2)
    kernel = gauss / torch.sum(gauss)
    return kernel.repeat(size[-2], 1)

def gaussian_kernel2d(size: torch.Tensor, rel_sigma: float) -> torch.Tensor:
    """Gaussian kernel as 2D tensor.

    Parameters
    ----------
    size
        shape tensor
    rel_sigma
        std dev of kernel, if image width/height was 1

    Returns
    -------
    Tensor
        2D tensor with shape of last 2 dim of size
    """
    sig_h = rel_sigma * size[-2]
    sig_w = rel_sigma * size[-1]
    height = torch.linspace(-(size[-2] - 1) / 2, (size[-2] - 1) / 2, size[-2])
    width = torch.linspace(-(size[-1] - 1) / 2, (size[-1] - 1) / 2, size[-1])
    gauss_h = torch.exp(-0.5 * torch.square(height) / sig_h**2)
    gauss_w = torch.exp(-0.5 * torch.square(width) / sig_w**2)
    kernel = torch.outer(gauss_h, gauss_w)
    return kernel / torch.sum(kernel)