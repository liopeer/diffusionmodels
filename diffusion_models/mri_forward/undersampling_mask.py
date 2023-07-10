import torch

class UndersamplingMask():
    def __init__(self, random_seed) -> None:
        pass

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