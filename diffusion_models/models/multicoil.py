import torch
from torch import nn
from jaxtyping import Float
from typing import List
from torch import Tensor

class MultiCoilConv2d(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(*args, **kwargs)

    def forward(self, x: Float[Tensor, "batch coils in_channels height width"]) -> Float[Tensor, "batch coils out_channels height width"]:
        orig_shape = x.shape
        x = self.conv2d(x.view(-1, *orig_shape[-3:]))
        return x.view(*orig_shape)
    
class MultiCoilReducer(nn.Module):
    def __init__(self, channel_factors: List[int]=(4, 8, 16, 32), kernel_size: int=3) -> None:
        """Constructor of MultiCoilReducer Class.

        This class takes every coil independently (treats them like a sub-fraction of a batch), increases the channel size
        massively (from 2 initial channels for complex k-space data) via several convolutional layers and then averages
        those channels over the coil dimension. Averaging is invariant to permutations of the input order, so the coil order
        or the number of coils will not matter anymore. Inspiration was drawn from point cloud processing [1]_, see below.

        .. [1] Qi et al., PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation, 2017

        Parameters
        ----------
        channel_factors
            sequence that includes all factors for channel increases
        kernel_size
            kernel size for conv layers
        """
        super().__init__()
        layers = [MultiCoilConv2d(in_channels=2*i, out_channels=2(i+1), kernel_size=kernel_size, padding="same") for i in channel_factors]

    def forward(self, x: Float[Tensor, "batch coils 2 height width"]) -> Float[Tensor, "batch out_channels height width"]:
        pass