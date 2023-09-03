import torch
from torch import nn, Tensor
from jaxtyping import Float
from typing import Union, Tuple

class VariationalAutoencoder(nn.Module):
    """Class implementing a Variational Autoencoder."""
    def __init__(self) -> None:
        super().__init__()
        pass

class ResNet18Encoder(nn.Module):
    """Class implementing the ResNet encoder.

    For exact details, see He et al: Deep Residual Learning for Image Recognition (2015).

    Implementation
    --------------
    1. Image size initially is the usual ImageNet crop of 224x224.
    2. Channels increased to 64, image size decreased to 56x56, before the repeated residual blocks begin.
    3. We split into 4 submodules, where every submodule consists of 2 residual blocks.
        a. standard residual blocks
        b. first residual block increases channels to 128, halves size with stride 2, second is standard
        c. like b., but to 256 channels
        d. like b., but to 512 channels
    Output of residual blocks has size 7x7 with 512 channels.
    """
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.in_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.resblock1 = nn.Sequential(
            ResNetBlock(64, 64),
            ResNetBlock(64, 64)
        )
        self.resblock2 = nn.Sequential(
            ResNetBlock(64, 128),
            ResNetBlock(128,128)
        )
        self.resblock3 = nn.Sequential(
            ResNetBlock(128, 256),
            ResNetBlock(256, 256)
        )
        self.resblock4 = nn.Sequential(
            ResNetBlock(256, 512),
            ResNetBlock(512, 512)
        )

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> Float[Tensor, "batch channels height width"]:
        for block in [self.in_layer, self.resblock1, self.resblock2, self.resblock3, self.resblock4]:
            x = block(x)
        return x

class ResNetBlock(nn.Module):
    """Class implementing the ResNet Basic Building Block.

    For visualization, see Fig. 2 in He et al: Deep Residual Learning for Image Recognition (2015).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple]=3, stride: int=1,downsampler: nn.Module=None) -> None:
        """Constructor of ResNetBlock.

        Parameters
        ----------
        downsampler
            nn.Module applying downsampling (needed if stride != 0)
        """
        super().__init__()
        if (stride != 1) and (downsampler is None):
            raise ValueError("Choose valid combination of downsampler and stride.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.downsampler = downsampler

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = "same",
            bias = False # REALLY?
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size = kernel_size,
            padding = "same",
            bias = False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> Float[Tensor, "batch channels height width"]:
        if self.downsampler is not None:
            x_0 = self.downsampler(x)
        else:
            x_0 = x

        for layer in [self.conv1, self.bn1, self.activation, self.conv2, self.activation]:
            x = layer(x)

        x = x + x_0
        return self.activation(x)