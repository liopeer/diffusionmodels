import torch.nn as nn
import torch
from type_hints.torch import *
from typing import Union

def dummy_func(dummy_arg: Union[str, int]):
    pass

class EncodingBlock(nn.Module):
    def __init__(self, in_channels: int, 
                 out_channels: int, 
                 time_embedding_size: int, 
                 kernel_size: int=3) -> None:
        """Initialize UNet Encoder Building Block.

        Parameters
        ---------
        in_channels
            number of input channels
        out_channels
            number of output channels
        time_embedding_size
            dimension of time embedding
        kernel_size
            size of convolutional kernel
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_size = time_embedding_size
        self.kernel_size = kernel_size

        self.time_embedding_fc = nn.Sequential(
            nn.Linear(self.time_embedding_size, self.out_channels),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_siez=self.kernel_size, padding="same"),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )
        self.scale = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor(Batch, Channels, Height, Width), time_embedding: Tensor(Features)) -> Tensor(Batch, Channels, Height, Width):
        """Forward pass of UNet Encoder Building Block.

        Parameters
        ---------
        x
            input tensor
        time_embedding
            time embedding tensor

        Returns
        -------
            output tensor
        """
        time_embedding = self.time_embedding_fc(time_embedding)
        time_embedding = time_embedding[(..., ) + (None, ) * 2]

        x = self.conv1(x)
        x = x + time_embedding
        x = self.conv2(x)
        x = self.scale(x)
        return x

class DecodingBlock(EncodingBlock):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 time_embedding_size: int,
                 kernel_size: int=3) -> None:
        """Initialize UNet Decoder Building Block.

        Parameters
        ---------
        in_channels
            number of input channels
        out_channels
            number of output channels
        time_embedding_size
            dimension of time embedding
        kernel_size
            size of convolutional kernel
        """
        super().__init__(in_channels, out_channels, time_embedding_size, kernel_size)
        self.scale = nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor(Batch, Channels, Height, Width), time_embedding: Tensor(Features)) -> Tensor(Batch, Channels, Height, Width):
        """Forward pass of UNet Decoder Building Block.

        Parameters
        ---------
        x
            input tensor
        time_embedding
            time embedding tensor

        Returns
        -------
            output tensor
        """
        return super().forward(x, time_embedding)

class UNet(nn.Module):
    def __init__(self, num_layers) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.encoding_channels = [64 * (2 ** i) for i in range(self.num_layers)]
        self.decoding_channels = self.encoding_channels[::-1]

    def forward(self, x):
        return x