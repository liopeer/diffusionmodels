import torch
from torch import nn, Tensor
from jaxtyping import Float
from typing import Union, Tuple

class VariationalAutoencoder(nn.Module):
    """Class implementing a Variational Autoencoder."""
    def __init__(self, in_channels: int=3, hidden_dim: int=256) -> None:
        super().__init__()
        self.encoder = ResNet18Encoder(in_channels, hidden_dim)
        self.decoder = ResNet18Decoder(hidden_dim//2, in_channels)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        hidden = self.reparameterize(mu, sigma)
        return self.decoder(hidden)
    
    def reparameterize(self, mu, sigma):
        sigma = torch.exp(sigma)
        std_normal_noise = torch.randn_like(sigma)
        return std_normal_noise * sigma + mu

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
    def __init__(self, in_channels: int, hidden_dim: int=256) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
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
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        if hidden_dim != 256:
            self.linear = nn.Linear(512, hidden_dim*2)

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> Tuple[Float[Tensor, "batch hidden_dim"], Float[Tensor, "batch hidden_dim"]]:
        for block in [self.in_layer, self.resblock1, self.resblock2, self.resblock3, self.resblock4]:
            x = block(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        if self.hidden_dim != 256:
            x = self.linear(x)
        mu, sigma = x[:, :self.hidden_dim], x[:, self.hidden_dim:]
        return mu, sigma

class ResNetBlock(nn.Module):
    """Class implementing the ResNet Basic Building Block, currently limited to usage in ResNet18 and ResNet34.

    For visualization, see Fig. 2 in He et al: Deep Residual Learning for Image Recognition (2015).
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Constructor of ResNetBlock."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels != out_channels:
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size = 3,
                stride = 2,
                padding = 1
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size = 3,
            padding = 1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

        if in_channels != out_channels:
            self.downsampler = SkipDownSampler(in_channels, out_channels)

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> Union[Float[Tensor, "batch channels height width"], Float[Tensor, "batch channels*2 height/2 width/2"]]:
        if self.in_channels != self.out_channels:
            x_0 = self.downsampler(x)
        else:
            x_0 = x
        for layer in [self.conv1, self.bn1, self.activation, self.conv2, self.bn2]:
            x = layer(x)
        x = x + x_0
        return self.activation(x)
    
class SkipDownSampler(nn.Module):
    """Class SkipDownSampler."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        return self.conv(x)
    
class ResNet18Decoder(nn.Module):
    def __init__(self, hidden_dim: int=256, out_channels: int=3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_channels = out_channels
        if hidden_dim != 256:
            self.linear = nn.Linear(hidden_dim*2, 512)
        self.inv_avgpool = nn.Upsample(scale_factor=7)
        self.resblock1 = nn.Sequential(
            ResNetDecoderBlock(512, 512),
            ResNetDecoderBlock(512, 256)
        )
        self.resblock2 = nn.Sequential(
            ResNetDecoderBlock(256, 256),
            ResNetDecoderBlock(256, 128)
        )
        self.resblock3 = nn.Sequential(
            ResNetDecoderBlock(128, 128),
            ResNetDecoderBlock(128, 64)
        )
        self.resblock4 = nn.Sequential(
            ResNetDecoderBlock(64, 64),
            ResNetDecoderBlock(64, 64)
        )
        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x: Tuple[Float[Tensor, "batch hidden_dim"], Float[Tensor, "batch hidden_dim"]]) -> Float[Tensor, "batch channels height width"]:
        if self.hidden_dim != 256:
            x = self.linear(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = self.inv_avgpool(x)
        for block in [self.resblock1, self.resblock2, self.resblock3, self.resblock4]:
            x = block(x)
        x = self.out_layer(x)
        return x
    
class ResNetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Constructor of ResNetDecoderBlock.

        Inspired by ResNet18/ResNet34, using strided transpose convolutions.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels != out_channels:
            self.conv1 = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size = 4,
                stride = 2,
                padding = 1
            )
        else:
            self.conv1 = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ConvTranspose2d(
            out_channels,
            out_channels,
            kernel_size = 3,
            padding = 1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

        if in_channels != out_channels:
            self.upsampler = SkipUpSampler(in_channels, out_channels)

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> Union[Float[Tensor, "batch channels height width"], Float[Tensor, "batch channels*2 height/2 width/2"]]:
        if self.in_channels != self.out_channels:
            x_0 = self.upsampler(x)
        else:
            x_0 = x
        for layer in [self.conv1, self.bn1, self.activation, self.conv2, self.bn2]:
            x = layer(x)
        x = x + x_0
        return self.activation(x)
    
class SkipUpSampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x