import torch.nn as nn
import torch
from typing import Union, Tuple, Literal
from jaxtyping import Float
from torch import Tensor

class EncodingBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            time_embedding_size: int, 
            kernel_size: int=3,
            dropout: float=0.5,
            verbose: bool=False
        ) -> None:
        """Initialize UNet Encoder Building Block.

        Parameters
        ----------
        in_channels
            number of input channels
        out_channels
            number of output channels
        time_embedding_size
            dimension of time embedding
        kernel_size
            size of convolutional kernel
        dropout
            probability of dropout layers
        verbose
            whether to print tensor shapes in forward pass
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_size = time_embedding_size
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.time_embedding_fc = nn.Linear(self.time_embedding_size, self.out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.scale = nn.MaxPool2d(kernel_size=2)

    def forward(
            self, 
            x: Float[Tensor, "batch channels height width"],
            time_embedding: Float[Tensor, "batch embedding_size"]
        ) -> Tuple[Float[Tensor, "batch double_channels half_height half_width"], Float[Tensor, "batch double_channels height width"]]:
        """Forward pass of UNet Encoder Building Block.

        Parameters
        ---------
        x
            input tensor
        time_embedding
            time embedding tensor

        Returns
        -------
        output
            convoluted and downscaled tensor
        skip
            convoluted but non-downscaled tensor for skip connection
        """
        print(self.conv1[0].weight.device, x.device)
        assert False
        x = self.conv1(x)
        if time_embedding is not None:
            time_embedding = self.time_embedding_fc(time_embedding)
            time_embedding = time_embedding.view(time_embedding.shape[0], time_embedding.shape[1], 1, 1)
            x = x + time_embedding.expand(time_embedding.shape[0], time_embedding.shape[1], x.shape[-2], x.shape[-1])
        x_skip = self.conv2(x)
        x = self.scale(x_skip)
        return x, x_skip

class DecodingBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            out_channels: int,
            time_embedding_size: int,
            kernel_size: int=3,
            dropout: float=0.5,
            verbose: bool=False
        ) -> None:
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
        dropout
            dropout probability of dropout layers
        verbose
            whether to print tensor shapes during forward pass
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_size = time_embedding_size
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.verbose = verbose

        self.time_embedding_fc = nn.Linear(self.time_embedding_size, self.out_channels)

        self.scale = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.out_channels * 2, self.out_channels, kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
    
    def forward(
            self, 
            x: Float[Tensor, "batch channels height width"],
            skip: Float[Tensor, "batch half_channels double_height double_width"],
            time_embedding: Float[Tensor, "batch embedding_size"]=None
        ) -> Float[Tensor, "batch half_channels double_height double_width"]:
        """Forward pass of UNet Decoder Building Block.

        Parameters
        ---------
        x
            input tensor
        skip
            skip connection to be merged
        time_embedding
            time embedding tensor

        Returns
        -------
            output tensor
        """
        if self.verbose:
            print(f"Decoder Input: {x.shape}\tSkip: {skip.shape}")

        x = self.scale(x)

        if self.verbose:
            print(f"After Scaling: {x.shape}")

        x = torch.cat([x, skip], dim=1)

        if self.verbose:
            print(f"After Concat {x.shape}")

        x = self.conv1(x)

        if time_embedding is not None:
            time_embedding = self.time_embedding_fc(time_embedding)
            time_embedding = time_embedding.view(time_embedding.shape[0], time_embedding.shape[1], 1, 1)
            x = x + time_embedding.expand(time_embedding.shape[0], time_embedding.shape[1], x.shape[-2], x.shape[-1])

        if self.verbose:
            print(f"After Conv1: {x.shape}")

        x = self.conv2(x)

        if self.verbose:
            print(f"After Conv2: {x.shape}")
        return x

class UNet(nn.Module):
    """Implementation of UNet architecture, close to original paper.

    Things that are different
    -------------------------
    - transpose convolution instead of upsampling
    - "same" padding for convolutions, so that output image has same size as input
    - flexibility to go to larger depths than original implementation
    - separate additional convolutional layer to go from the input to 64 channels
    - possibility to inject conditioning (time/position encodings and more) into the network

    Kernel sizes
    ------------
    Make sure to use odd kernel sizes.

    Images sizes
    ------------
    Image height and width should both still be even at the bottleneck layer. This means they should be divisible by 2, num_encoder+1 times
    and still yield an even number at this point. Best to just stick with powers of 2.

    References
    ----------
    .. [1] [Ronneberger15] O. Ronneberger, P. Fischer, and T. Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation
    """
    def __init__(
            self, 
            num_encoding_blocks,
            in_channels: int=3,
            kernel_size: int=3,
            time_emb_size: int=256,
            dropout: float=0.5,
            verbose: bool=False
        ) -> None:
        super().__init__()
        self.num_layers = num_encoding_blocks
        self.in_channels = in_channels
        if kernel_size % 2 != 1:
            raise ValueError("Choose odd kernel size.")
        self.kernel_size = kernel_size
        self.time_embedding_size = time_emb_size
        self.dropout = dropout
        self.verbose = verbose

        self.in_conv = nn.Conv2d(in_channels, 64, kernel_size=kernel_size, padding="same")

        self.encoding_channels = [64]
        for i in range(self.num_layers):
            self.encoding_channels.append(64 * (2 ** i))
        self.encoder = nn.ModuleList([EncodingBlock(self.encoding_channels[i], self.encoding_channels[i+1], time_emb_size, kernel_size, dropout, verbose) for i in range(len(self.encoding_channels[:-1]))])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.encoding_channels[-1], self.encoding_channels[-1] * 2, kernel_size=self.kernel_size, padding="same"),
            nn.Conv2d(self.encoding_channels[-1] * 2, self.encoding_channels[-1] * 2, kernel_size=self.kernel_size, padding="same")
        )

        self.decoding_channels = []
        for i in range(self.num_layers + 1):
            self.decoding_channels.append(64 * (2 ** i))
        self.decoding_channels = self.decoding_channels[::-1]
        self.decoder = nn.ModuleList([DecodingBlock(self.decoding_channels[i], self.decoding_channels[i+1], time_emb_size, kernel_size, dropout, verbose) for i in range(len(self.encoding_channels[:-1]))])

        self.out_conv = nn.Conv2d(64, in_channels, kernel_size=kernel_size, padding="same")

    def forward(
            self, 
            x: Float[Tensor, "batch channels height width"], 
            t: Float[Tensor, "batch embedding_size"]=None
        ) -> Float[Tensor, "batch channels height width"]:
        """Forward Method of UNet class.

        Parameters
        ----------
        x
            input image batch
        t
            time embedding

        Returns
        -------
        out
            output image batch
        """
        if self.verbose:
            print("Encoding Channels", self.encoding_channels, "\tDecoding Channels", self.decoding_channels)
        if not self._check_sizes(x):
            raise ValueError("Choose appropriate image size.")

        # in_layer - to 64 channels
        x = self.in_conv(x)

        # Encoder
        skips = []
        if self.verbose:
            print("Encoder Starting...")
        for i, block in enumerate(self.encoder):
            x, skip = block(x, t)
            if self.verbose:
                print(f"Output Encoder{i}\tx: {x.shape}\tskip: {skip.shape}")
            skips.append(skip)
        
        # Bottleneck
        if self.verbose:
            print("x before bottleneck", x.shape)
        x = self.bottleneck(x)
        if self.verbose:
            print("x after bottleneck", x.shape)

        # Decoder
        skips = skips[::-1]
        for i, block in enumerate(self.decoder):
            if self.verbose:
                print(f"Input into Decoder{i}\tx: {x.shape}\tskip: {skips[i].shape}")
            x = block(x, skips[i], t)

        # out_layer - from 64 channels back
        x = self.out_conv(x)
        return x
    
    def _check_sizes(self, x):
        width, height = x.shape[-1], x.shape[-2]
        widths = [width/(2**i) for i in range(self.num_layers + 1)]
        heights = [height/(2**i) for i in range(self.num_layers + 1)]
        widths = [(elem.is_integer() and (elem % 2 == 0)) for elem in widths]
        heights = [(elem.is_integer() and (elem % 2 == 0)) for elem in heights]
        if (False in widths) or (False in heights):
            return False
        return True