import torch.nn as nn
import torch
from typing import Union, Tuple, Literal
from jaxtyping import Float
from torch import Tensor
import math
from einops import rearrange

class EncodingBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            time_embedding_size: int, 
            kernel_size: int=3,
            dropout: float=0.5,
            activation: nn.Module=nn.SiLU,
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
        activation
            non-linearity of neural network
        verbose
            whether to print tensor shapes in forward pass
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_size = time_embedding_size
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.activation = activation
        self.verbose = verbose

        self.time_embedding_fc = nn.Linear(self.time_embedding_size, self.out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(self.out_channels),
            self.activation(),
            nn.Dropout(self.dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(self.out_channels),
            self.activation(),
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
            activation: nn.Module=nn.SiLU,
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
        self.activation = activation
        self.verbose = verbose

        self.time_embedding_fc = nn.Linear(self.time_embedding_size, self.out_channels)

        self.scale = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.out_channels),
            self.activation(),
            nn.Dropout(self.dropout)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.out_channels * 2, self.out_channels, kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(self.out_channels),
            self.activation(),
            nn.Dropout(self.dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(self.out_channels),
            self.activation(),
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
            time_embedding = time_embedding[..., None, None]
            x = x + time_embedding

        if self.verbose:
            print(f"After Conv1: {x.shape}")

        x = self.conv2(x)

        if self.verbose:
            print(f"After Conv2: {x.shape}")
        return x

class UNet(nn.Module):
    """Implementation of UNet architecture, close to original paper. [1]_

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
            num_encoding_blocks: int,
            in_channels: int=3,
            kernel_size: int=3,
            time_emb_size: int=256,
            dropout: float=0.5,
            activation: nn.Module=nn.SiLU,
            verbose: bool=False,
            init_channels: int=64,
            attention: bool=True,
            attention_heads: int=4,
            attention_ff_dim: int=None
        ) -> None:
        """Constructor of UNet.

        Parameters
        ----------
        num_encoding_blocks
            how many basic encoder building blocks; each block will double the channels and half the resolution
        in_channels
            start channels, e.g. 1 or 3
        kernel_size
            size of convolutional kernels
        time_emb_size
            initial size of time step encoding
        dropout
            probability parameter of dropout layers
        activation
            activation function to be used
        verbose
            verbose printing of tensor shapes for debbugging
        init_channels
            number of channels to initially transform the input to (usually 64, 128, ...)
        attention
            whether to use self-attention layers
        attention_heads
            number of attention heads to be used
        attention_ff_dim
            hidden dimension of feedforward layer in self attention module, None defaults to input dimension
        """
        super().__init__()
        self.num_layers = num_encoding_blocks
        self.in_channels = in_channels
        if kernel_size % 2 != 1:
            raise ValueError("Choose odd kernel size.")
        self.kernel_size = kernel_size
        self.time_embedding_size = time_emb_size
        self.dropout = dropout
        self.activation = activation
        self.verbose = verbose
        self.init_channels = init_channels
        self.attention = attention
        self.attention_heads = attention_heads
        self.attention_ff_dim = attention_ff_dim

        self.encoding_channels, self.decoding_channels = self._get_channel_lists(init_channels, num_encoding_blocks)

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.encoding_channels[0], kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(init_channels),
            self.activation(),
            nn.Dropout(self.dropout)
        )

        if attention:
            self.encoder = nn.ModuleList([AttentionEncodingBlock(self.encoding_channels[i], self.encoding_channels[i+1], time_emb_size, kernel_size, dropout, self.activation, verbose, attention_heads, attention_ff_dim) for i in range(len(self.encoding_channels[:-1]))])
        else:
            self.encoder = nn.ModuleList([EncodingBlock(self.encoding_channels[i], self.encoding_channels[i+1], time_emb_size, kernel_size, dropout, self.activation, verbose) for i in range(len(self.encoding_channels[:-1]))])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.encoding_channels[-1], self.encoding_channels[-1] * 2, kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(self.encoding_channels[-1] * 2),
            self.activation(),
            nn.Dropout(self.dropout),
            nn.Conv2d(self.encoding_channels[-1] * 2, self.encoding_channels[-1] * 2, kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(self.encoding_channels[-1] * 2),
            self.activation(),
            nn.Dropout(self.dropout)
        )

        if attention:
            self.decoder = nn.ModuleList([AttentionDecodingBlock(self.decoding_channels[i], self.decoding_channels[i+1], time_emb_size, kernel_size, dropout, self.activation, verbose, attention_heads, attention_ff_dim) for i in range(len(self.encoding_channels[:-1]))])
        else:
            self.decoder = nn.ModuleList([DecodingBlock(self.decoding_channels[i], self.decoding_channels[i+1], time_emb_size, kernel_size, dropout, self.activation, verbose) for i in range(len(self.encoding_channels[:-1]))])
        
        self.out_conv = nn.Conv2d(init_channels, in_channels, kernel_size=kernel_size, padding="same")

    def _get_channel_lists(self, start_channels, num_layers):
        if not math.log2(start_channels).is_integer():
            raise ValueError("Choose power of 2 as number of start channels (e.g. 64, 128, ...).")
        
        encoding_channels = [start_channels]
        for i in range(num_layers):
            encoding_channels.append(start_channels * (2 ** i))

        decoding_channels = []
        for i in range(num_layers + 1):
            decoding_channels.append(start_channels * (2 ** i))

        return encoding_channels, decoding_channels[::-1]

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
        if x.dim() != 4:
            raise ValueError("Image data should be 4 dimensional.", x.shape)
        if t.dim() != 2:
            raise ValueError("Time embedding must have dimension 2.", t.shape)
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
    
class SelfAttention(nn.Module):
    def __init__(
            self, 
            channels: int,
            num_heads: int,
            dropout: float,
            dim_feedforward: int=None,
            activation: nn.Module=nn.SiLU
        ) -> None:
        """Constructor of SelfAttention module.
        
        Implementation of self-attention layer for image data.

        Parameters
        ----------
        channels
            number of input channels
        num_heads
            number of desired attention heads
        dropout
            dropout probability value
        dim_feedforward
            dimension of hidden layers in feedforward NN, defaults to number of input channels
        activation
            activation function to be used, as uninstantiated nn.Module
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.dropout = dropout
        if dim_feedforward is not None:
            self.dim_feedforward = dim_feedforward
        else:
            self.dim_feedforward = channels
        self.activation = activation()
        self.attention_layer = nn.TransformerEncoderLayer(
            channels,
            num_heads,
            self.dim_feedforward,
            dropout,
            self.activation,
            batch_first=True,
            norm_first=True
        )

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> Float[Tensor, "batch channels height width"]:
        """Forward method of SelfAttention module.
        
        Parameters
        ----------
        x
            input tensor
        
        Returns
        -------
        out
            output tensor
        """
        # transform feature maps into vectors and put feature dimension (channels) at the end
        orig_size = x.size()
        x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]).swapaxes(1,2)
        x = self.attention_layer(x)
        return x.swapaxes(1,2).view(*orig_size)
    
class AttentionEncodingBlock(EncodingBlock):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            time_embedding_size: int, 
            kernel_size: int = 3, 
            dropout: float = 0.5, 
            activation: nn.Module = nn.SiLU, 
            verbose: bool = False,
            attention_heads: int=4,
            attention_ff_dim: int=None
        ) -> None:
        super().__init__(in_channels, out_channels, time_embedding_size, kernel_size, dropout, activation, verbose)
        self.sa = SelfAttention(out_channels, attention_heads, dropout, attention_ff_dim, activation)

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tuple[Tensor, Tensor]:
        out, skip = super().forward(x, time_embedding)
        return self.sa(out), skip

class AttentionDecodingBlock(DecodingBlock):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            time_embedding_size: int, 
            kernel_size: int = 3, 
            dropout: float = 0.5, 
            activation: nn.Module = nn.SiLU, 
            verbose: bool = False,
            attention_heads: int=4,
            attention_ff_dim: int=None
        ) -> None:
        super().__init__(in_channels, out_channels, time_embedding_size, kernel_size, dropout, activation, verbose)
        self.sa = SelfAttention(out_channels, attention_heads, dropout, attention_ff_dim, activation)

    def forward(self, x: Tensor, skip: Tensor, time_embedding: Tensor = None) -> Tensor:
        out = super().forward(x, skip, time_embedding)
        return self.sa(out)