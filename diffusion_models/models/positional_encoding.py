import torch
from torch import nn
from torch import Tensor
import math
from jaxtyping import Float, Int, Int64

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int=256, dropout: float = 0.1, max_len: int = 5000):
        """Constructor of PositionalEncoding class.

        Parameters
        ----------
        d_model
            feature dimensionality of the model
        dropout
            probability value of dropout layers
        max_len
            maximum length of sequence (shorter setting will free up GPU memory)
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def get_pos_encoding(self, t: Int64[Tensor, "batch"]) -> Float[Tensor, "batch features"]:
        """Get positional encoding for position/timestep t.

        Parameters
        ----------
        t
            timesteps to get positional encoding for (one batch)
        
        Returns
        -------
        out
            positional encodings for batch
        """
        x = self.pe[t]
        return x.squeeze()

    def forward(self, x: Float[Tensor, "length batch features"]) -> Float[Tensor, "length batch features"]:
        """
        Parameters
        ----------
        x
            input sequence

        Returns
        -------
        output
            output sequence with added positional encoding
        """
        x = x + self.pe[:x.shape[0]]
        return self.dropout(x)
    

class PositionalEncoding2D(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass