import torch
from torch import nn, Tensor
from jaxtyping import Float

class VariationalAutoencoder(nn.Module):
    """Class implementing a Variational Autoencoder."""
    def __init__(self) -> None:
        super().__init__()
        pass