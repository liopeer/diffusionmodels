import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()