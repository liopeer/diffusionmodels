from jaxtyping import Float, Complex
from torch import Tensor
import torch

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def bytes_to_gb(bytes: int):
    kb = bytes / 1024
    mb = kb / 1024
    gb = mb / 1024
    return gb

def complex_to_2channelfloat(x: Complex[Tensor, "*batch height width"]) -> Float[Tensor, "*batch 2 height width"]:
    x = torch.view_as_real(x)
    return x.permute(*[i for i in range(x.dim()-2)],-1,-3,-2)