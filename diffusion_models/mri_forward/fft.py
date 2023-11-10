from torch.fft import fftn, ifftn, ifftshift, fftshift
from typing import Union
from jaxtyping import Float, Complex
from torch import Tensor
import torch
from utils.helpers import complex_to_2channelfloat

def to_kspace(
        x: Union[
            Float[Tensor, "*batch 2 height width"], 
            Complex[Tensor, "*batch height width"]
        ]
    ) -> Union[Float[Tensor, "*batch 2 height width"], Complex[Tensor, "*batch height width"]]:
    if torch.is_complex(x):
        x = fftn(x, dim=(-2,-1))
        return fftshift(x, dim=(-2,-1))
    else:
        x = torch.view_as_complex(x.permute(0,2,3,1))
        x = fftn(x, dim=(-2,-1))
        x = fftshift(x, dim=(-2,-1))
        return complex_to_2channelfloat(x)
    
def to_imgspace(
        x: Union[
            Float[Tensor, "*batch 2 height width"], 
            Complex[Tensor, "*batch height width"]
        ]
    ) -> Union[Float[Tensor, "*batch 2 height width"], Complex[Tensor, "*batch height width"]]:
    if torch.is_complex(x):
        x = ifftn(x, dim=(-2,-1))
        return ifftshift(x, dim=(-2,-1))
    else:
        x = torch.view_as_complex(x.permute(0,2,3,1))
        x = ifftn(x, dim=(-2,-1))
        x = ifftshift(x, dim=(-2,-1))
        return complex_to_2channelfloat(x)