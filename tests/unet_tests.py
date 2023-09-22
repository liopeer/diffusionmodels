import context
import torch
from models.unet import UNet, EncodingBlock, DecodingBlock
from models.diffusion import DiffusionModel

model = UNet(4)

x1 = torch.rand((2,3,512,512))
x2 = torch.rand((2,3,256,256))
x3 = torch.rand((2,3,128,128))
x4 = torch.rand((2,3,64,64))
x5 = torch.rand((2,3,32,32))
x8 = torch.rand((2,3,50,50))
x6 = torch.rand((2,3,16,16))
x7 = torch.rand((2,3,8,8))

for elem in [x1, x2, x3, x4, x5, x8, x6, x7]:
    out = model(elem, None)
    print(out.shape)