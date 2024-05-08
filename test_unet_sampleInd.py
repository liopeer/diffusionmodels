import torch
from diffusion_models.models.unet import UNet, DecodingBlock, EncodingBlock
import torch.nn.functional as F
import torch.nn as nn
import pytest

def test_decoderblock_sample_independence():
    model = DecodingBlock(
        4,
        2,
        256
    )
    input1 = torch.randn((2,4,32,32), requires_grad=True)
    input2 = torch.randn((2,2,64,64), requires_grad=True)
    time_emb = torch.randn((2,256), requires_grad=True)
    pred = model(input1, input2, time_emb)
    # pred = model(input)
    print(pred.shape)

    mask = torch.ones_like(pred)
    mask[0] = 0
    pred_masked = pred * mask
    target = torch.zeros_like(pred_masked)
    print(target.shape)

    loss = F.mse_loss(pred_masked, target)
    grad_input1 = torch.autograd.grad(loss, input1, retain_graph=True)[0]
    grad_input2 = torch.autograd.grad(loss, input2, retain_graph=True)[0]
    grad_timeemb = torch.autograd.grad(loss, time_emb)[0]
    print(grad_input1.shape, grad_input2.shape, grad_timeemb.shape)
    # assert grad_input1[0].mean() == pytest.approx(0)
    assert grad_input1[1].sum() != pytest.approx(0)
    assert grad_input2[0].mean() == pytest.approx(0)
    assert grad_input2[1].sum() != pytest.approx(0)
    assert grad_timeemb[0] == pytest.approx(0)
    assert grad_timeemb[1].sum() != pytest.approx(0)

def test_conv2d_sample_independence():
    model = nn.Conv2d(
        2,2,3
    )
    input = torch.randn((2,2,32,32), requires_grad=True)
    pred = model(input)
    # pred = model(input)
    print(pred.shape)

    mask = torch.ones_like(pred)
    mask[0] = 0
    pred_masked = pred * mask
    target = torch.zeros_like(pred_masked)
    print(target.shape)

    loss = F.mse_loss(pred_masked, target)
    grad_input = torch.autograd.grad(loss, input)[0]
    assert (grad_input[0] == 0).all()
    assert grad_input[1].sum() != 0