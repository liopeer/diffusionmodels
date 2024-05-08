from SDS_guidance_helpers import setup_sampler
import torch
import nibabel as nib
import torch
import numpy as np
from tqdm import tqdm
from diffusion_models.utils.datasets import LumbarSpineDataset

ds = LumbarSpineDataset()

# load guidance slice
sample = ds[64][0]
reference = torch.randn((1,2,3))
assert sample.size() == (1,128,128)

ckpt = "/home/peerli/net_scratch/stellar-fire-1/checkpoint61.pt"

sampler = setup_sampler(ckpt)

num_samples = 128

beta_t = sampler.model.fwd_diff.betas[-1].view(-1,1,1,1)
x = sampler.model.init_noise(num_samples) * beta_t

for i in tqdm(reversed(range(1, sampler.model.fwd_diff.timesteps))):
    t = i * torch.ones((num_samples), dtype=torch.long, device=list(sampler.model.parameters())[0].device)

    # prepare guidance slice
    sample_t, _ = sampler.model.fwd_diff(sample.unsqueeze(0), t[:1])
    sample_t = sample_t.squeeze(0)
    # print(sample_t.shape)
    assert sample_t.size() == (1, 128, 128)
    # sample_t is of shape [1, 128, 128]
    if reference.size() == (1,2,3):
        sample_t = sample_t
    elif reference.size() == (1,3,2):
        sample_t = sample_t.permute(0,2,1)
    elif reference.size() == (2,1,3):
        sample_t = sample_t.permute(1,0,2)
    elif reference.size() == (2,3,1):
        sample_t = sample_t.permute(1,2,0)
    elif reference.size() == (3,1,2):
        sample_t = sample_t.permute(2,0,1)
    elif reference.size() == (3,2,1):
        sample_t = sample_t.permute(2,1,0)

    x = x.squeeze()
    if sample_t.shape[0] == 1:
        # print(x.shape, sample_t.shape)
        x[64] = sample_t.squeeze()
    elif sample_t.shape[1] == 1:
        # print(x.shape, sample_t.shape)
        x[:,64] = sample_t.squeeze()
    elif sample_t.shape[2] == 1:
        # print(x.shape, sample_t.shape)
        x[:,:,64] = sample_t.squeeze()
    x = x.unsqueeze(1)

    x = sampler.model.denoise_singlestep(x, t)
    # turn the tensor around
    x = x.squeeze()
    if torch.rand(1).item() > 0.5:
        x = torch.rot90(x, dims=(0,1))
        reference = torch.rot90(reference, dims=(0,1))
    else:
        x = torch.rot90(x, dims=(1,2))
        reference = torch.rot90(reference, dims=(1,2))
    x = x.unsqueeze(1)

img = nib.Nifti1Image(x.squeeze().numpy(), affine=np.eye(4))
nib.save(img, "ThreeDOut_num2.nii.gz")