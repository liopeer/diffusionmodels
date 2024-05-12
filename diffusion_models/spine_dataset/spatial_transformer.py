"""Class and helper functions used for the random deformations."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf


def gauss_gen3D(n=10, s=5, sigma=3) -> torch.Tensor:
    """Generate blur kernel.

    Args:
        n (int, optional): mean of gaussian. Defaults to 10.
        s (int, optional): steps where value is defined. Defaults to 5.
        sigma (int, optional): sigma of the gaussian. Defaults to 3.

    Returns:
        torch.Tensor: blur_kernel
    """
    sigma = 3

    x = np.linspace(-(n - 1) / 2, (n - 1) / 2, s)
    y = np.linspace(-(n - 1) / 2, (n - 1) / 2, s)
    z = np.linspace(-(n - 1) / 2, (n - 1) / 2, s)

    xv, yv, zv = np.meshgrid(x, y, z)
    hg = np.exp(-(xv**2 + yv**2 + zv**2) / (2 * sigma**2))
    h = hg / np.sum(hg)

    blur_kernel = torch.from_numpy(h)[None, None, :]

    return blur_kernel


class SpatialTransformer(nn.Module):
    """N-D Spatial Transformer."""

    def __init__(self, size: tuple, mode="bilinear") -> None:
        """Initialize Spatial Transformer.

        Args:
            size (tuple): Size tuple, shape
            mode (str, optional): Which mode to use in sampling. Defaults to "bilinear".
        """
        super().__init__()
        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.float)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Forward pass of the spatial transformer. Generates the deformed image.

        Args:
            src (torch.Tensor): image to deform.
            flow (torch.Tensor): blur kernel.

        Returns:
            torch.Tensor: deformed image.
        """
        # new locations
        # print("This is the self grid shape: ", self.grid.shape)
        new_locs = self.grid.to(flow.device) + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (
                new_locs[:, i, ...] / (shape[i] - 1) - 0.5
            )

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(
            src, new_locs, align_corners=True, mode=self.mode
        )
