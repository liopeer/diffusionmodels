"""ShapeNet Dataset class, adopted from [1]_

.. [1] https://github.com/Zhengxinyang/LAS-Diffusion
"""
import skimage.measure
import torch
import os
from pathlib import Path
from typing import Literal, List, Optional, Union
from jaxtyping import Float32, UInt64
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import skimage
from spine_diffusion.utils.shapenet_utils import (
    snc_synth_id_to_category_all,
    snc_category_to_synth_id_13,
    snc_category_to_synth_id_5,
    snc_category_to_synth_id_all
)

class ShapeNet_Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            sdf_folder: str,
            mode: Literal["train","val","test"],
            data_class: str,
            resolution: int,
            rand_coords: bool,
            num_points: int,
            occ: bool = True,
            occ_threshold = 1/32,
            split_ratio: float = 0.9
        ):
        """ Constructor for Occupancy Field Dataset.

        Args:
            sdf_folder: where to find the files
            data_class: e.g. chair, airplane, ... or all
            resolution: resolution of the SDF/occ volumes
            rand_coords: whether to use random coordinates instead of voxe centers
            num_points: how many points to sample
            mode: which split of the data to load
            occ: whether to load occupancy or SDF
            occ_threshold: threshold for making occupancies from SDF
            split_ratio: split between train/val and test set
        """
        super().__init__()

        if data_class == "all":
            _data_classes = snc_category_to_synth_id_all.keys()
        else:
            # single category shapenet
            _data_classes = [data_class]

        self.resolution = resolution
        if not rand_coords:
            raise NotImplementedError("yet to be implemented")
        self.rand_coords = rand_coords
        self.num_points = num_points
        self.split = mode
        self.split_ratio = split_ratio
        self.occ = occ
        self.occ_threshold = occ_threshold

        self.sdf_paths = []
        for _data_class in _data_classes:
            _label = snc_category_to_synth_id_all[_data_class]
            _path = os.path.join(sdf_folder, _label)
            self.sdf_paths.extend(
                [p for p in Path(f'{_path}').glob('**/*.npy')])
            
        if self.split == "train":
            self.sdf_paths = self.sdf_paths[:int(len(self.sdf_paths)*0.9*split_ratio)]
        elif self.split == "val":
            self.sdf_paths = self.sdf_paths[int(len(self.sdf_paths)*0.9*split_ratio):int(len(self.sdf_paths)*split_ratio)]
        elif self.split == "test":
            self.sdf_paths = self.sdf_paths[int(len(self.sdf_paths)*split_ratio):]
        else:
            raise ValueError("unknown split value")
                
        self.multiplier = 128 // resolution

    def __len__(self):
        return len(self.sdf_paths)

    def __getitem__(self, index):
        sdf_path = self.sdf_paths[index]
        sdf = np.load(sdf_path)

        # downsample to requested resolution
        sdf = skimage.measure.block_reduce(sdf, (self.multiplier, self.multiplier, self.multiplier), np.mean)

        res = {}
        # sample coordinates
        coords = self._sample_normalized_coords(self.num_points)
        res["coords"] = coords

        if not self.occ:
            res["sdf"] = torch.from_numpy(sdf).unsqueeze(0).to(torch.float32)
            if torch.abs(res["sdf"].max()) > torch.abs(res["sdf"].min()):
                res["sdf"] = res["sdf"] / torch.abs(res["sdf"].max())
            else:
                res["sdf"] = res["sdf"] / torch.abs(res["sdf"].min())
            res["targets_sdf"] = self._get_coord_values(res["sdf"], coords).squeeze(0)
            res["loss_fn"] = "mse"
        else:
            occ = np.where(
                abs(sdf) < self.occ_threshold, 
                np.ones_like(sdf, dtype=np.float32), 
                np.zeros_like(sdf, dtype=np.float32)
            )
            res["occ"] = torch.from_numpy(occ).unsqueeze(0)
            res["targets_occ"] = self._get_coord_values(res["occ"], coords).to(torch.long).squeeze(0)
            res["occ"] = res["occ"].to(torch.long)
            res["loss_fn"] = "crossentropylogits"

        return res
    
    def _get_coord_values(
            self,
            volume: Float32[Tensor, "ch depth height width"], 
            coords: Float32[Tensor, "num_coords 3"]
        ) -> Union[
            Float32[Tensor, "ch num_coords"],
            UInt64[Tensor, "ch num_coords"]
        ]:
        volume_unsq = volume.unsqueeze(0) # add batch dim
        coords_unsq = coords[None, None, None, :, :]
        if self.occ:
            vals = F.grid_sample(volume_unsq, coords_unsq, mode="nearest", align_corners=True)
            vals = vals.to(torch.long)[0,:,0,0]
        else:
            vals = F.grid_sample(volume_unsq, coords_unsq, mode="bilinear", align_corners=True)
            vals = vals[0,:,0,0]
        assert vals.size() == (volume.shape[0], coords.shape[0]), f"{vals.size()}{volume.size()}{coords.size()}"
        return vals
    
    def _sample_normalized_coords(self, num_coords: int) -> Float32[Tensor, "num_coords 3"]:
        coords = torch.rand((num_coords, 3))
        return coords * 2 - 1