import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Literal, Dict, Any, Optional
from jaxtyping import Float32, UInt, UInt64

class BaseDataset(Dataset, ABC):
    """Interface for Datasets in Spine Diffusion package.
    
    This interface is currently not enforced, but any dataset implementation
    should follow the guidelines outlined here, this is especially true for the
    exact returns of the __getitem__ method.
    """
    def __init__(
        self,
        resolution: int,
        random_crop: bool,
        crop_size: int,
        mode: Literal["train","val","test"],
        **kwargs
    ):
        """Constructor of BaseDataset.

        Args:
            resolution: determines base resolution of the dataset, i.e. a
                dataset with an original size of 256 (in 3D) will be downsampled
                to that resolution
            random_crop: 
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """__getitem__ method of BaseDataset.
        
        Args:
            idx: index of desired sample

        Returns:
            dictionary with keys and items as below (not all keys necessary)

        .. code-block:: python
            dict(
                # ch corresponds to num_classes where applicable
                sdf: Optional[
                    Float32[Tensor, "1 res res res"],
                    None
                ] = None,

                occ: Optional[
                    UInt64[Tensor, "1 res res res"], # with unique values in range(2, num_classes+1)
                ] = None,

                coords: Optional[Float32[Tensor, "num_points 3"], None] = None,
                targets_occ: Optional[
                    UInt64[Tensor, "num_points"], # 2 or multi class with probabilities 1
                    Float32[Tensor, "num_points num_classes"], # 2 or multi class with probabilities in [0,1]
                    None
                ] = None,
                targets_sdf: Optional[
                    Float32[Tensor, "num_points"],
                    None
                ]
                loss_fn: Literal["crossentropylogits","mse"] = "crossentropylogits"
                metadata: Optional[Any, None] = None
            )
        
        - "sdf" is full volume and should be normalized to [-1,1] range
        - "sdf_target" is cropped volume, equally normalized, may be a TSDF
          of the original data to enhance learning
        - "occ_float" full volume occupancy as torch.float32, normalized to [0,1] range
        - "occ_target" cropped binary/multi-class torch.long tensor
        - "vox_coords" contains coords of voxel centers of "sdf_target" or "occ_target",
          normalized to [-1,1] range (see torch.grid_sample(align_corners=True) for reference). 
          If random_crop is False, this is not needed and will default to
          all voxel centers in the volume.
        - "rand_coords" can be used for randomized sampling of coordinates instead
          of voxel centers
        - "rand_targets" can be used for interpolated SDF values
        - "metadata" anything
        - while the channels "ch" will usually be 1, it might be good for multi-class
          problems to split classes between channels, should be float for 
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
    @staticmethod
    def check_output(output: Dict[str, Any]):
        pass