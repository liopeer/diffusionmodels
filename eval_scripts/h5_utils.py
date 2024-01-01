import h5py
import torch
from typing import Literal
from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import mse_loss
import numpy as np

samples_file = "/home/peerli/net_scratch/longRes_samples.h5"

def load_corrupted_kspace_and_mask(index: int, debug=False):
    """index in [0,1,2]"""
    f = h5py.File(samples_file, "r")
    samples = f[f"mask{index}"]["corrupted_kspace"]
    mask = f[f"mask{index}"]["mask"]
    kspace, mask = torch.from_numpy(np.array(samples)), torch.from_numpy(np.array(mask))
    f.close()
    if debug:
        return samples[:2], mask
    return kspace, mask

def write_results_lossguidance(
        index: int,
        results: Float[Tensor, "batch channel height width"],
        type: Literal["globalResampling","localResampling","slowingDown","direct"],
        guidance_factor: float,
        losses = Float[Tensor, "timesteps"],
        jump_length: int=None,
        num_resamplings: int=None,
        slowdown_factor: float=None,
        debug=False
    ):
    if type not in ["globalResampling","localResampling","slowingDown","direct"]:
        raise ValueError(type)
    if (type == "globalResampling") and (jump_length is None):
        raise ValueError("specify jump length")
    if (type == "localResampling") and ((jump_length is None) or (num_resamplings is None)):
        raise ValueError("specify both jump length and num resamplings")
    #print(slowdown_factor)
    if (type == "slowingDown") and (slowdown_factor is None):
        raise ValueError("specify slowdown factor")
    
    file = h5py.File(samples_file, "a")
    f = file[f"mask{index}"]
    if "reconstruction" not in f.keys():
        f.create_group("reconstruction")
        f["reconstruction"].create_group(type)
    else:
        if type not in f["reconstruction"].keys():
            f["reconstruction"].create_group(type)
    f = f["reconstruction"][type]
    keys = [int(elem) for elem in f.keys()]
    keys = sorted(keys)
    if len(keys) == 0:
        key = str(0)
    else:
        key = str(keys[-1] + 1)
    f.create_group(key)
    f = f[key]
    f["results"] = results.cpu()
    f["guidance_factor"] = guidance_factor
    f["losses"] = losses.cpu()
    if jump_length is not None:
        f["jump_length"] = jump_length
    if num_resamplings is not None:
        f["num_resamplings"] = num_resamplings
    if slowdown_factor is not None:
        f["slowdown_factor"] = slowdown_factor

    # calculate accuracy
    truth = torch.from_numpy(np.array(file["samples"]))
    if debug:
        truth = truth[:2]
    f["mse"] = mse_loss(results.cpu(), truth)

    file.close()