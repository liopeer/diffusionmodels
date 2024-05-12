import torch
from typing import List

def collate_fn(batch: List[dict]):
    res = {key: [] for key in batch[0].keys()}
    res["loss_fn"] = batch[0]["loss_fn"]
    for sample in batch:
        for key, elem in sample.items():
            if isinstance(elem, torch.Tensor):
                res[key].append(elem)
    for key, elem in res.items():
        if isinstance(elem, list):
            res[key] = torch.stack(res[key], dim=0)
        elif isinstance(elem, str):
            assert key == "loss_fn"
        else:
            raise ValueError(f"{key}")
    return res