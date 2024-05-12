from spine_diffusion.datasets.random_deformation_dataset import (
    RandomDeformationDataset
)
from spine_diffusion.datasets.shapenet import ShapeNet_Dataset
from typing import Literal, Tuple
from torch.utils.data import DataLoader
from spine_diffusion.datasets.collations import collate_fn

def get_trainval_dataloaders(
        dataset: Literal["spine", "shapenet"],
        config: dict,
        batch_size: int
) -> Tuple[DataLoader]:
    if dataset == "spine":
        train_ds = RandomDeformationDataset(
            mode = "train",
            **config
        )
        val_ds = RandomDeformationDataset(
            mode = "val",
            **config
        )
    elif dataset == "shapenet":
        train_ds = ShapeNet_Dataset(
            mode = "train",
            **config
        )
        val_ds = ShapeNet_Dataset(
            mode = "val",
            **config
        )
    else:
        raise ValueError("no such dataset")
    train_dl = DataLoader(
        train_ds,
        batch_size = batch_size,
        shuffle = True,
        num_workers = batch_size,
        collate_fn = collate_fn,
        pin_memory=True
    )
    val_dl = DataLoader(
        val_ds,
        batch_size = batch_size,
        shuffle = False,
        num_workers = batch_size,
        collate_fn = collate_fn,
        pin_memory=True
    )
    return train_dl, val_dl