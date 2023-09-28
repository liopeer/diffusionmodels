from typing import Callable, Optional, Tuple
from torchvision.datasets import MNIST, CIFAR10, ImageNet
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from typing import Any
from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import torch

class Cifar10Dataset(CIFAR10):
    def __init__(self, root: str, train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, download: bool = False) -> None:
        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        download = True
        super().__init__(root, train, transform, target_transform, download)

class MNISTTrainDataset(MNIST):
    def __init__(self, root: str, train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, download: bool = False) -> None:
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,)), Resize((32,32), antialias=True)])
        download = True
        super().__init__(root, train, transform, target_transform, download)

class MNISTTestDataset(MNIST):
    def __init__(self, root: str, train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, download: bool = False) -> None:
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        download = True
        super().__init__(root, train, transform, target_transform, download)

class MNISTDebugDataset(MNISTTrainDataset):
    __len__ = lambda x: 100

class ImageNet64Dataset(Dataset):
    def __init__(self, root: str) -> None:
        super().__init__()
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,)), Resize((32,32), antialias=True)])
        self.batches = [os.path.join(root, elem) for elem in os.listdir(root) if "train_data_batch" in elem]
        self.lengths = []
        for elem in self.batches:
            d = self._unpickle(elem)
            x = d["data"]
            self.lengths.append(d.shape[0])
        self.len = np.sum(self.lengths)
        self.cum_sum_lengths = np.cumsum(self.lengths)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index) -> Any:
        mybatch = None
        for i, batch in enumerate(self.cum_sum_lengths):
            if index > batch:
                mybatch = self.batches[i-1]
        if mybatch is None:
            raise ValueError("no batch was selected for loading.")
        index = index - self.cum_sum_lengths[i-1]
        d = self._unpickle(mybatch)
        x = d["data"][index]
        y = d["labels"][index]
        return torch.tensor(x), torch.tensor(y)

    def _unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict