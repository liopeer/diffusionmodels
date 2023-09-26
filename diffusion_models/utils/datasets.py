from typing import Callable, Optional, Tuple
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from typing import Any

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
