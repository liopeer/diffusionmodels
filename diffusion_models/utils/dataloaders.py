from typing import Callable, Optional
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from typing import Any

class MNISTTrainLoader(MNIST):
    def __init__(self, root: str, train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, download: bool = False) -> None:
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        download = True
        super().__init__(root, train, transform, target_transform, download)

class MNISTTestLoader(MNIST):
    def __init__(self, root: str, train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, download: bool = False) -> None:
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        download = False
        super().__init__(root, train, transform, target_transform, download)
