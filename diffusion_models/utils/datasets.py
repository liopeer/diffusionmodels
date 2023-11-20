from typing import Callable, Optional, Tuple
from torchvision.datasets import MNIST, CIFAR10, ImageNet
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop, InterpolationMode
from torchvision.transforms.functional import resize, center_crop
from typing import Any
from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import torch
import h5py
from torch.fft import ifft2, fft2, fftshift, ifftshift, fftn, ifftn
from utils.helpers import complex_to_2channelfloat

class Cifar10Dataset(CIFAR10):
    def __init__(self, root: str, train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, download: bool = False) -> None:
        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        download = True
        super().__init__(root, train, transform, target_transform, download)

class Cifar10DebugDataset(Cifar10Dataset):
    __len__ = lambda x: 5000

class MNISTTrainDataset(MNIST):
    def __init__(self, root: str, train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, download: bool = False) -> None:
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,)), Resize((32,32), antialias=True)])
        download = True
        super().__init__(root, train, transform, target_transform, download)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x = super().__getitem__(index)
        return (x[0],)
    
class MNISTKSpace(MNISTTrainDataset):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x = super().__getitem__(index)[0]
        x = fftshift(fftn(x, dim=(1,2), norm="ortho"), dim=(1,2))
        x = torch.view_as_real(x.squeeze(0))
        return (x.permute(2,0,1),)
    
class MNISTKSpaceDebug(MNISTKSpace):
    __len__ = lambda x: 500

class MNISTDebugDataset(MNISTTrainDataset):
    __len__ = lambda x: 100

class FastMRIBrainKSpace(Dataset):
    def __init__(self, root: str, size: int=128, kspace_interpolation: InterpolationMode=InterpolationMode.BILINEAR, antialias: bool=True) -> None:
        super().__init__()
        self.size = size
        self.kspace_interpolation = kspace_interpolation
        self.antialias = antialias
        h5_files = [os.path.join(root, elem) for elem in sorted(os.listdir(root))]
        self.samples = []
        for file_name in h5_files:
            file = h5py.File(file_name, 'r')
            data = file["kspace"]
            for slice in range(data.shape[0]):
                for coil in range(data.shape[1]):
                    self.samples.append({"file_name": file_name, "slice": slice, "coil": coil})

    def __len__(self):
        return len(self.samples)
    
    def get_kspace(self, index):
        file_name = self.samples[index]["file_name"]
        slice = self.samples[index]["slice"]
        coil = self.samples[index]["coil"]
        file = h5py.File(file_name, "r")
        data = torch.tensor(np.array(file["kspace"][slice, coil])).unsqueeze(0)
        file.close()
        # to image space for downsampling
        data = ifftshift(ifftn(data, dim=(1,2), norm="ortho"), dim=(1,2))
        data = torch.abs(data)
        data = center_crop(data, output_size=320)
        data = resize(data, size=(self.size, self.size), antialias=self.antialias, interpolation=self.kspace_interpolation)
        data = data - data.min()
        data = data * (1 / data.max())
        # back to kspace
        data = fftshift(fftn(data, dim=(1,2), norm="ortho"), dim=(1,2))
        data = complex_to_2channelfloat(data.squeeze(0))
        return data
    
    def get_rss(self, index):
        file_name = self.samples[index]["file_name"]
        slice = self.samples[index]["slice"]
        file = h5py.File(file_name, "r")
        x = torch.tensor(np.array(file["reconstruction_rss"][slice])).unsqueeze(0)
        file.close()
        x = resize(x, size=(self.size, self.size), antialias=True, interpolation=InterpolationMode.BILINEAR)
        x = x - x.min()
        x = x * (1 / x.max())
        return x
    
    def __getitem__(self, index) -> Any:
        data = self.get_kspace(index)

        #target = self.get_rss(index)

        return (data,)
    
class FastMRIBrainKSpaceDebug(FastMRIBrainKSpace):
    __len__ = lambda x: 500

class FastMRIBrainTrain(Dataset):
    def __init__(self, root: str, size: int=128) -> None:
        super().__init__()
        h5_files = [os.path.join(root, elem) for elem in sorted(os.listdir(root))]
        self.imgs = []
        for file_name in h5_files:
            file = h5py.File(file_name, 'r')
            slices = file["reconstruction_rss"].shape[0]
            for i in range(slices):
                self.imgs.append({"file_name":file_name, "index":i})
            file.close()
        self.transform = Compose([ToTensor(), Resize((size, size), antialias=True)])

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index) -> Any:
        file_name = self.imgs[index]["file_name"]
        index = self.imgs[index]["index"]
        file = h5py.File(file_name, 'r')
        x = file["reconstruction_rss"][index]
        x = self.transform(np.array(x))
        file.close()
        x = x - x.min()
        x = x * (1 / x.max())
        return (x, )
    
class FastMRIDebug(FastMRIBrainTrain):
    def __len__(self):
        return 128
    
class QuarterFastMRI(FastMRIBrainTrain):
    """only every 4th image of original dataset"""
    def __len__(self):
        return int(super().__len__() / 4)
    
    def __getitem__(self, index) -> Any:
        return super().__getitem__(int(index*4))

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