import context
from utils.datasets import UnconditionedCifar10Dataset
from torch.utils.data import DataLoader

ds = UnconditionedCifar10Dataset("./data")
dl = DataLoader(ds, batch_size=10)

k = next(iter(dl))
print(type(k))