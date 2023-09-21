import torch.nn as nn
import torch

class MNISTEncoder(nn.Module):
    def __init__(self, out_classes=10, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_classes = out_classes
        channels = [2**i for i in range(5)]
        self.encoder = []
        for i in range(4):
            self.encoder.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=kernel_size, padding="same"))
            self.encoder.append(nn.BatchNorm2d(channels[i+1]))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.MaxPool2d(2))
        self.conv = nn.Sequential(*self.encoder)
        self.fc = nn.Linear(16, self.out_classes)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.squeeze())