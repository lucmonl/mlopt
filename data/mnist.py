import numpy as np
from torchvision.datasets import CIFAR10
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import os
import sys
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision import datasets, transforms

def load_mnist(loss: str, batch_size: int):
    im_size             = 28
    padded_im_size      = 32
    input_ch            = 1
    C                   = 10
    transform_to_one_hot = True
    transform = transforms.Compose([transforms.Pad((padded_im_size - im_size)//2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.1307,0.3081)])

    train_dataset = datasets.MNIST('/projects/dali/data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True)
    
    test_dataset = datasets.MNIST('/projects/dali/data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False)

    analysis_dataset = torch.utils.data.Subset(train_dataset, range(batch_size))
    analysis_loader = torch.utils.data.DataLoader(
        analysis_dataset,
        batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, analysis_loader, input_ch, C, transform_to_one_hot