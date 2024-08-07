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

def take_first(dataset: TensorDataset, num_to_keep: int):
    #print(dataset.data.dtype)
    #return TensorDataset(dataset.data[0:num_to_keep], dataset.targets[0:num_to_keep])
    return torch.utils.data.Subset(dataset, np.arange(num_to_keep))

def load_mnist(loss: str, batch_size: int, train_size: int=-1):
    data_params = {"compute_acc": True}
    num_pixels          = 32 * 32
    im_size             = 28
    padded_im_size      = 32
    input_ch            = 1
    C                   = 10
    
    transform = transforms.Compose([transforms.Pad((padded_im_size - im_size)//2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.1307,0.3081)])
    if loss == "MSELoss":
        transform_to_one_hot = False
        target_transform = transforms.Compose([
                                    lambda x:torch.LongTensor([x]), # or just torch.tensor
                                    lambda x:F.one_hot(x,C),
                                    lambda x:torch.squeeze(x),
                                    lambda x:x.float()])
    else:
        transform_to_one_hot = True
        target_transform = None

    train_dataset = datasets.MNIST('/projects/dali/data', train=True, download=True, transform=transform, target_transform=target_transform)
    if train_size != -1:
        train_dataset = take_first(train_dataset, train_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True)
    
    test_dataset = datasets.MNIST('/projects/dali/data', train=False, download=True, transform=transform, target_transform=target_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False)

    analysis_size = max(batch_size, 128)
    analysis_dataset = torch.utils.data.Subset(train_dataset, range(analysis_size))
    analysis_loader = torch.utils.data.DataLoader(
        analysis_dataset,
        batch_size=analysis_size, shuffle=False)
    
    analysis_test_dataset = torch.utils.data.Subset(test_dataset, range(analysis_size))
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test_dataset,
        batch_size=analysis_size, shuffle=False)
    return train_loader, test_loader, analysis_loader, analysis_test_loader, input_ch, num_pixels, C, transform_to_one_hot, data_params