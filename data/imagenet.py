import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from typing import Tuple
import torch.utils.data as data
from torch.utils.data.dataset import TensorDataset
import os
import sys
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision import datasets, transforms

def load_imagenet_tiny(batch_size: int, tiny_analysis: bool=False):
    data_params = {"compute_acc": True}
    input_ch = 3
    num_pixels = 224 * 224 * 3
    C = 200
    transform_to_one_hot = True

    data_dir = "/projects/dali/data/tiny-224/"
    num_workers = {"train": 4, "val": 0, "test": 0}
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]
        ),
    }
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val", "test"]
    }
    train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True)
    analysis_size = 32 if tiny_analysis else max(batch_size, 128)
    analysis = torch.utils.data.Subset(image_datasets['train'], range(analysis_size))
    analysis_test = torch.utils.data.Subset(image_datasets['val'], range(analysis_size))
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        batch_size=analysis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        batch_size=analysis_size, shuffle=False)
    return train_loader, test_loader, analysis_loader, analysis_test_loader, input_ch, num_pixels, C, transform_to_one_hot, data_params