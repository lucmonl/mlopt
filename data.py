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

DATASETS_FOLDER = "/projects/dali/data/" #os.environ["DATASETS"]

def center(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(0)
    return X_train - mean, X_test - mean

def standardize(X_train: np.ndarray, X_test: np.ndarray):
    std = X_train.std(0)
    return (X_train / std, X_test / std)

def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: np.ndarray, shape: Tuple):
    return arr.reshape(arr.shape[0], *shape)

def _one_hot(tensor: Tensor, num_classes: int, default=0):
    M = F.one_hot(tensor, num_classes)
    M[M == 0] = default
    return M.float()

def make_labels(y, loss):
    if loss == "CrossEntropyLoss":
        return y
    elif loss == "MSELoss":
        return _one_hot(y, 10, 0)
    elif loss == "exp":
        return _one_hot(y, 10, 0)
    else:
        raise NotImplementedError


def load_cifar(loss: str, batch_size: int):
    input_ch = 3
    num_pixels = 32 * 32 * 3
    cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True)
    cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    #y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
    #    make_labels(torch.tensor(cifar10_test.targets), loss)
    y_train, y_test = torch.LongTensor(cifar10_train.targets), \
                        torch.LongTensor(cifar10_test.targets)
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_test)
    analysis = torch.utils.data.Subset(train, range(batch_size))
    analysis_test = torch.utils.data.Subset(test, range(batch_size))
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        batch_size=batch_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, analysis_loader, analysis_test_loader, input_ch, num_pixels

def load_mnist(loss: str, batch_size: int):
    im_size             = 28
    padded_im_size      = 32
    input_ch = 1
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
    return train_loader, test_loader, analysis_loader, input_ch