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

def take_first(dataset: TensorDataset, num_to_keep: int):
    return TensorDataset(dataset.tensors[0][0:num_to_keep], dataset.tensors[1][0:num_to_keep])

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
    
def get_transform():
    train_transform = []
    test_transform = []
    train_transform += [
        transforms.RandomCrop(size=32, padding=4)
    ]

    train_transform += [transforms.RandomHorizontalFlip()]

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.247, 0.2435, 0.2616]
    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    
    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform


def load_cifar(loss: str, batch_size: int, train_size = -1, use_small_analysis=False):
    data_params = {"compute_acc": True}
    input_ch = 3
    num_pixels = 32 * 32 * 3
    C = 10
    transform_to_one_hot = True
    """
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip()])
    cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True, transform=train_transform)
    cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    #y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
    #    make_labels(torch.tensor(cifar10_test.targets), loss)
    y_train, y_test = torch.LongTensor(cifar10_train.targets), \
                        torch.LongTensor(cifar10_test.targets)
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_train)
    if train_size != -1:
        train = take_first(train, batch_size)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_test)
    """

    train_transform, test_transform = get_transform()
    train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True, transform=train_transform)
    test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False, transform=test_transform)


    #analysis_size = max(batch_size, 128)
    analysis_size = 32 if use_small_analysis else max(batch_size, 128)
    analysis = torch.utils.data.Subset(train, range(analysis_size))
    analysis_test = torch.utils.data.Subset(test, range(analysis_size))
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        batch_size=analysis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        batch_size=analysis_size, shuffle=False)
    return train_loader, test_loader, analysis_loader, analysis_test_loader, input_ch, num_pixels, C, transform_to_one_hot, data_params

def load_cifar_federated(loss: str, batch_size: int, train_size = -1, client_num=1):
    data_params = {"compute_acc": True}
    input_ch = 3
    num_pixels = 32 * 32 * 3
    C = 10
    transform_to_one_hot = True

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip()])

    cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True, transform=train_transform)
    cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False)
    """
    X_train = train_transform(cifar10_train.data)
    """
    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    #y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
    #    make_labels(torch.tensor(cifar10_test.targets), loss)
    y_train, y_test = torch.LongTensor(cifar10_train.targets), \
                        torch.LongTensor(cifar10_test.targets)
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_train)
    if train_size != -1:
        train = take_first(train, batch_size)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_test)
    
    analysis_size = max(batch_size, 128)
    analysis = torch.utils.data.Subset(train, range(analysis_size))
    analysis_test = torch.utils.data.Subset(test, range(analysis_size))
    client_loaders = []
    randperm = np.random.permutation(len(train))
    for i in range(client_num):
        data_index = randperm[i:-1:client_num]
        client_train = torch.utils.data.Subset(train, data_index)
        client_loaders.append(torch.utils.data.DataLoader(
                        client_train,
                        batch_size=batch_size, shuffle=True))
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        batch_size=analysis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        batch_size=analysis_size, shuffle=False)
    return train_loader, client_loaders, test_loader, analysis_loader, analysis_test_loader, input_ch, num_pixels, C, transform_to_one_hot, data_params