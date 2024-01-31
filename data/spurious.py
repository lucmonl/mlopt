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

def load_spurious_data(loss_name, train_size, batch_size):

    "The dataset is motivated by Condition 1 in https://arxiv.org/pdf/2307.11007.pdf"
    def get_label(X):
        """y = x[0] * x[1]"""
        return X[:,0] * X[:,1]

    torch.manual_seed(1)
    feat_size =50
    C = 1 #output dim
    transform_to_one_hot = False

    assert feat_size > 2
    assert loss_name == "MSELoss"
    X_train, X_test = torch.randn(train_size, feat_size), torch.randn(train_size, feat_size)
    X_train, X_test = 2*((X_train > 0).float()-0.5), 2*((X_test > 0).float()-0.5)
    y_train, y_test = get_label(X_train), get_label(X_test)

    train = TensorDataset(X_train, y_train)
    test = TensorDataset(X_test, y_test)
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
    return train_loader, test_loader, analysis_loader, analysis_test_loader, feat_size, C, transform_to_one_hot
