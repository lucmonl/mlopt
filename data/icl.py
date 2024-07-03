import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import os
import sys
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision import datasets, transforms

DATA_DIR = "/projects/dali/data/icl/"

def load_icl_data(batch_size: int, dim: int, length: int, train_size = -1):
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.isfile(DATA_DIR + "train.pth"):
        print("Generating ICL dataset...")
        feature_size = int(0.4 * dim)
        non_feature_size = int(0.2 * dim)
        feature_length, non_feature_length = int(0.6 * length),  length - int(0.6 * length)
        basis, _ = np.linalg.qr(np.random.rand(dim, feature_size + non_feature_size))
        feature_basis, non_feature_basis = basis[:, :feature_size], basis[:, feature_size:]

        feature_vector = (np.random.rand(train_size, feature_length, feature_size) - 0.5) @ feature_basis.T # train_size * legnth * dim
        w_truth = torch.from_numpy(np.random.rand(train_size, feature_size) @ feature_basis.T).unsqueeze(-1) #train_size * dim * 1

        non_feature_vector = (np.random.rand(train_size, non_feature_length, non_feature_size) - 0.5) @ non_feature_basis.T
        data_vector = torch.from_numpy(np.concatenate([non_feature_vector, feature_vector], axis=1)) # train_size * legnth * dim

        label_vector = torch.bmm(data_vector, w_truth) # train_size * length * 1
        data_matrix = torch.cat([data_vector, label_vector], dim=-1).float() # train_size * length * (dim+1)
        y_labels = data_matrix[:,-1,-1].clone().float() #train_size
        data_matrix[:,-1,-1] = 0

        train, test = TensorDataset(data_matrix, y_labels), TensorDataset(data_matrix[:batch_size], y_labels[:batch_size])
        torch.save(train, DATA_DIR + "train.pth")
        torch.save(test, DATA_DIR + "test.pth")
    else:
        print("Using cached ICL dataset.")
        train, test = torch.load(DATA_DIR + "train.pth"), torch.load(DATA_DIR + "test.pth")
    train_loader = torch.utils.data.DataLoader(
                        train,
                        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                        test,
                        batch_size=batch_size, shuffle=True)

    data_params = {"compute_acc": False}
    transform_to_one_hot = False
    C = 1

    #analysis_size = max(batch_size, 128)
    analysis_size = max(batch_size, 128)
    analysis = torch.utils.data.Subset(train, range(analysis_size))
    analysis_test = torch.utils.data.Subset(test, range(analysis_size))
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        batch_size=analysis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        batch_size=analysis_size, shuffle=False)
    return train_loader, test_loader, analysis_loader, analysis_test_loader, C, transform_to_one_hot, data_params