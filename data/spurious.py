import numpy as np
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import os
import sys
import torch
from torch import Tensor
import torch.nn.functional as F

def load_spurious_data(loss_name, feat_dim, train_size, batch_size):

    "The dataset is motivated by Condition 1 in https://arxiv.org/pdf/2307.11007.pdf"
    def get_label(X):
        """y = x[0] * x[1]"""
        return X[:,0] * X[:,1]

    torch.manual_seed(1)
    C = 1 #output dim
    transform_to_one_hot = False

    assert feat_dim > 2
    assert loss_name == "MSELoss"
    X_train, X_test = torch.randn(train_size, feat_dim), torch.randn(train_size, feat_dim)
    X_train, X_test = 2*((X_train > 0).float()-0.5), 2*((X_test > 0).float()-0.5)
    y_train, y_test = get_label(X_train), get_label(X_test)

    train = TensorDataset(X_train, y_train)
    test = TensorDataset(X_test, y_test)
    anaylsis_size = min(train_size, max(batch_size, 128))
    analysis = torch.utils.data.Subset(train, range(anaylsis_size))
    analysis_test = torch.utils.data.Subset(test, range(anaylsis_size))

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        batch_size=anaylsis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        batch_size=anaylsis_size, shuffle=False)
    return train_loader, test_loader, analysis_loader, analysis_test_loader, feat_dim, C, transform_to_one_hot


def load_signal_noise_data_2d(loss_name, patch_dim, feat_dim, train_size, batch_size):
    """The dataset is motivated by https://arxiv.org/pdf/2310.07269.pdf Definition 2.1"""
    #assert loss_name == "CrossEntropyLoss"

    torch.manual_seed(1)
    C = 1 #output dim
    transform_to_one_hot = False

    #signal = torch.zeros(feat_dim)
    signal = torch.randn(feat_dim)
    flip_prob = 0.1
    assert int(flip_prob * train_size) > 1

    X_train, X_test = torch.randn(train_size, patch_dim, feat_dim) / (2*patch_dim), torch.randn(train_size, patch_dim, feat_dim) / (2*patch_dim)
    X_train = X_train - torch.einsum('ij,k->ijk', (X_train @ signal), signal /  (np.linalg.norm(signal)**2)) 
    X_test = X_test - torch.einsum('ij,k->ijk', (X_test @ signal), signal /  (np.linalg.norm(signal)**2)) 

    y_train_true, y_test_true = torch.bernoulli(0.5*torch.ones(train_size)), torch.bernoulli(0.5*torch.ones(train_size)) #equal prob for 0 and 1
    
    signal_index_train   = torch.randint(low=0, high=patch_dim, size=(train_size,))
    signal_index_test    = torch.randint(low=0, high=patch_dim, size=(train_size,))

    # insert signal
    X_train[range(train_size), signal_index_train, :] = torch.outer((y_train_true - 0.5)*2, signal) # adjust to +- 1
    X_test[range(train_size), signal_index_test, :]   = torch.outer((y_test_true - 0.5)*2, signal)

    # flip labels
    flip_index_train = torch.randperm(train_size)[:int(flip_prob * train_size)]
    flip_index_test = torch.randperm(train_size)[:int(flip_prob * train_size)]
    y_train_true[flip_index_train]  = (1 - y_train_true[flip_index_train])
    y_test_true[flip_index_test]    = (1 - y_test_true[flip_index_test])
    #y_train_true, y_test_true = y_train_true.to(torch.int64), y_test_true.to(torch.int64)
    y_train_true, y_test_true = 2*(y_train_true-0.5), 2*(y_test_true-0.5)
    train = TensorDataset(X_train, y_train_true)
    test = TensorDataset(X_test, y_test_true)
    anaylsis_size = min(train_size, max(batch_size, 128))
    analysis = torch.utils.data.Subset(train, range(anaylsis_size))
    analysis_test = torch.utils.data.Subset(test, range(anaylsis_size))

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        batch_size=anaylsis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        batch_size=anaylsis_size, shuffle=False)
    data_params = {"signal": [signal], "compute_acc": True, "signal_patch_index": signal_index_train}
    return train_loader, test_loader, analysis_loader, analysis_test_loader, feat_dim, C, transform_to_one_hot, data_params


def load_multi_view_data(loss_name, patch_dim, feat_dim, train_size, batch_size):
    """The dataset is motivated by https://arxiv.org/pdf/2207.05931.pdf """
    #assert loss_name == "CrossEntropyLoss"

    torch.manual_seed(1)
    C = 1 #output dim
    transform_to_one_hot = False

    signal = torch.randn(feat_dim)

    X_train, X_test = torch.randn(train_size, patch_dim, feat_dim), torch.randn(train_size, patch_dim, feat_dim)
    X_train = X_train - torch.einsum('ij,k->ijk', (X_train @ signal), signal /  (np.linalg.norm(signal)**2)) 
    X_test = X_test - torch.einsum('ij,k->ijk', (X_test @ signal), signal /  (np.linalg.norm(signal)**2)) 
    y_train_true, y_test_true = torch.bernoulli(0.5*torch.ones(train_size)), torch.bernoulli(0.5*torch.ones(train_size)) #equal prob for 0 and 1
    y_train_true, y_test_true = 2*(y_train_true-0.5), 2*(y_test_true-0.5) # adjust to +- 1
    
    signal_index_train   = torch.randint(low=0, high=patch_dim, size=(train_size,))
    signal_index_test    = torch.randint(low=0, high=patch_dim, size=(train_size,))

    # insert weak signal
    weak_multiplier = feat_dim ** (-0.251)
    X_train[range(train_size), signal_index_train, :] = torch.outer(weak_multiplier * y_train_true, signal)
    X_test[range(train_size), signal_index_test, :]   = torch.outer(weak_multiplier * y_test_true, signal)
    

    # insert strong signal
    strong_ratio = int(0.7 * train_size)
    strong_multiplier = 2* np.sqrt(feat_dim) * weak_multiplier
    X_train[range(train_size)[:strong_ratio], signal_index_train[:strong_ratio], :] = torch.outer(strong_multiplier * y_train_true[:strong_ratio], signal) 
    X_test[range(train_size)[:strong_ratio], signal_index_test[:strong_ratio], :]   = torch.outer(strong_multiplier * y_test_true[:strong_ratio], signal)

    #y_train_true, y_test_true = y_train_true.to(torch.int64), y_test_true.to(torch.int64)
    
    train = TensorDataset(X_train, y_train_true)
    test = TensorDataset(X_test, y_test_true)
    anaylsis_size = min(train_size, max(batch_size, 128))
    analysis = torch.utils.data.Subset(train, range(anaylsis_size))
    analysis_test = torch.utils.data.Subset(test, range(anaylsis_size))

    shuffle = False
    assert not shuffle ## will use the exact order of signal index in analysis/align
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        batch_size=anaylsis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        batch_size=anaylsis_size, shuffle=False)
    
    data_params = {"signal": [signal], "compute_acc": True, "signal_patch_index": signal_index_train}
    return train_loader, test_loader, analysis_loader, analysis_test_loader, feat_dim, C, transform_to_one_hot, data_params

def load_orthogonal_data(loss_name, patch_dim, feat_dim, train_size, batch_size):
    #assert loss_name == "CrossEntropyLoss"
    data_params = {}

    torch.manual_seed(1)
    C = 1 #output dim
    
    w_true = torch.randn(feat_dim)
    transform_to_one_hot = False

    X_train, X_test = torch.randn(feat_dim, patch_dim), torch.randn(feat_dim, patch_dim)
    X_train, _ = torch.linalg.qr(X_train)
    X_train = X_train.repeat(train_size, 1, 1)
    X_train = torch.permute(X_train, [0,2,1])
    X_test, _ = torch.linalg.qr(X_test)
    X_test = X_test.repeat(train_size, 1, 1)
    X_test = torch.permute(X_test, [0,2,1])

    X_train = torch.einsum('ijk,ij->ijk', X_train, torch.rand(train_size, patch_dim))
    X_test = torch.einsum('ijk,ij->ijk', X_test, torch.rand(train_size, patch_dim))

    flip_prob = 0.2
    flip_index_train = torch.randperm(train_size)[:int(flip_prob * train_size)]
    flip_index_test = torch.randperm(train_size)[:int(flip_prob * train_size)]

    #y_train_true, y_test_true = torch.bernoulli(0.5*torch.ones(train_size)), torch.bernoulli(0.5*torch.ones(train_size)) #equal prob for 0 and 1
    #y_train_true, y_test_true = 2*(y_train_true-0.5), 2*(y_test_true-0.5) # adjust to +- 1
    y_train_true, y_test_true = ((X_train[:,0,:] @ w_true) > 0).int(), ((X_test[:,0,:] @ w_true) > 0).int()
    y_train_true[flip_index_train]  = (1 - y_train_true[flip_index_train])
    y_test_true[flip_index_test]    = (1 - y_test_true[flip_index_test])

    y_train_true, y_test_true = 2*(y_train_true-0.5), 2*(y_test_true-0.5) # adjust to +- 1
    
    train = TensorDataset(X_train, y_train_true)
    test = TensorDataset(X_test, y_test_true)
    anaylsis_size = min(train_size, max(batch_size, 128))
    analysis = torch.utils.data.Subset(train, range(anaylsis_size))
    analysis_test = torch.utils.data.Subset(test, range(anaylsis_size))

    shuffle = False
    assert not shuffle ## will use the exact order of signal index in analysis/align
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        batch_size=anaylsis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        batch_size=anaylsis_size, shuffle=False)
    
    data_params = {"compute_acc": True}
    return train_loader, test_loader, analysis_loader, analysis_test_loader, feat_dim, C, transform_to_one_hot, data_params

def load_scalaraized_data(loss_name, patch_dim, feat_dim, train_size, batch_size):
    #assert loss_name == "CrossEntropyLoss"
    data_params = {}

    torch.manual_seed(1)
    C = 1 #output dim
    
    w_true = torch.randn(feat_dim)
    transform_to_one_hot = False

    X_train, X_test = torch.randn(feat_dim, patch_dim), torch.randn(feat_dim, patch_dim)
    X_train, _ = torch.linalg.qr(X_train)
    X_train = X_train.repeat(train_size, 1, 1)
    X_train = torch.permute(X_train, [0,2,1])
    X_test, _ = torch.linalg.qr(X_test)
    X_test = X_test.repeat(train_size, 1, 1)
    X_test = torch.permute(X_test, [0,2,1])

    X_train_coef = torch.rand(train_size, patch_dim)
    X_train = torch.einsum('ijk,ij->ijk', X_train, X_train_coef)
    X_test_coef = torch.rand(train_size, patch_dim)
    X_test = torch.einsum('ijk,ij->ijk', X_test, X_test_coef)

    flip_prob = 0.2
    flip_index_train = torch.randperm(train_size)[:int(flip_prob * train_size)]
    flip_index_test = torch.randperm(train_size)[:int(flip_prob * train_size)]

    #y_train_true, y_test_true = torch.bernoulli(0.5*torch.ones(train_size)), torch.bernoulli(0.5*torch.ones(train_size)) #equal prob for 0 and 1
    #y_train_true, y_test_true = 2*(y_train_true-0.5), 2*(y_test_true-0.5) # adjust to +- 1
    y_train_true, y_test_true = ((X_train[:,0,:] @ w_true) > 0).int(), ((X_test[:,0,:] @ w_true) > 0).int()
    y_train_true[flip_index_train]  = (1 - y_train_true[flip_index_train])
    y_test_true[flip_index_test]    = (1 - y_test_true[flip_index_test])

    y_train_true, y_test_true = 2*(y_train_true-0.5), 2*(y_test_true-0.5) # adjust to +- 1
    
    train = TensorDataset(X_train_coef, y_train_true)
    test = TensorDataset(X_test_coef, y_test_true)
    anaylsis_size = min(train_size, max(batch_size, 128))
    analysis = torch.utils.data.Subset(train, range(anaylsis_size))
    analysis_test = torch.utils.data.Subset(test, range(anaylsis_size))

    shuffle = False
    assert not shuffle ## will use the exact order of signal index in analysis/align
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        batch_size=anaylsis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        batch_size=anaylsis_size, shuffle=False)
    
    data_params = {"compute_acc": True}
    return train_loader, test_loader, analysis_loader, analysis_test_loader, feat_dim, C, transform_to_one_hot, data_params


def load_multi_view_orthogonal_data(loss_name, patch_dim, feat_dim, train_size, batch_size):
    """The dataset is motivated by https://arxiv.org/pdf/2207.05931.pdf """
    #assert loss_name == "CrossEntropyLoss"

    torch.manual_seed(1)
    C = 1 #output dim
    transform_to_one_hot = False

    signal = torch.randn(feat_dim)

    X_train, X_test = torch.randn(train_size, feat_dim, patch_dim), torch.randn(train_size, feat_dim, patch_dim)
    X_train, _ = torch.linalg.qr(X_train)
    X_train = torch.permute(X_train, [0,2,1])
    X_test, _ = torch.linalg.qr(X_test)
    X_test = torch.permute(X_test, [0,2,1])

    X_train = torch.einsum('ijk,ij->ijk', X_train, torch.rand(train_size, patch_dim))
    X_test = torch.einsum('ijk,ij->ijk', X_test, torch.rand(train_size, patch_dim))

    y_train_true, y_test_true = torch.bernoulli(0.5*torch.ones(train_size)), torch.bernoulli(0.5*torch.ones(train_size)) #equal prob for 0 and 1
    y_train_true, y_test_true = 2*(y_train_true-0.5), 2*(y_test_true-0.5) # adjust to +- 1
    
    signal_index_train   = torch.randint(low=0, high=patch_dim, size=(train_size,))
    signal_index_test    = torch.randint(low=0, high=patch_dim, size=(train_size,))

    # insert weak signal
    weak_multiplier = feat_dim ** (-0.251)
    X_train[range(train_size), signal_index_train, :] = torch.outer(weak_multiplier * y_train_true, signal)
    X_test[range(train_size), signal_index_test, :]   = torch.outer(weak_multiplier * y_test_true, signal)

    # insert strong signal
    strong_ratio = int(0.7 * train_size)
    strong_multiplier = 2* np.sqrt(feat_dim) * weak_multiplier
    X_train[range(train_size)[:strong_ratio], signal_index_train[:strong_ratio], :] = torch.outer(strong_multiplier * y_train_true[:strong_ratio], signal) 
    X_test[range(train_size)[:strong_ratio], signal_index_test[:strong_ratio], :]   = torch.outer(strong_multiplier * y_test_true[:strong_ratio], signal)

    #y_train_true, y_test_true = y_train_true.to(torch.int64), y_test_true.to(torch.int64)
    
    train = TensorDataset(X_train, y_train_true)
    test = TensorDataset(X_test, y_test_true)
    anaylsis_size = min(train_size, max(batch_size, 128))
    analysis = torch.utils.data.Subset(train, range(anaylsis_size))
    analysis_test = torch.utils.data.Subset(test, range(anaylsis_size))

    shuffle = False
    assert not shuffle ## will use the exact order of signal index in analysis/align
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        batch_size=anaylsis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        batch_size=anaylsis_size, shuffle=False)
    
    data_params = {"signal": [signal], "compute_acc": True, "signal_patch_index": signal_index_train}
    return train_loader, test_loader, analysis_loader, analysis_test_loader, feat_dim, C, transform_to_one_hot, data_params


def load_secondary_feature_data(loss_name, patch_dim, feat_dim, train_size, batch_size):
    """The dataset is motivated by https://arxiv.org/pdf/2207.05931.pdf """
    #assert loss_name == "CrossEntropyLoss"

    torch.manual_seed(1)
    C = 1 #output dim
    transform_to_one_hot = False

    signal_1 = torch.randn(feat_dim)
    signal_2 = torch.randn(feat_dim)
    signal_2 = signal_2 - (signal_1 @ signal_2) * signal_1 / (np.linalg.norm(signal_1)**2) # orthogonalize
    assert(signal_1 @ signal_2 < 1e-5)

    X_train, X_test = torch.randn(train_size, patch_dim, feat_dim), torch.randn(train_size, patch_dim, feat_dim)
    X_train = X_train - torch.einsum('ij,k->ijk', (X_train @ signal_1), signal_1 /  (np.linalg.norm(signal_1)**2)) 
    X_train = X_train - torch.einsum('ij,k->ijk', (X_train @ signal_2), signal_2 /  (np.linalg.norm(signal_2)**2)) 
    assert((torch.mean(X_train @ signal_1) < 1e-5) and (torch.mean(X_train @ signal_2) < 1e-5))

    X_test = X_test - torch.einsum('ij,k->ijk', (X_test @ signal_1), signal_1 /  (np.linalg.norm(signal_1)**2)) 
    X_test = X_test - torch.einsum('ij,k->ijk', (X_test @ signal_2), signal_2 /  (np.linalg.norm(signal_2)**2)) 
    assert((torch.mean(X_test @ signal_1) < 1e-5) and (torch.mean(X_test @ signal_2) < 1e-5))

    y_train_true, y_test_true = torch.bernoulli(0.5*torch.ones(train_size)), torch.bernoulli(0.5*torch.ones(train_size)) #equal prob for 0 and 1
    y_train_true, y_test_true = 2*(y_train_true-0.5), 2*(y_test_true-0.5) # adjust to +- 1
    
    signal_index_train   = torch.randint(low=0, high=patch_dim, size=(train_size,))
    signal_index_test    = torch.randint(low=0, high=patch_dim, size=(train_size,))

    # insert weak signal
    #weak_multiplier = feat_dim ** (-0.251)
    weak_multiplier = np.sqrt(feat_dim)
    X_train[range(train_size), signal_index_train, :] = torch.outer(weak_multiplier * y_train_true, signal_2) 
    X_test[range(train_size), signal_index_test, :]   = torch.outer(weak_multiplier * y_test_true, signal_2)

    # insert strong signal
    strong_ratio = int(0.7 * train_size)
    strong_multiplier = 2 * weak_multiplier
    X_train[range(train_size)[:strong_ratio], signal_index_train[:strong_ratio], :] = torch.outer(strong_multiplier * y_train_true[:strong_ratio], signal_1) 
    X_test[range(train_size)[:strong_ratio], signal_index_test[:strong_ratio], :]   = torch.outer(strong_multiplier * y_test_true[:strong_ratio], signal_1)

    #y_train_true, y_test_true = y_train_true.to(torch.int64), y_test_true.to(torch.int64)
    
    train = TensorDataset(X_train, y_train_true)
    test = TensorDataset(X_test, y_test_true)
    anaylsis_size = min(train_size, max(batch_size, 128))
    analysis = torch.utils.data.Subset(train, range(anaylsis_size))
    analysis_test = torch.utils.data.Subset(test, range(anaylsis_size))

    shuffle = False
    assert not shuffle ## will use the exact order of signal index in analysis/align
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        batch_size=anaylsis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        batch_size=anaylsis_size, shuffle=False)
    
    data_params = {"signal": [signal_1, signal_2], "compute_acc": True, "signal_patch_index": signal_index_train}
    return train_loader, test_loader, analysis_loader, analysis_test_loader, feat_dim, C, transform_to_one_hot, data_params