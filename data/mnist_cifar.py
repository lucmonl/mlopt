import random
import os, copy, pickle, time
import itertools
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision

DATASETS_FOLDER = "/projects/dali/data/"

def get_mnist(fpath=DATASETS_FOLDER, flatten=False, binarize=False, normalize=True, y0={0,1,2,3,4}):
    """get preprocessed mnist torch.TensorDataset class"""
    def _to_torch(d):
        X, Y = [], []
        for xb, yb in d:
            X.append(xb)
            Y.append(yb)
        return torch.Tensor(np.stack(X)), torch.LongTensor(np.stack(Y))

    to_tensor = torchvision.transforms.ToTensor()
    to_flat = torchvision.transforms.Lambda(lambda X: X.reshape(-1).squeeze())
    to_norm = torchvision.transforms.Normalize((0.5, ), (0.5, ))
    to_binary = torchvision.transforms.Lambda(lambda y: 0 if y in y0 else 1)

    transforms = [to_tensor]
    if normalize: transforms.append(to_norm)
    if flatten: transforms.append(to_flat)
    tf = torchvision.transforms.Compose(transforms)
    ttf = to_binary if binarize else None

    X_tr = torchvision.datasets.MNIST(fpath, download=True, transform=tf, target_transform=ttf)
    X_te = torchvision.datasets.MNIST(fpath, download=True, train=False, transform=tf, target_transform=ttf)

    return _to_torch(X_tr), _to_torch(X_te)

def get_mnist_dl(fpath=DATASETS_FOLDER, to_np=False, bs=128, pm=False, shuffle=False,
                 normalize=True, flatten=False, binarize=False, y0={0,1,2,3,4}):
    (X_tr, Y_tr), (X_te, Y_te) = get_mnist(fpath, normalize=normalize, flatten=flatten, binarize=binarize, y0=y0)
    tr_dl = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=bs, shuffle=shuffle, pin_memory=pm)
    te_dl = DataLoader(TensorDataset(X_te, Y_te), batch_size=bs, pin_memory=pm)
    return tr_dl, te_dl

def get_cifar(fpath=DATASETS_FOLDER, use_cifar10=False, flatten_data=False, transform_type='none',
              means=None, std=None, use_grayscale=False, binarize=False, normalize=True, y0={0,1,2,3,4}):
    """get preprocessed cifar torch.Dataset class"""

    if transform_type == 'none':
        normalize_cifar = lambda: torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        tensorize = torchvision.transforms.ToTensor()
        to_grayscale = torchvision.transforms.Grayscale()
        flatten = torchvision.transforms.Lambda(lambda X: X.reshape(-1).squeeze())

        transforms = [tensorize]
        if use_grayscale: transforms = [to_grayscale] + transforms
        if normalize: transforms.append(normalize_cifar())
        if flatten_data: transforms.append(flatten)
        tr_transforms = te_transforms = torchvision.transforms.Compose(transforms)

    if transform_type == 'basic':
        normalize_cifar = lambda: torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        tr_transforms= [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ]

        te_transforms = [
            torchvision.transforms.Resize(32),
            torchvision.transforms.CenterCrop(32),
            torchvision.transforms.ToTensor(),
        ]

        if normalize:
            tr_transforms.append(normalize_cifar())
            te_transforms.append(normalize_cifar())

        tr_transforms = torchvision.transforms.Compose(tr_transforms)
        te_transforms = torchvision.transforms.Compose(te_transforms)

    to_binary = torchvision.transforms.Lambda(lambda y: 0 if y in y0 else 1)
    target_transforms = to_binary if binarize else None
    dset = 'cifar10' if use_cifar10 else 'cifar100'
    func = torchvision.datasets.CIFAR10 if use_cifar10 else torchvision.datasets.CIFAR100

    X_tr = func(fpath, download=True, transform=tr_transforms, target_transform=target_transforms)
    X_te = func(fpath, download=True, train=False, transform=te_transforms, target_transform=target_transforms)

    return X_tr, X_te

def get_cifar_dl(fpath=DATASETS_FOLDER, use_cifar10=False, bs=128, shuffle=True, transform_type='none',
                 means=None, std=None, normalize=True, flatten_data=False, use_grayscale=False, nw=4, pm=False, binarize=False, y0={0,1,2,3,4}):
    """data in dataloaders have has shape (B, C, W, H)"""
    d_tr, d_te = get_cifar(fpath, use_cifar10=use_cifar10, use_grayscale=use_grayscale, transform_type=transform_type, normalize=normalize, means=means, std=std, flatten_data=flatten_data, binarize=binarize, y0=y0)
    tr_dl = DataLoader(d_tr, batch_size=bs, shuffle=shuffle, num_workers=nw, pin_memory=pm)
    te_dl = DataLoader(d_te, batch_size=bs, num_workers=nw, pin_memory=pm)
    return tr_dl, te_dl

def extract_tensors_from_loader(dl, repeat=1, transform_fn=None):
    X, Y = [], []
    for _ in range(repeat):
        for xb, yb in dl:
            if transform_fn:
                xb, yb = transform_fn(xb, yb)
            X.append(xb)
            Y.append(yb)
    X = torch.FloatTensor(torch.cat(X))
    Y = torch.LongTensor(torch.cat(Y))
    return X, Y

def extract_numpy_from_loader(dl, repeat=1, transform_fn=None):
    X, Y = extract_tensors_from_loader(dl, repeat=repeat, transform_fn=transform_fn)
    return X.numpy(), Y.numpy()

def _to_dl(X, Y, bs, shuffle=True):
    return DataLoader(TensorDataset(torch.Tensor(X), torch.LongTensor(Y)), batch_size=bs, shuffle=shuffle)

def get_binary_datasets(X, Y, y1, y2, image_width=28, use_cnn=False):
    assert type(X) is np.ndarray and type(Y) is np.ndarray
    idx0 = (Y==y1).nonzero()[0]
    idx1 = (Y==y2).nonzero()[0]
    idx = np.concatenate((idx0, idx1))
    X_, Y_ = X[idx,:], (Y[idx]==y2).astype(int)
    P = np.random.permutation(len(X_))
    X_, Y_ = X_[P,:], Y_[P]
    if use_cnn: X_ = X_.reshape(X.shape[0], -1, image_width)[:, None, :, :]
    return X_[P,:], Y_[P]

def get_binary_mnist(y1=0, y2=1, apply_padding=True, repeat_channels=True):
    
    def _make_cifar_compatible(X):
        if apply_padding: X = np.stack([np.pad(X[i][0], 2)[None,:] for i in range(len(X))]) # pad
        if repeat_channels: X = np.repeat(X, 3, axis=1) # add channels
        return X

    binarize = lambda X,Y: get_binary_datasets(X, Y, y1=y1, y2=y2)
    
    tr_dl, te_dl = get_mnist_dl(normalize=False)
    Xtr, Ytr = binarize(*extract_numpy_from_loader(tr_dl))
    Xte, Yte = binarize(*extract_numpy_from_loader(te_dl))
    Xtr, Xte = map(_make_cifar_compatible, [Xtr, Xte])
    return (Xtr, Ytr), (Xte, Yte)

def get_binary_cifar(y1=3, y2=5, c={0,1,2,3,4}, use_cifar10=True):
    binarize = lambda X,Y: get_binary_datasets(X, Y, y1=y1, y2=y2)
    binary = False if y1 is not None and y2 is not None else True
    if binary: print ("grouping cifar classes")
    tr_dl, te_dl = get_cifar_dl(use_cifar10=use_cifar10, shuffle=False, normalize=False, binarize=binary, y0=c)

    Xtr, Ytr = binarize(*extract_numpy_from_loader(tr_dl))
    Xte, Yte = binarize(*extract_numpy_from_loader(te_dl))
    return (Xtr, Ytr), (Xte, Yte)

def combine_datasets(Xm, Ym, Xc, Yc, randomize_order=False, randomize_first_block=False, randomize_second_block=False):
    """combine two datasets"""

    def partition(X, Y, randomize=False):
        """partition randomly or using labels"""
        if randomize: 
            n = len(Y)
            p = np.random.permutation(n)
            ni, pi = p[:n//2], p[n//2:]
        else:
            ni, pi = (Y==0).nonzero()[0], (Y==1).nonzero()[0]
        return X[pi], X[ni]

    def _combine(X1, X2):
        """concatenate images from two sources"""
        X = []
        for i in range(min(len(X1), len(X2))):
            x1, x2 = X1[i], X2[i]
            # randomize order 
            if randomize_order and random.random() < 0.5:
                x1, x2 = x2, x1
            x = np.concatenate((x1,x2), axis=1)
            X.append(x)
        return np.stack(X)

    Xmp, Xmn = partition(Xm, Ym, randomize=randomize_first_block)
    Xcp, Xcn = partition(Xc, Yc, randomize=randomize_second_block)
    n = min(map(len, [Xmp, Xmn, Xcp, Xcn]))
    Xmp, Xmn, Xcp, Xcn = map(lambda Z: Z[:n], [Xmp, Xmn, Xcp, Xcn])

    Xp = _combine(Xmp, Xcp)
    Yp = np.ones(len(Xp))

    Xn = _combine(Xmn, Xcn)
    Yn = np.zeros(len(Xn))
    
    X = np.concatenate([Xp, Xn], axis=0)
    Y = np.concatenate([Yp, Yn], axis=0)
    P = np.random.permutation(len(X))
    X, Y = X[P], Y[P]
    return X, Y

def get_mnist_cifar(mnist_classes=(0,1), cifar_classes=None, c={0,1,2,3,4}, 
                    randomize_mnist=False, randomize_cifar=False):  
        
    y1, y2 = mnist_classes
    (Xtrm, Ytrm), (Xtem, Ytem) = get_binary_mnist(y1=y1, y2=y2)
    
    y1, y2 = (None, None) if cifar_classes is None else cifar_classes
    (Xtrc, Ytrc), (Xtec, Ytec) = get_binary_cifar(c=c, y1=y1, y2=y2)
    
    Xtr, Ytr = combine_datasets(Xtrm, Ytrm, Xtrc, Ytrc, randomize_first_block=randomize_mnist, randomize_second_block=randomize_cifar)
    Xte, Yte = combine_datasets(Xtem, Ytem, Xtec, Ytec, randomize_first_block=randomize_mnist, randomize_second_block=randomize_cifar)
    return (Xtr, Ytr), (Xte, Yte)

def get_mnist_cifar_dl(mnist_classes=(0,1), cifar_classes=None, c={0,1,2,3,4}, bs=256, 
                       randomize_mnist=False, randomize_cifar=False):
    (Xtr, Ytr), (Xte, Yte) = get_mnist_cifar(mnist_classes=mnist_classes, cifar_classes=cifar_classes, 
                                             c=c, randomize_mnist=randomize_mnist, randomize_cifar=randomize_cifar)
    tr_dl = _to_dl(Xtr, Ytr, bs=bs, shuffle=True)
    te_dl = _to_dl(Xte, Yte, bs=100, shuffle=False)
    return tr_dl, te_dl

def load_mnist_cifar(mnist_classes=(0,1), cifar_classes=None, batch_size=256, 
                       randomize_mnist=False, randomize_cifar=False):
    data_params = {"compute_acc": True}
    input_ch = 6
    num_pixels = -1
    C = len(mnist_classes)
    transform_to_one_hot = True

    (Xtr, Ytr), (Xte, Yte) = get_mnist_cifar(mnist_classes=mnist_classes, cifar_classes=cifar_classes, 
                                             c={0,1,2,3,4}, randomize_mnist=randomize_mnist, randomize_cifar=randomize_cifar)
    train_loader = _to_dl(Xtr, Ytr, bs=batch_size, shuffle=True)
    test_loader = _to_dl(Xte, Yte, bs=batch_size, shuffle=False)

    analysis_loader = _to_dl(Xtr[:batch_size], Ytr[:batch_size], bs=batch_size, shuffle=False)
    analysis_test_loader = _to_dl(Xte[:batch_size], Yte[:batch_size], bs=batch_size, shuffle=False)

    return train_loader, test_loader, analysis_loader, analysis_test_loader, input_ch, num_pixels, C, transform_to_one_hot, data_params