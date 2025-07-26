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

DATASETS_FOLDER = os.environ["DATA_HOME"]

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

    train_dataset = datasets.MNIST(DATASETS_FOLDER, train=True, download=True, transform=transform, target_transform=target_transform)
    if train_size != -1:
        train_dataset = take_first(train_dataset, train_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True)
    
    test_dataset = datasets.MNIST(DATASETS_FOLDER, train=False, download=True, transform=transform, target_transform=target_transform)
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



def load_mnist_federated(loss: str, batch_size: int, train_size = -1, client_num=1, alpha=0.0):
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

    train = datasets.MNIST(DATASETS_FOLDER, train=True, download=True, transform=transform, target_transform=target_transform)
    if train_size != -1:
        train = take_first(train, train_size)
    #train_loader = torch.utils.data.DataLoader(
    #    train_dataset,
    #    batch_size=batch_size, shuffle=True)
    
    test = datasets.MNIST(DATASETS_FOLDER, train=False, download=True, transform=transform, target_transform=target_transform)
    #test_loader = torch.utils.data.DataLoader(
    #    test_dataset,
    #    batch_size=batch_size, shuffle=False)
    
    torch.manual_seed(42)
    val, test = torch.utils.data.random_split(test, [int(0.5*len(test)), len(test) - int(0.5*len(test))])
    
    analysis_size = max(batch_size, 128)
    analysis = torch.utils.data.Subset(train, range(analysis_size))
    analysis_test = torch.utils.data.Subset(val, range(analysis_size))
    client_loaders = []

    if alpha == 0:
        randperm = np.random.permutation(len(train))
        for i in range(client_num):
            data_index = randperm[i:-1:client_num]
            client_train = torch.utils.data.Subset(train, data_index)
            client_loaders.append(torch.utils.data.DataLoader(
                            client_train,
                            batch_size=batch_size, shuffle=True))
    else:
        import random
        num_chunks = 4
        majority_index, minority_index = [], []
        class_randperm = np.random.permutation(C)
        class_chunk_perm = [(i,j) for i in range(C) for j in range(int(client_num*num_chunks/C))]
        random.seed(42)
        random.shuffle(class_chunk_perm)

        for i in range(C):
            class_index = np.where(np.array(train.targets) == i)[0].tolist()
            majority_index.append(class_index[:int(alpha * len(class_index))])
            minority_index = minority_index + class_index[int(alpha * len(class_index)):]
        minority_randperm = np.random.permutation(len(minority_index))
        minority_index = np.array(minority_index)

        for i in range(client_num):
            data_index = minority_index[minority_randperm[i::client_num]].tolist()

            """
            majority_classes = class_randperm[i::client_num]
            for class_id in majority_classes:
                data_index += majority_index[class_id]
            """
            for (class_id, chunk_id) in class_chunk_perm[i*num_chunks:(i+1)*num_chunks]:
                data_index += majority_index[class_id][chunk_id::int(client_num*num_chunks/C)]
                
            client_train = torch.utils.data.Subset(train, data_index)
            client_loaders.append(torch.utils.data.DataLoader(
                            client_train,
                            batch_size=batch_size, shuffle=True))
            #from collections import Counter
            #print(Counter(np.array(train.targets)[data_index]))
        #sys.exit()
    
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        batch_size=analysis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        batch_size=analysis_size, shuffle=False)
    
    torch.manual_seed(torch.initial_seed())
    return train_loader, client_loaders, val_loader, test_loader, analysis_loader, analysis_test_loader, input_ch, num_pixels, C, transform_to_one_hot, data_params
