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

#DATASETS_FOLDER = "/projects/dali/data/" #os.environ["DATASETS"]
DATASETS_FOLDER = "/u/lucmon/lucmon/data"

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
    print("getting my transform")
    train_transform = []
    test_transform = []
    """
    train_transform += [
        transforms.RandomCrop(size=32, padding=4)
    ]
    """
    train_transform += [
        #transforms.Resize((280, 280)),
        #transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip()]

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.247, 0.2435, 0.2616]
    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    
    test_transform += [
        #transforms.Resize((280, 280)),
        #transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform


def load_cifar(loss: str, batch_size: int, train_size = -1, augment: int = 0, tiny_analysis=False):
    data_params = {"compute_acc": True}
    input_ch = 3
    num_pixels = 32 * 32 * 3
    C = 10
    transform_to_one_hot = True
    
    if augment == 0:
        train_transform, test_transform = get_transform()
        train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True, transform=train_transform)
        test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False, transform=test_transform)
    elif augment == 1:
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
            train = take_first(train, train_size)
        test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_test)
    else:
        raise NotImplementedError
   

    #analysis_size = max(batch_size, 128)
    analysis_size = 32 if tiny_analysis else max(batch_size, 128)
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

def load_cifar_vit(model_name, batch_size, tiny_analysis=False):
    """***deprecated"""
    data_params = {"compute_acc": True}
    transform_to_one_hot = True
    C = 10
    from datasets import load_dataset
    from transformers import ViTImageProcessor

    train_ds, test_ds = load_dataset('cifar10', split=['train', 'test'])
    id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
    label2id = {label:id for id,label in id2label.items()}

    processor = ViTImageProcessor.from_pretrained(model_name)

    from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]

    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    _val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
        return examples
    
    train_ds.set_transform(train_transforms)
    test_ds.set_transform(val_transforms)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    analysis_size = 32 if tiny_analysis else max(batch_size, 128)
    analysis = torch.utils.data.Subset(train_ds, range(analysis_size))
    analysis_test = torch.utils.data.Subset(test_ds, range(analysis_size))
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        collate_fn=collate_fn, 
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        collate_fn=collate_fn, 
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        collate_fn=collate_fn, 
        batch_size=analysis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        collate_fn=collate_fn, 
        batch_size=analysis_size, shuffle=False)

    return train_loader, test_loader, analysis_loader, analysis_test_loader, id2label, label2id, C, transform_to_one_hot, data_params


def load_cifar100(loss: str, batch_size: int, train_size = -1):
    data_params = {"compute_acc": True}
    input_ch = 3
    num_pixels = 32 * 32 * 3
    C = 100
    transform_to_one_hot = True

    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    train_transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])

    train = CIFAR100(root=DATASETS_FOLDER, download=True, train=True, transform=train_transform)
    test = CIFAR100(root=DATASETS_FOLDER, download=True, train=False, transform=test_transform)

    if train_size != -1:
        train = take_first(train, batch_size)
    analysis_size = max(batch_size, 128)
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

def load_cifar_federated(loss: str, batch_size: int, train_size = -1, client_num=1, alpha=0.0):
    data_params = {"compute_acc": True}
    input_ch = 3
    num_pixels = 32 * 32 * 3
    C = 10
    transform_to_one_hot = True

    #train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
    #                                      transforms.RandomHorizontalFlip()])
    
    #cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True, transform=train_transform)
    #cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False)
    """
    #X_train = train_transform(cifar10_train.data)
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
    
    analysis_size = max(batch_size, 128)
    analysis = torch.utils.data.Subset(train, range(analysis_size))
    analysis_test = torch.utils.data.Subset(test, range(analysis_size))
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

def load_cifar_vit_federated(model_name: str, batch_size: int, train_size = -1, client_num=1, alpha=0.0):
    data_params = {"compute_acc": True}
    transform_to_one_hot = True
    C = 10
    from datasets import load_dataset
    from transformers import ViTImageProcessor

    train, test = load_dataset('cifar10', split=['train', 'test'])
    id2label = {id:label for id, label in enumerate(train.features['label'].names)}
    label2id = {label:id for id,label in id2label.items()}

    processor = ViTImageProcessor.from_pretrained(model_name)

    from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]

    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    _val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
        return examples
    
    train.set_transform(train_transforms)
    test.set_transform(val_transforms)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    analysis_size = max(batch_size, 128)
    analysis = torch.utils.data.Subset(train, range(analysis_size))
    analysis_test = torch.utils.data.Subset(test, range(analysis_size))
    client_loaders = []

    if alpha == 0:
        randperm = np.random.permutation(len(train))
        for i in range(client_num):
            data_index = randperm[i:-1:client_num]
            client_train = torch.utils.data.Subset(train, data_index)
            client_loaders.append(torch.utils.data.DataLoader(
                            client_train,
                            collate_fn=collate_fn, 
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

            for (class_id, chunk_id) in class_chunk_perm[i*num_chunks:(i+1)*num_chunks]:
                data_index += majority_index[class_id][chunk_id::int(client_num*num_chunks/C)]
                
            client_train = torch.utils.data.Subset(train, data_index)
            client_loaders.append(torch.utils.data.DataLoader(
                            client_train,
                            collate_fn=collate_fn, 
                            batch_size=batch_size, shuffle=True))
            #from collections import Counter
            #print(Counter(np.array(train.targets)[data_index]))
        #sys.exit()

    train_loader = torch.utils.data.DataLoader(
        train,
        collate_fn=collate_fn, 
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test,
        collate_fn=collate_fn, 
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        collate_fn=collate_fn, 
        batch_size=analysis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        collate_fn=collate_fn, 
        batch_size=analysis_size, shuffle=False)
    return train_loader, client_loaders, test_loader, analysis_loader, analysis_test_loader, id2label, label2id, C, transform_to_one_hot, data_params



def load_cifar100_federated(loss: str, batch_size: int, train_size = -1, client_num=1):
    data_params = {"compute_acc": True}
    input_ch = 3
    num_pixels = 32 * 32 * 3
    C = 100
    transform_to_one_hot = True

    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    train_transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])

    train = CIFAR100(root=DATASETS_FOLDER, download=True, train=True, transform=train_transform)
    test = CIFAR100(root=DATASETS_FOLDER, download=True, train=False, transform=test_transform)

    if train_size != -1:
        train = take_first(train, batch_size)
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

def load_cifar100_vit_federated(model_name: str, batch_size: int, train_size = -1, client_num=1, alpha=0.0):
    #def load_cifar100_vit_federated(n_clients, alpha, seed):
    seed = 42
    data_params = {"compute_acc": True}
    transform_to_one_hot = True
    C = 100
    from data.dirichlet import partition_dirichlet
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])
    trainset = datasets.CIFAR100(root=DATASETS_FOLDER, train=True, download=True, transform=transform)
    id2label = {} # {id:label for id, label in enumerate(trainset.features['fine_label'].names)}
    label2id = {} # {label:id for id,label in id2label.items()}
    N = len(trainset)
    trainidx = np.arange(0, int(N*0.8))
    Y_tr = np.array([trainset.targets[i] for i in trainidx])
    clientidx = partition_dirichlet(Y_tr, client_num, alpha, seed)
    clients = [torch.utils.data.Subset(trainset, trainidx[cidx]) for cidx in clientidx]
    validx = np.arange(int(N*0.8), N)
    #valset = torch.utils.data.Subset(trainset, validx)
    testset = datasets.CIFAR100(root=DATASETS_FOLDER, train=False, download=True, transform=test_transform)
    #return clients, valset, testset
    """
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["fine_label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    """
    client_loaders = [torch.utils.data.DataLoader(
                        client_train,
                        #collate_fn=collate_fn, 
                        batch_size=batch_size, shuffle=True) for client_train in clients]
    if client_num != 1:
        batch_size = 512
    analysis_size = max(batch_size, 128)
    analysis = torch.utils.data.Subset(trainset, range(analysis_size))
    analysis_test = torch.utils.data.Subset(testset, range(analysis_size))
    train_loader = torch.utils.data.DataLoader(
        trainset,
        #collate_fn=collate_fn, 
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        testset,
        #collate_fn=collate_fn, 
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        #collate_fn=collate_fn, 
        batch_size=analysis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        #collate_fn=collate_fn, 
        batch_size=analysis_size, shuffle=False)
    return train_loader, client_loaders, test_loader, analysis_loader, analysis_test_loader, id2label, label2id, C, transform_to_one_hot, data_params


def load_cifar100_vit_federated_deprecated(model_name: str, batch_size: int, train_size = -1, client_num=1, alpha=0.0):
    """*** deprecated"""
    data_params = {"compute_acc": True}
    transform_to_one_hot = True
    C = 100
    from datasets import load_dataset
    from transformers import ViTImageProcessor

    train, test = load_dataset('cifar100', split=['train', 'test'])
    id2label = {id:label for id, label in enumerate(train.features['fine_label'].names)}
    label2id = {label:id for id,label in id2label.items()}

    processor = ViTImageProcessor.from_pretrained(model_name)

    from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]

    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    _val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
        return examples
    
    train.set_transform(train_transforms)
    test.set_transform(val_transforms)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["fine_label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    analysis_size = max(batch_size, 128)
    analysis = torch.utils.data.Subset(train, range(analysis_size))
    analysis_test = torch.utils.data.Subset(test, range(analysis_size))
    client_loaders = []

    if alpha == 0:
        randperm = np.random.permutation(len(train))
        for i in range(client_num):
            data_index = randperm[i:-1:client_num]
            client_train = torch.utils.data.Subset(train, data_index)
            client_loaders.append(torch.utils.data.DataLoader(
                            client_train,
                            collate_fn=collate_fn, 
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

            for (class_id, chunk_id) in class_chunk_perm[i*num_chunks:(i+1)*num_chunks]:
                data_index += majority_index[class_id][chunk_id::int(client_num*num_chunks/C)]
                
            client_train = torch.utils.data.Subset(train, data_index)
            client_loaders.append(torch.utils.data.DataLoader(
                            client_train,
                            collate_fn=collate_fn, 
                            batch_size=batch_size, shuffle=True))
            #from collections import Counter
            #print(Counter(np.array(train.targets)[data_index]))
        #sys.exit()

    train_loader = torch.utils.data.DataLoader(
        train,
        collate_fn=collate_fn, 
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test,
        collate_fn=collate_fn, 
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        collate_fn=collate_fn, 
        batch_size=analysis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        collate_fn=collate_fn, 
        batch_size=analysis_size, shuffle=False)
    return train_loader, client_loaders, test_loader, analysis_loader, analysis_test_loader, id2label, label2id, C, transform_to_one_hot, data_params


def load_cifar100_federated_non_iid(loss: str, batch_size: int, train_size = -1, client_num=1, alpha=0):
    assert alpha < 1
    data_params = {"compute_acc": True}
    input_ch = 3
    num_pixels = 32 * 32 * 3
    C = 100
    transform_to_one_hot = True

    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    train_transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])

    train = CIFAR100(root=DATASETS_FOLDER, download=True, train=True, transform=train_transform)
    test = CIFAR100(root=DATASETS_FOLDER, download=True, train=False, transform=test_transform)

    if train_size != -1:
        train = take_first(train, batch_size)
    analysis_size = max(batch_size, 128)
    analysis = torch.utils.data.Subset(train, range(analysis_size))
    analysis_test = torch.utils.data.Subset(test, range(analysis_size))
    client_loaders = []
    #randperm = np.random.permutation(len(train))

    majority_index, minority_index = [], []
    class_randperm = np.random.permutation(C)
    for i in range(C):
        class_index = np.where(np.array(train.targets) == i)[0].tolist()
        majority_index.append(class_index[:int(alpha * len(class_index))])
        minority_index = minority_index + class_index[int(alpha * len(class_index)):]
    minority_randperm = np.random.permutation(len(minority_index))
    minority_index = np.array(minority_index)

    for i in range(client_num):
        data_index = minority_index[minority_randperm[i::client_num]].tolist()

        majority_classes = class_randperm[i::client_num]
        for class_id in majority_classes:
            data_index += majority_index[class_id]
            
        client_train = torch.utils.data.Subset(train, data_index)
        client_loaders.append(torch.utils.data.DataLoader(
                        client_train,
                        batch_size=batch_size, shuffle=True))
        #from collections import Counter
        #print(Counter(np.array(train.targets)[data_index]))
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