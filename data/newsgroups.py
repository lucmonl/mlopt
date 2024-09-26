import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.datasets import fetch_20newsgroups
import os
import json
import numpy as np
from PIL import Image
from data.dirichlet import partition_dirichlet
#from train_utils import test_batch_cls, test_batch_nwp

#DATA = "/projects/dali/data/" 
DATA = "/u/lucmon/lucmon/data"



def build_20newsgroups():
    train_pt = f"{DATA}/20newsgroups/20newsgroups_train.pt"
    test_pt = f"{DATA}/20newsgroups/20newsgroups_test.pt"
    if not os.path.exists(train_pt) or not os.path.exists(test_pt):
        generate_20newsgroups_dump()
    tr_d = torch.load(train_pt)
    ev_d = torch.load(test_pt)
    trainset = list(zip(tr_d['X'], tr_d['Y']))
    testset = list(zip(ev_d['X'], ev_d['Y']))
    return trainset, testset

def build_20newsgroups_federated(n_clients, alpha, seed):
    train_pt = f"{DATA}/20newsgroups/20newsgroups_train.pt"
    test_pt = f"{DATA}/20newsgroups/20newsgroups_test.pt"
    if not os.path.exists(train_pt) or not os.path.exists(test_pt):
        generate_20newsgroups_dump()
    tr_d = torch.load(train_pt)
    ev_d = torch.load(test_pt)
    trainset = list(zip(tr_d['X'], tr_d['Y']))
    testset = list(zip(ev_d['X'], ev_d['Y']))
    N = len(trainset)
    trainidx = np.arange(0, int(N*0.8))
    Y_tr = tr_d['Y'][trainidx]
    clientidx = partition_dirichlet(Y_tr, n_clients, alpha, seed)
    clients = [torch.utils.data.Subset(trainset, trainidx[cidx]) for cidx in clientidx]
    """
    for cidx in clientidx:
        print(Y_tr[cidx])
    import sys
    sys.exit()
    """
    validx = np.arange(int(N*0.8), N)
    valset = torch.utils.data.Subset(trainset, validx)
    return trainset, clients, valset, testset

def generate_20newsgroups_dump():
    print("Generating 20newsgroups cache...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token_id = 50256
    ng_train = fetch_20newsgroups(subset='train')
    tr_X = torch.LongTensor([tokenizer.encode(x, max_length=128, padding='max_length', truncation=True) for x in ng_train['data']])

    ng_test = fetch_20newsgroups(subset='test')
    ev_X = torch.LongTensor([tokenizer.encode(x, max_length=128, padding='max_length', truncation=True) for x in ng_test['data']])

    tr_Y = torch.LongTensor(ng_train['target'])
    ev_Y = torch.LongTensor(ng_test['target'])

    os.makedirs(f"{DATA}/20newsgroups", exist_ok=True)
    torch.save({'X': tr_X, 'Y': tr_Y}, f"{DATA}/20newsgroups/20newsgroups_train.pt")
    torch.save({'X': ev_X, 'Y': ev_Y}, f"{DATA}/20newsgroups/20newsgroups_test.pt")

def load_20newsgroups_federated(loss: str, batch_size: int, client_num=1, alpha=0.1):
    data_params = {"compute_acc": True}
    C = 20
    transform_to_one_hot = True
    trainset, clients, _, testset = build_20newsgroups_federated(client_num, alpha, seed=42)
    analysis_size = max(batch_size, 128)
    analysis = torch.utils.data.Subset(trainset, range(analysis_size))
    analysis_test = torch.utils.data.Subset(testset, range(analysis_size))

    client_loaders = [DataLoader(client, batch_size=batch_size, shuffle=True, num_workers=0) for client in clients]
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        batch_size=analysis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        batch_size=analysis_size, shuffle=False)
    return train_loader, client_loaders, test_loader, analysis_loader, analysis_test_loader, C, transform_to_one_hot, data_params


def load_20newsgroups(loss: str, batch_size: int):
    data_params = {"compute_acc": True}
    C = 20
    transform_to_one_hot = True
    trainset, testset = build_20newsgroups()
    analysis_size = max(batch_size, 128)
    analysis = torch.utils.data.Subset(trainset, range(analysis_size))
    analysis_test = torch.utils.data.Subset(testset, range(analysis_size))

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        batch_size=analysis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        batch_size=analysis_size, shuffle=False)
    return train_loader, test_loader, analysis_loader, analysis_test_loader, C, transform_to_one_hot, data_params