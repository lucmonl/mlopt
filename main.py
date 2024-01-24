import os
import sys

import torch
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import gc
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models

from tqdm import tqdm
from collections import OrderedDict
#os.environ["SCIPY_USE_PROPACK"] =  "1"
from scipy.sparse.linalg import svds

from IPython import embed

from optimizer.sam import SAM, disable_running_stats, enable_running_stats
from normalized_sgd import Normalized_Optimizer
from goldstein import Goldstein
from data import load_cifar, load_mnist
 
# setting path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
print(parent_dir)
from utilities import get_hessian_eigenvalues_weight_decay, get_hessian_eigenvalues
     

def train(model, criterion, device, num_classes, train_loader, optimizer, epoch):
    model.train()
    
    pbar = tqdm(total=len(train_loader), position=0, leave=True)

    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        if data.shape[0] != batch_size:
                continue
        
        data, target = data.to(device), target.to(device)

        #if opt_name == 'sam':
        #    enable_running_stats(model)

        optimizer.zero_grad()
        out = model(data)
        if str(criterion) == 'CrossEntropyLoss()':
            loss = criterion(out, target)
        elif str(criterion) == 'MSELoss()':
            #print(out[0], target[0])
            loss = criterion(out, F.one_hot(target, num_classes=num_classes).float()) * num_classes
            #loss = criterion(out, target).float() * num_classes
        
        loss.backward()
        if opt_name == "sam":
            optimizer.first_step(zero_grad=True)
            # second forward-backward step
            disable_running_stats(model)
            out = model(data)
            loss = criterion(out, F.one_hot(target, num_classes=num_classes).float()) * num_classes
            #loss = criterion(out, target).float() * num_classes
            loss.backward()
            optimizer.second_step(zero_grad=True)
            enable_running_stats(model)
        elif opt_name == "goldstein":
            gold_iters = 0
            while gold_iters < 1:
                gold_iters += 1
                optimizer.first_step(zero_grad=True)
                disable_running_stats(model)
                out = model(data)
                loss = criterion(out, F.one_hot(target, num_classes=num_classes).float()) * num_classes
                #loss = criterion(out, target).float() * num_classes
                loss.backward()
                optimizer.second_step(zero_grad=False)
            optimizer.third_step(zero_grad=True)
            enable_running_stats(model)
        else:
            optimizer.step()

        accuracy = torch.mean((torch.argmax(out,dim=1)==target).float()).item()

        pbar.update(1)
        pbar.set_description(
            'Train\t\tEpoch: {} [{}/{} ({:.0f}%)] \t'
            'Batch Loss: {:.6f} \t'
            'Batch Accuracy: {:.6f}'.format(
                epoch,
                batch_idx,
                len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item(),
                accuracy))
        
        if debug and batch_idx > 20:
            break

    pbar.close()
    """
    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, F.one_hot(target, num_classes=num_classes).float())
        print("loss after train:", loss.item())
    """
    #eigs, _ = get_hessian_eigenvalues_weight_decay(model, criterion_summed, weight_decay, analysis_loader, neigs=10, num_classes=10, device=device)
    #print("in train:", eigs)

def analysis(graphs, model, criterion_summed, device, num_classes, loader_abridged, test_loader):
    model.train()
    disable_running_stats(model)
    eigs, _ = get_hessian_eigenvalues_weight_decay(model, criterion_summed, weight_decay, loader_abridged, neigs=10, num_classes=num_classes, device=device)
    #eigs, _, _, _ = get_hessian_eigenvalues(model, criterion_summed, lr, analysis_dataset, neigs=10, return_smallest = False)
    graphs.eigs.append(eigs[0].item())
    print("train eigs:", graphs.eigs)

    loss_sum = 0
    accuracy_sum = 0
    for batch_idx, (data, target) in enumerate(loader_abridged, start=1):
        data, target = data.to(device), target.to(device)
        out = model(data)
        if str(criterion) == 'CrossEntropyLoss()':
            loss = criterion(out, target)
        elif str(criterion) == 'MSELoss()':
            #print(out[0], target[0])
            loss = criterion(out, F.one_hot(target, num_classes=num_classes).float()) * num_classes
            #loss = criterion_summed(out, target).float()

        accuracy = torch.sum((torch.argmax(out,dim=1)==target).float()).item()
        loss_sum += loss.item()
        accuracy_sum += accuracy
    graphs.loss.append(loss_sum / len(loader_abridged.dataset))
    graphs.accuracy.append(accuracy_sum / len(loader_abridged.dataset))

    enable_running_stats(model)

    model.eval()
    pbar = tqdm(total=len(test_loader), position=0, leave=True)
    loss_sum = 0
    accuracy_sum = 0
    for batch_idx, (data, target) in enumerate(test_loader, start=1):
        data, target = data.to(device), target.to(device)
        out = model(data)
        if str(criterion) == 'CrossEntropyLoss()':
            loss = criterion(out, target)
        elif str(criterion) == 'MSELoss()':
            #print(out[0], target[0])
            loss = criterion(out, F.one_hot(target, num_classes=num_classes).float()) * num_classes
            #loss = criterion_summed(out, target).float()

        accuracy = torch.sum((torch.argmax(out,dim=1)==target).float()).item()

        pbar.update(1)
        pbar.set_description(
            'Test\t\t [{}/{} ({:.0f}%)] \t'
            'Batch Loss: {:.6f} \t'
            'Batch Accuracy: {:.6f}'.format(
                batch_idx,
                len(test_loader),
                100. * batch_idx / len(test_loader),
                (loss / data.shape[0]).item(),
                accuracy / data.shape[0]))
        loss_sum += loss.item()
        accuracy_sum += accuracy
    pbar.close()
    graphs.test_loss.append(loss_sum / len(test_loader.dataset))
    graphs.test_accuracy.append(accuracy_sum / len(test_loader.dataset))
    print("Mean Test Loss: {} \t Accuarcy: {}".format(graphs.test_loss[-1], graphs.test_accuracy[-1]))

class features:
    pass

def hook(self, input, output):
    features.value = input[0].clone()

class graphs:
    def __init__(self):
        self.log_epochs   = []
        self.accuracy     = []
        self.loss         = []
        self.eigs         = []

        self.test_loss    = []
        self.test_accuracy= []

def get_directory(lr, weight_decay, batch_size, epochs):
    results_dir = "results"
    directory = f"{results_dir}/{dataset_name}/{opt_name}/lr_{lr}/wd_{weight_decay}/batch_size_{batch_size}/epoch_{epochs}/"
    return directory

def continue_training(lr, weight_decay, batch_size, epochs):
    results_dir = "results"
    lookup_dir = f"{results_dir}/{dataset_name}/{opt_name}/lr_{lr}/wd_{weight_decay}/batch_size_{batch_size}/"
    if not os.path.exists(lookup_dir):
        return 0
    epoch_dir = os.listdir(lookup_dir)
    trained_epochs = [int(x.split("_")[-1]) for x in epoch_dir]
    trained_epochs.sort(reverse=True)
    load_from_epoch = 0
    for trained_epoch in trained_epochs:
        if trained_epoch < epochs:
            load_from_epoch = trained_epoch
            break
    return load_from_epoch
    
if __name__ == "__main__":
    debug = False # Only runs 20 batches per epoch for debugging

    # dataset parameters
    dataset_name        = "cifar"
    C                   = 10

    # Optimization Criterion
    # loss_name = 'CrossEntropyLoss'
    loss_name = 'MSELoss'
    opt_name = 'goldstein'#'norm-sgd'

    # Optimization hyperparameters
    lr_decay            = 1# 0.1

    epochs              = 80000
    epochs_lr_decay     = [epochs//3, epochs*2//3]

    batch_size          = 512 # 128

    #hyperparameters for sam
    sam_rho = 0.1
    sam_adaptive = False

    #hyperparameters for gold
    gold_delta = 0.05

    # Best lr after hyperparameter tuning
    if loss_name == 'CrossEntropyLoss':
        lr = 0.0679
    elif loss_name == 'MSELoss':
        lr = 0.005 #0.0184
    momentum            = 0 # 0.9
    weight_decay        = 0 # 5e-4 * 10

    if dataset_name == "cifar":
        train_loader, test_loader, analysis_loader, input_ch = load_cifar(loss_name, batch_size)
    elif dataset_name == "mnist":
        train_loader, test_loader, analysis_loader, input_ch = load_mnist(loss_name, batch_size)

    load_from_epoch = continue_training(lr, weight_decay, batch_size, epochs)

    # analysis parameters
    """
    epoch_list          = [1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11,
                        12,  13,  14,  16,  17,  19,  20,  22,  24,  27,   29,
                        32,  35,  38,  42,  45,  50,  54,  59,  65,  71,   77,
                        85,  92,  101, 110, 121, 132, 144, 158, 172, 188,  206,
                        225, 245, 268, 293, 320, 350]
    """
    #epoch_list = np.arange(1, epochs+1, 100).tolist()
    epoch_list = np.arange(load_from_epoch+1, epochs+1, 1000).tolist()

    train_graphs = graphs()
    
    model = models.resnet18(pretrained=False, num_classes=C)
    if dataset_name == "mnist":
        model.conv1 = nn.Conv2d(input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias=False) # Small dataset filter size used by He et al. (2015)
        model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
    if load_from_epoch != 0:
        print("loading from trained epoch {}".format(load_from_epoch))
        load_from_dir = get_directory(lr, weight_decay, batch_size, load_from_epoch)
        model.load_state_dict(torch.load(os.path.join(load_from_dir, "model.ckpt")))
        with open(f'{load_from_dir}/train_graphs.pk', 'rb') as f:
            train_graphs = pickle.load(f)
    model = model.to(device)


    # register hook that saves last-layer input into features
    classifier = model.fc
    classifier.register_forward_hook(hook)


    if loss_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
        criterion_summed = nn.CrossEntropyLoss(reduction='sum')
    elif loss_name == 'MSELoss':
        criterion = nn.MSELoss()
        criterion_summed = nn.MSELoss(reduction='sum')

    if opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    elif opt_name == "sam":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=sam_rho, adaptive=sam_adaptive, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == "norm-sgd":
        base_optimizer = torch.optim.SGD
        optimizer = Normalized_Optimizer(model.parameters(), base_optimizer,
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    elif opt_name == "goldstein":
        base_optimizer = torch.optim.SGD
        optimizer = Goldstein(model.parameters(), base_optimizer, delta=gold_delta, lr=lr, momentum=momentum, weight_decay=weight_decay)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=epochs_lr_decay,
                                                gamma=lr_decay)

    
    directory = get_directory(lr, weight_decay, batch_size, epochs)
    os.makedirs(directory, exist_ok=True)

    import pickle

    cur_epochs = []
    for epoch in range(load_from_epoch+1, epochs + 1):
        train(model, criterion, device, C, train_loader, optimizer, epoch)
        #lr_scheduler.step()

        if epoch in epoch_list:
            train_graphs.log_epochs.append(epoch)
            analysis(train_graphs, model, criterion_summed, device, C, analysis_loader, test_loader)
            pickle.dump(train_graphs, open(f"{directory}/train_graphs.pk", "wb"))
            torch.save(model.state_dict(), f"{directory}/model.ckpt")