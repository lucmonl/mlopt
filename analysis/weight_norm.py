import numpy as np
import matplotlib.pyplot as plt
import torch
#from sklearn.preprocessing import normalize
import scipy
import json
from torch import nn
from torch.nn.utils import parameters_to_vector, weight_norm
from torch.utils.data import DataLoader
import torch.nn.functional as F

def get_min_weight_norm(graphs, model, C, model_name):
    norm_min = 1e10
    for name, param in model.named_parameters():
        print(name, param.shape)
        """
        if 'weight_g' in name:  #if param.shape[1] == 1:
            continue
        if 'output_layer' in name:
            continue
        print(name)
        """
        if model_name == "weight_norm":
            if 'output_layer' in name:
                continue
            norm_min = min(norm_min, torch.min(torch.norm(param, dim=-1)).cpu().detach().numpy())
        if model_name == 'weight_norm_torch' and 'weight_v' in name:
            norm_min = min(norm_min, torch.min(torch.norm(param, dim=-1)).cpu().detach().numpy())
    graphs.wn_norm_min.append(norm_min)

def get_grad_loss_ratio(graphs, model, loss_name, loader, criterion, criterion_summed, num_classes, device):
    loss_sum = 0
    for batch_idx, (data, target) in enumerate(loader, start=1):
        data, target = data.to(device), target.to(device)
        out = model(data)
        if loss_name == 'CrossEntropyLoss':
            loss = criterion(out, target)
        elif loss_name == 'MSELoss':
            #print(out[0], target[0])
            #loss = criterion(out, F.one_hot(target, num_classes=num_classes).float()) * num_classes
            #loss = criterion_summed(out, F.one_hot(target, num_classes=num_classes).float()).float()
            loss = criterion_summed(out, target)
            #print(loss)
        loss_sum += loss
    loss_mean = loss_sum / len(loader.dataset)
    loss_mean.backward()
    grad_norm = 0
    for param in model.parameters():
        grad_norm += torch.norm(param.grad)**2
    graphs.wn_grad_loss_ratio.append((grad_norm/loss_mean).item())
    print(graphs.wn_grad_loss_ratio[-1])
    print(graphs.wn_norm_min[-1])

def get_min_weight_norm_with_g(graphs, model, C, model_name):
    norm_min = 1e10
    for name, param in model.named_parameters():
        #print(name)
        print(name, param.shape)
        """
        if 'weight_g' in name:  #if param.shape[1] == 1:
            continue
        if 'output_layer' in name:
            continue
        print(name)
        """
        if model_name == "weight_norm":
            if 'output_layer' in name:
                continue
            norm_min = min(norm_min, torch.min(torch.norm(param, dim=-1)).cpu().detach().numpy())
        if model_name == 'weight_norm_torch' and 'weight_g' in name:
            weight_g = param
        if model_name == 'weight_norm_torch' and 'weight_v' in name:
            print((weight_g * param).shape)
            norm_min = min(norm_min, torch.min(torch.norm(weight_g * param, dim=-1)).cpu().detach().numpy())
    graphs.wn_norm_min_with_g.append(norm_min)