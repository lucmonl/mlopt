import torch
from tqdm import tqdm
import torch.nn.functional as F
import sys
import numpy as np

"""
def get_activation_pattern(graphs, model, device, train_loader):
    activation = []

    w_plus, w_minus = model.w_plus, model.w_minus
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(train_loader, start=1):
            data = data.to(device)
            activation_plus,  activation_minus= ((data @ w_plus) >= 0).cpu().numpy(), ((data @ w_minus) >= 0).cpu().numpy() # B * P * n_filter
            activation_plus = activation_plus.reshape(activation_plus.shape[0], -1)
            activation_minus = activation_minus.reshape(activation_minus.shape[0], -1)
            activation.append(np.concatenate([activation_plus, activation_minus], axis=-1)) #B * (2P * n_filter)
                                
    activation = np.concatenate(activation, axis=0) #train_size * 2P
    graphs.activation_pattern.append(activation)
"""

def get_activation_pattern(graphs, model, device, train_loader):
    activation = []

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(train_loader, start=1):
            data = data.to(device)
            activation_pattern = model.get_activation(data)
            activation.append(activation_pattern.detach().cpu().numpy()) #B * (2P * n_filter)
                                
    activation = np.concatenate(activation, axis=0) #train_size * 2P
    graphs.activation_pattern.append(activation)