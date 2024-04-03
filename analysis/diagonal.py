import torch
from tqdm import tqdm
import torch.nn.functional as F
import sys
import numpy as np

def get_diagonal_coef(graphs, model, device, train_loader):
    diagonal_coef = []
    v_plus, v_minus = model.v_plus, model.v_minus # (P)
    w_plus, w_minus = model.w_plus, model.w_minus # (P, m)
    v = torch.cat([v_plus, v_minus], axis=-1) 
    w = torch.cat([w_plus.reshape(-1), w_minus.reshape(-1)], axis=-1) # (P * 2m)

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(train_loader, start=1):
            data = data.to(device)
            activation_pattern = model.get_activation(data)
            activated_neuron =  activation_pattern * w # (B, P*2m) 
            activated_neuron = activated_neuron.reshape(data.shape[0], w_plus.shape[0] * 2, w_plus.shape[1]) #(B, 2P, m)
            activated_neuron = torch.sum(activated_neuron, axis=-1) #(B, 2P)
            diagonal_coef.append((activated_neuron * v).detach().cpu().numpy()) # (B, 2P)
                                
    diagonal_coef = np.concatenate(diagonal_coef, axis=0) #train_size * 2P
    graphs.diagonal_coef.append(diagonal_coef)

def get_diagonal_invariate(graphs, model, device, train_loader):
    diagonal_coef = []
    v_plus, v_minus = model.v_plus, model.v_minus # (P)
    w_plus, w_minus = model.w_plus, model.w_minus # (P, m)
    v = torch.cat([v_plus, v_minus], axis=-1) 
    w = torch.cat([w_plus.reshape(-1), w_minus.reshape(-1)], axis=-1) / np.sqrt(w_plus.shape[-1]) # (P * 2m)

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(train_loader, start=1):
            data = data.to(device)
            activation_pattern = model.get_activation(data)
            activated_neuron =  activation_pattern * w # (B, P*2m) 
            activated_neuron = activated_neuron.reshape(data.shape[0], w_plus.shape[0] * 2, w_plus.shape[1]) #(B, 2P, m)
            activated_neuron = torch.sum(activated_neuron, axis=-1) #(B, 2P)
            diagonal_coef.append((activated_neuron**2 - v**2).detach().cpu().numpy()) # (B, 2P)
                                
    diagonal_coef = np.concatenate(diagonal_coef, axis=0) #train_size * 2P
    graphs.diagonal_invariate.append(diagonal_coef)