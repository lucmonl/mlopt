import numpy as np
import torch 
from torchvision import datasets, transforms
from pyhessian import hessian # Hessian computation
import sys
from utilities import dict_to_

def compute_eigen_spectrum(graphs, model, criterion, analysis_loader, device):
    for _, inputs in enumerate(analysis_loader):
        data, target = inputs
        data, target = data.to(device), target.to(device)
        hessian_comp = hessian(model, criterion, data=(data, target), cuda=True)
        break
    density_eigen, density_weight = hessian_comp.density()

    graphs.density_eigen.append(density_eigen)
    graphs.density_weight.append(density_weight)