import numpy as np
import matplotlib.pyplot as plt
import torch
#from sklearn.preprocessing import normalize
import scipy
import json
from torch import nn
from torch.utils.data import DataLoader

class mlp_sim_bn(nn.Module):
    def __init__(self, num_pixels, C):
        super().__init__()
        width = 50
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(num_pixels, width, bias=True)
        self.linear_2 = nn.Linear(width, C, bias=False)
        self.gamma = torch.nn.parameter.Parameter(torch.randn(1), requires_grad=True)

    def simple_batch_norm(self, batched_x):
        simple_var = torch.sqrt(torch.mean(batched_x**2, dim=0))
        return self.gamma * batched_x / (simple_var + 1e-6)

    def forward(self, batched_x):
        batched_x = self.flatten(batched_x)
        batched_x = torch.nn.ReLU()(self.linear_1(batched_x))
        batched_x = self.simple_batch_norm(batched_x)
        batched_x = self.linear_2(batched_x)
        return batched_x.squeeze()

