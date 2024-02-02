import numpy as np
import matplotlib.pyplot as plt
import torch
#from sklearn.preprocessing import normalize
import scipy
import json
from torch import nn
from torch.nn.utils import parameters_to_vector, weight_norm
from torch.utils.data import DataLoader


class weight_norm_net(nn.Module):
    def __init__(self, num_pixels, widths, init_mode, basis_var, scale):
        super().__init__()
        self.scale = scale
        self.num_pixels = num_pixels
        self.basis_std = np.sqrt(basis_var)
        self.flatten = nn.Flatten()
        self.module_list = nn.ModuleList()
        self.scale_list = []
        for l in range(len(widths)):
            prev_width = widths[l-1] if l > 0 else num_pixels
            self.module_list.append(nn.Linear(prev_width, widths[l], bias=False))
            self.scale_list.append(torch.nn.Parameter(torch.rand(widths[l])).cuda())
        self.output_layer = nn.Linear(widths[-1],10,bias=False)
        self.__initialize__(widths, init_mode)
    
    def __initialize__(self, widths, init_mode):
        for l in range(len(widths)):
            prev_width = widths[l-1] if l > 0 else self.num_pixels
            if init_mode == "O(1/sqrt{m})":
                #nn.init.normal_(self.module_list[l].weight, mean=0, std=self.basis_std/np.sqrt(prev_width))
                nn.init.uniform_(self.module_list[l].weight, a=-self.basis_std/np.sqrt(prev_width), b=self.basis_std/np.sqrt(prev_width))
            elif init_mode == "O(1)":
                nn.init.normal_(self.module_list[l].weight, mean=0, std=self.basis_std)
            else:
                assert False
        nn.init.normal_(self.output_layer.weight, mean=0, std=1/np.sqrt(widths[-1]))
        self.output_layer.weight.data = torch.nn.functional.normalize(self.output_layer.weight, dim=-1)

    def forward(self, x):
        x = self.flatten(x)
        for linear, scale_coef in zip(self.module_list, self.scale_list):
            width = x.shape[-1]
            #temp_x = torch.FloatTensor(x.detach().numpy())
            #x = torch.nn.functional.normalize(x, dim=-1)
            x = linear(x) / torch.norm(linear.weight, dim=-1)
            #for i in range(x.shape[0]):
            #    x[i] = x[i] / torch.norm(temp_x[i])
            x = self.scale * x #/ np.sqrt(width)
            #x = linear(x)
            x = torch.tanh(x)
        width = x.shape[-1]
        #print(torch.norm(self.output_layer.weight))
        x = self.output_layer(x)# / torch.norm(self.output_layer.weight)
        x = self.scale * x #/ np.sqrt(width)
        return x

class weight_norm_net_old(nn.Module):
    def __init__(self, num_pixels, widths, init_mode, basis_var, scale):
        super().__init__()
        self.scale = scale
        self.num_pixels = num_pixels
        self.basis_std = np.sqrt(basis_var)
        self.flatten = nn.Flatten()
        self.module_list = nn.ModuleList()
        for l in range(len(widths)):
            prev_width = widths[l-1] if l > 0 else num_pixels
            self.module_list.append(nn.Linear(prev_width, widths[l], bias=False))
        self.output_layer = nn.Linear(widths[-1],10,bias=False)
        self.__initialize__(widths, init_mode)
    
    def __initialize__(self, widths, init_mode):
        for l in range(len(widths)):
            prev_width = widths[l-1] if l > 0 else self.num_pixels
            if init_mode == "O(1/sqrt{m})":
                nn.init.normal_(self.module_list[l].weight, mean=0, std=self.basis_std/np.sqrt(prev_width))
            elif init_mode == "O(1)":
                nn.init.normal_(self.module_list[l].weight, mean=0, std=self.basis_std)
            else:
                assert False
        nn.init.normal_(self.output_layer.weight, mean=0, std=1/np.sqrt(widths[-1]))
        self.output_layer.weight.data = torch.nn.functional.normalize(self.output_layer.weight, dim=-1)

    def forward(self, x):
        x = self.flatten(x)
        for linear in self.module_list:
            width = x.shape[-1]
            #temp_x = torch.FloatTensor(x.detach().numpy())
            #x = torch.nn.functional.normalize(x, dim=-1)
            x = linear(x) / torch.norm(linear.weight, dim=-1)
            #for i in range(x.shape[0]):
            #    x[i] = x[i] / torch.norm(temp_x[i])
            x = self.scale * x / np.sqrt(width)
            #x = linear(x)
            x = torch.tanh(x)
        width = x.shape[-1]
        #print(torch.norm(self.output_layer.weight))
        x = self.output_layer(x)# / torch.norm(self.output_layer.weight)
        x = self.scale * x / np.sqrt(width)
        return x

class weight_norm_torch(nn.Module):
    def __init__(self, num_pixels, widths):
        super().__init__()
        self.num_pixels = num_pixels
        self.flatten = nn.Flatten()
        self.module_list = nn.ModuleList()
        for l in range(len(widths)):
            prev_width = widths[l-1] if l > 0 else num_pixels
            self.module_list.append(weight_norm(nn.Linear(prev_width, widths[l], bias=False)))
        self.output_layer = nn.Linear(widths[-1],10,bias=True)
        #self.__initialize__(widths, init_mode)

    def forward(self, x):
        x = self.flatten(x)
        for linear in self.module_list:
            x = linear(x)
            x = torch.tanh(x)
        x = self.output_layer(x)
        return x