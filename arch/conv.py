import torch
from torch import nn
import math

class conv_fixed_last_layer(nn.Module):
    "The model is motivated by https://arxiv.org/pdf/2310.07269.pdf 2.2"
    """All Flat Minima cannot generalize well."""
    def __init__(self, feat_dim, num_filters):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_filters = num_filters
        self.std_v = 1/math.sqrt(self.feat_dim)
        self.w_plus  = torch.nn.parameter.Parameter(torch.empty(feat_dim, num_filters), requires_grad=True)
        self.w_minus = torch.nn.parameter.Parameter(torch.empty(feat_dim, num_filters), requires_grad=True)
        #self.v_plus = torch.nn.parameter.Parameter(torch.empty(1,), requires_grad=True)
        #self.v_minus = torch.nn.parameter.Parameter(torch.empty(1,), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.w_plus.data.uniform_(-self.std_v, self.std_v)
        self.w_minus.data.uniform_(-self.std_v, self.std_v)
        #self.v_plus.data.uniform_(0,1)
        #self.v_minus.data.uniform_(0,1)

    def forward(self, batched_x):
        batched_x_plus = torch.nn.ReLU()(batched_x @ self.w_plus) ## (B*P*d) * (d*width)
        batched_x_minus = torch.nn.ReLU()(batched_x @ self.w_minus)

        #batched_x = 1/self.num_filters*(self.v_plus*torch.sum(batched_x_plus, [1,2]) - self.v_minus*torch.sum(batched_x_minus, [1,2]))
        batched_x = 1/self.num_filters*(torch.sum(batched_x_plus, [1,2]) - torch.sum(batched_x_minus, [1,2]))
        #batched_x2 = torch.zeros_like(batched_x)
        #return torch.stack([batched_x, batched_x2], dim=1)
        return batched_x
    

class conv_with_last_layer(nn.Module):
    "The model is motivated by https://arxiv.org/pdf/2310.07269.pdf 2.2"
    """All Flat Minima cannot generalize well."""
    def __init__(self, feat_dim, num_filters):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_filters = num_filters
        self.std_v = 1/math.sqrt(self.feat_dim)
        self.w_plus  = torch.nn.parameter.Parameter(torch.empty(feat_dim, num_filters), requires_grad=True)
        self.w_minus = torch.nn.parameter.Parameter(torch.empty(feat_dim, num_filters), requires_grad=True)
        self.v_plus = torch.nn.parameter.Parameter(torch.empty(1,), requires_grad=True)
        self.v_minus = torch.nn.parameter.Parameter(torch.empty(1,), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.w_plus.data.uniform_(-self.std_v, self.std_v)
        self.w_minus.data.uniform_(-self.std_v, self.std_v)
        self.v_plus.data.uniform_(0,1)
        self.v_minus.data.uniform_(0,1)

    def forward(self, batched_x):
        batched_x_plus = torch.nn.ReLU()(batched_x @ self.w_plus) ## (B*P*d) * (d*width)
        batched_x_minus = torch.nn.ReLU()(batched_x @ self.w_minus)

        batched_x = 1/self.num_filters*(self.v_plus*torch.sum(batched_x_plus, [1,2]) - self.v_minus*torch.sum(batched_x_minus, [1,2]))
        #batched_x = 1/self.num_filters*(torch.sum(batched_x_plus, [1,2]) - torch.sum(batched_x_minus, [1,2]))
        #batched_x2 = torch.zeros_like(batched_x)
        #return torch.stack([batched_x, batched_x2], dim=1)
        return batched_x