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
        self.w_plus  = torch.nn.parameter.Parameter(torch.randn(feat_dim, num_filters), requires_grad=True)
        self.w_minus = torch.nn.parameter.Parameter(torch.randn(feat_dim, num_filters), requires_grad=True)

    def forward(self, batched_x):
        batched_x_plus = torch.nn.ReLU()(batched_x @ self.w_plus) ## (B*P*d) * (d*width)
        batched_x_minus = torch.nn.ReLU()(batched_x @ self.w_minus)

        batched_x = 1/self.num_filters*(torch.sum(batched_x_plus, [1,2]) - torch.sum(batched_x_minus, [1,2])) / math.sqrt(self.feat_dim)
        #batched_x2 = torch.zeros_like(batched_x)
        #return torch.stack([batched_x, batched_x2], dim=1)
        return batched_x
