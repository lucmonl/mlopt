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
        out_init = torch.rand(1)
        #self.v_plus.data.uniform_(0,1)
        #self.v_minus.data.uniform_(0,1)
        self.v_plus.data, self.v_minus.data = out_init, out_init

    def forward(self, batched_x):
        batched_x_plus = torch.nn.ReLU()(batched_x @ self.w_plus) ## (B*P*d) * (d*width)
        batched_x_minus = torch.nn.ReLU()(batched_x @ self.w_minus)

        batched_x = 1/(self.num_filters**0.5)*(self.v_plus*torch.sum(batched_x_plus, [1,2]) - self.v_minus*torch.sum(batched_x_minus, [1,2]))
        #batched_x = self.v_plus*torch.sum(batched_x_plus, [1,2]) - self.v_minus*torch.sum(batched_x_minus, [1,2])
        #batched_x = 1/self.num_filters*(torch.sum(batched_x_plus, [1,2]) - torch.sum(batched_x_minus, [1,2]))
        #batched_x2 = torch.zeros_like(batched_x)
        #return torch.stack([batched_x, batched_x2], dim=1)
        return batched_x
    


class scalarized_conv(nn.Module):
    # make w*signal and w*noise to be one model
    def __init__(self, num_filters, patch_dim):
        super().__init__()
        self.num_filters = num_filters
        self.std_v = 1/math.sqrt(self.num_filters)
        self.w_plus  = torch.nn.parameter.Parameter(torch.empty(patch_dim, num_filters), requires_grad=True)
        self.w_minus = torch.nn.parameter.Parameter(torch.empty(patch_dim, num_filters), requires_grad=True)
        self.v_plus = torch.nn.parameter.Parameter(torch.empty(patch_dim,), requires_grad=True)
        self.v_minus = torch.nn.parameter.Parameter(torch.empty(patch_dim,), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        std_v = 0.5
        self.w_plus.data.uniform_(-std_v, std_v)
        self.w_minus.data.uniform_(-std_v, std_v)
        #out_init = torch.rand(1)
        self.v_plus.data.uniform_(0,std_v)
        self.v_minus.data.uniform_(0,std_v)
        #self.v_plus.data, self.v_minus.data = out_init, out_init

    def forward(self, batched_x):
        batched_x_plus = torch.nn.ReLU()(torch.einsum('ij,jk->ijk', batched_x, self.w_plus)) ## (B*P) * (P*width) -> (B * P * width)
        batched_x_minus = torch.nn.ReLU()(torch.einsum('ij,jk->ijk', batched_x, self.w_minus))

        batched_x_plus = torch.einsum('ijk,j->ijk', batched_x_plus, self.v_plus)
        batched_x_minus = torch.einsum('ijk,j->ijk', batched_x_minus, self.v_minus)

        batched_x = 1/(self.num_filters**0.5)*(torch.sum(batched_x_plus, [1,2]) - torch.sum(batched_x_minus, [1,2]))
        #batched_x = self.v_plus*torch.sum(batched_x_plus, [1,2]) - self.v_minus*torch.sum(batched_x_minus, [1,2])
        #batched_x = 1/self.num_filters*(torch.sum(batched_x_plus, [1,2]) - torch.sum(batched_x_minus, [1,2]))
        #batched_x2 = torch.zeros_like(batched_x)
        #return torch.stack([batched_x, batched_x2], dim=1)
        return batched_x
    
    def get_activation(self, batched_x):
        batched_x_plus = (torch.einsum('ij,jk->ijk', batched_x, self.w_plus) > 0).int() ## (B*P) * (P*width) -> (B * P * width)
        batched_x_minus = (torch.einsum('ij,jk->ijk', batched_x, self.w_minus) > 0).int()
        activation = torch.cat([batched_x_plus.reshape(batched_x_plus.shape[0], -1), 
                                batched_x_minus.reshape(batched_x_plus.shape[0], -1)], axis=-1) 
        return activation