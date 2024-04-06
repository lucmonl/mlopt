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
    def __init__(self, num_pixels, widths, init_mode, basis_var, scale, C):
        super().__init__()
        self.scale = scale
        self.num_pixels = num_pixels
        self.basis_std = np.sqrt(basis_var)
        self.flatten = nn.Flatten()
        self.module_list = nn.ModuleList()
        for l in range(len(widths)):
            prev_width = widths[l-1] if l > 0 else num_pixels
            self.module_list.append(nn.Linear(prev_width, widths[l], bias=False))
        self.output_layer = nn.Linear(widths[-1],C,bias=False)
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
        #for linear, scale_coef in zip(self.module_list, self.scale_list):
        for linear in self.module_list:
            x = linear(x) / torch.norm(linear.weight, dim=-1)
            x = self.scale * x
            x = torch.tanh(x)
        x = self.output_layer(x)
        x = self.scale * x
        return x.squeeze()

class weight_norm_net_old(nn.Module):
    def __init__(self, num_pixels, widths, init_mode, basis_var, scale, C):
        super().__init__()
        self.scale = scale
        self.num_pixels = num_pixels
        self.basis_std = np.sqrt(basis_var)
        self.flatten = nn.Flatten()
        self.module_list = nn.ModuleList()
        for l in range(len(widths)):
            prev_width = widths[l-1] if l > 0 else num_pixels
            self.module_list.append(nn.Linear(prev_width, widths[l], bias=False))
        self.output_layer = nn.Linear(widths[-1],C,bias=False)
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
    def __init__(self, num_pixels, widths, C):
        super().__init__()
        self.num_pixels = num_pixels
        self.flatten = nn.Flatten()
        self.module_list = nn.ModuleList()
        for l in range(len(widths)):
            prev_width = widths[l-1] if l > 0 else num_pixels
            self.module_list.append(weight_norm(nn.Linear(prev_width, widths[l], bias=False)))
        self.output_layer = nn.Linear(widths[-1],C,bias=True)
        #self.__initialize__(widths, init_mode)

    def forward(self, x):
        x = self.flatten(x)
        for linear in self.module_list:
            x = linear(x)
            x = torch.tanh(x)
        x = self.output_layer(x)
        return x
    
from torch.nn.parameter import Parameter, UninitializedParameter
from torch import _weight_norm, norm_except_dim
from typing import Any, TypeVar
import warnings
#from ..modules import Module

#__all__ = ['WeightNorm', 'weight_norm', 'remove_weight_norm']

class WeightNorm_woG:
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    # TODO Make return type more specific
    def compute_weight(self, module: nn.Module) -> Any:
        g = torch.ones_like(getattr(module, self.name + '_g'), requires_grad=False)
        v = getattr(module, self.name + '_v')
        return _weight_norm(v, g, self.dim)

    @staticmethod
    def apply(module, name: str, dim: int) -> 'WeightNorm_WoG':
        warnings.warn("torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.")

        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm_woG) and hook.name == name:
                raise RuntimeError(f"Cannot register two weight_norm hooks on the same parameter {name}")

        if dim is None:
            dim = -1

        fn = WeightNorm_woG(name, dim)

        weight = getattr(module, name)
        if isinstance(weight, UninitializedParameter):
            raise ValueError(
                'The module passed to `WeightNorm` can\'t have uninitialized parameters. '
                'Make sure to run the dummy forward before applying weight normalization')
        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(name + '_g', Parameter(norm_except_dim(weight, 2, dim).data))
        module.register_parameter(name + '_v', Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: nn.Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        del module._parameters[self.name + '_v']
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: nn.Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))


T_module = TypeVar('T_module', bound=nn.Module)

def weight_norm_woG(module: T_module, name: str = 'weight', dim: int = 0) -> T_module:
    r"""Apply weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    .. warning::

        This function is deprecated.  Use :func:`torch.nn.utils.parametrizations.weight_norm`
        which uses the modern parametrization API.  The new ``weight_norm`` is compatible
        with ``state_dict`` generated from old ``weight_norm``.

        Migration guide:

        * The magnitude (``weight_g``) and direction (``weight_v``) are now expressed
          as ``parametrizations.weight.original0`` and ``parametrizations.weight.original1``
          respectively.  If this is bothering you, please comment on
          https://github.com/pytorch/pytorch/issues/102999

        * To remove the weight normalization reparametrization, use
          :func:`torch.nn.utils.parametrize.remove_parametrizations`.

        * The weight is no longer recomputed once at module forward; instead, it will
          be recomputed on every access.  To restore the old behavior, use
          :func:`torch.nn.utils.parametrize.cached` before invoking the module
          in question.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    WeightNorm_woG.apply(module, name, dim)
    return module



def remove_weight_norm(module: T_module, name: str = 'weight') -> T_module:
    r"""Remove the weight normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm_woG) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError(f"weight_norm of '{name}' not found in {module}")

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicUnit(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super(BasicUnit, self).__init__()
        self.block = nn.Sequential(OrderedDict([
            ("0_normalization", nn.BatchNorm2d(channels)),
            ("1_activation", nn.ReLU(inplace=True)),
            ("2_convolution", weight_norm_woG(nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False))),
            ("3_normalization", nn.BatchNorm2d(channels)),
            ("4_activation", nn.ReLU(inplace=True)),
            ("5_dropout", nn.Dropout(dropout, inplace=True)),
            ("6_convolution", weight_norm_woG(nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False))),
        ]))

    def forward(self, x):
        return x + self.block(x)


class DownsampleUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float):
        super(DownsampleUnit, self).__init__()
        self.norm_act = nn.Sequential(OrderedDict([
            ("0_normalization", nn.BatchNorm2d(in_channels)),
            ("1_activation", nn.ReLU(inplace=True)),
        ]))
        self.block = nn.Sequential(OrderedDict([
            ("0_convolution", weight_norm_woG(nn.Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=1, bias=False))),
            ("1_normalization", nn.BatchNorm2d(out_channels)),
            ("2_activation", nn.ReLU(inplace=True)),
            ("3_dropout", nn.Dropout(dropout, inplace=True)),
            ("4_convolution", weight_norm_woG(nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1, bias=False))),
        ]))
        self.downsample = weight_norm_woG(nn.Conv2d(in_channels, out_channels, (1, 1), stride=stride, padding=0, bias=False))

    def forward(self, x):
        x = self.norm_act(x)
        return self.block(x) + self.downsample(x)


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, depth: int, dropout: float):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            DownsampleUnit(in_channels, out_channels, stride, dropout),
            *(BasicUnit(out_channels, dropout) for _ in range(depth))
        )

    def forward(self, x):
        return self.block(x)


class WideResNet_WN_woG(nn.Module):
    def __init__(self, depth: int, width_factor: int, dropout: float, in_channels: int, labels: int):
        super(WideResNet_WN_woG, self).__init__()

        self.filters = [16, 1 * 16 * width_factor, 2 * 16 * width_factor, 4 * 16 * width_factor]
        self.block_depth = (depth - 4) // (3 * 2)

        self.f = nn.Sequential(OrderedDict([
            ("0_convolution", weight_norm_woG(nn.Conv2d(in_channels, self.filters[0], (3, 3), stride=1, padding=1, bias=False))),
            ("1_block", Block(self.filters[0], self.filters[1], 1, self.block_depth, dropout)),
            ("2_block", Block(self.filters[1], self.filters[2], 2, self.block_depth, dropout)),
            ("3_block", Block(self.filters[2], self.filters[3], 2, self.block_depth, dropout)),
            ("4_normalization", nn.BatchNorm2d(self.filters[3])),
            ("5_activation", nn.ReLU(inplace=True)),
            ("6_pooling", nn.AvgPool2d(kernel_size=8)),
            ("7_flattening", nn.Flatten()),
            ("8_classification", nn.Linear(in_features=self.filters[3], out_features=labels)),
        ]))

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, x):
        return self.f(x)