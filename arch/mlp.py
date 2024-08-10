from torch.nn import Linear, Tanh, Sequential, Flatten, Module
import sys

class Scale(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 2*x
    
class Squeeze(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze()

def get_mlp(D_in, D_hidden, D_out, depth):
    use_bias = True
    assert depth >= 1
    if depth == 1:
        blocks = [Flatten(), Linear(D_in, D_hidden), Tanh()]
    else:
        blocks = [Flatten(), Linear(D_in, D_hidden, bias=use_bias), Tanh()]
        for _ in range(depth-2):
            blocks.append(Linear(D_hidden, D_hidden, bias=use_bias))
            blocks.append(Tanh())
        blocks.append(Linear(D_hidden, D_out, bias=use_bias))
    #blocks.append(Tanh())
    #blocks.append(Flatten(start_dim=0))
    blocks.append(Squeeze())
    #blocks.append(Scale())
    return Sequential(*blocks)

    return  Sequential(
        Flatten(),
        Linear(D_in, D_hidden),
        Tanh(),
        Linear(D_hidden, D_hidden),
        #Tanh(),
        #Linear(D_hidden, D_hidden),
        #Tanh(),
        #Linear(D_hidden, D_hidden),
        Tanh(),
        Linear(D_hidden, D_out),
    )

