from torch.nn import Linear, Tanh, Sequential, Flatten, Module

class Scale(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 2*x

def get_mlp(D_in, D_hidden, D_out, depth):
    assert depth >= 1
    if depth == 1:
        blocks = [Flatten(), Linear(D_in, D_hidden), Tanh()]
    else:
        blocks = [Flatten(), Linear(D_in, D_hidden, bias=False), Tanh()]
        for _ in range(depth-2):
            blocks.append(Linear(D_hidden, D_hidden, bias=False))
            blocks.append(Tanh())
        blocks.append(Linear(D_hidden, D_out, bias=False))
    #blocks.append(Tanh())
    blocks.append(Flatten(start_dim=0))
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

