from torch.nn import Linear, Tanh, Sequential, Flatten

def get_mlp(D_in, D_hidden, D_out):
    return  Sequential(
        Flatten(),
        Linear(D_in, D_hidden),
        Tanh(),
        Linear(D_hidden, D_hidden),
        Tanh(),
        Linear(D_hidden, D_out),
    )