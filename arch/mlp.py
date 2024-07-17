from torch.nn import Linear, Tanh, Sequential, Flatten

def get_mlp(D_in, D_hidden, D_out, depth):
    assert depth >= 1
    if depth == 1:
        blocks = [Flatten(), Linear(D_in, D_hidden), Tanh()]
    else:
        blocks = [Flatten(), Linear(D_in, D_hidden), Tanh()]
        for _ in range(depth-2):
            blocks.append(Linear(D_hidden, D_hidden))
            blocks.append(Tanh())
        blocks.append(Linear(D_hidden, D_out))

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

