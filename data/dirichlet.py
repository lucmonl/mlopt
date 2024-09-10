import numpy as np
from scipy import stats

def partition_dirichlet(Y, n_clients, alpha, seed):
    clients = []
    ex_per_class = np.unique(Y, return_counts=True)[1]
    n_classes = len(ex_per_class)
    print(f"Found {n_classes} classes")
    rv_tr = stats.dirichlet.rvs(np.repeat(alpha, n_classes), size=n_clients, random_state=seed) 
    rv_tr = rv_tr / rv_tr.sum(axis=0)
    rv_tr = (rv_tr*ex_per_class).round().astype(int)
    class_to_idx = {i: np.where(Y == i)[0] for i in range(n_classes)}
    curr_start = np.zeros(n_classes).astype(int)
    for client_classes in rv_tr:
        curr_end = curr_start + client_classes
        client_idx = np.concatenate([class_to_idx[c][curr_start[c]:curr_end[c]] for c in range(n_classes)])
        curr_start = curr_end
        clients.append(client_idx)
        # will be empty subset if all examples have been exhausted
    return clients