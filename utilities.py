from typing import List, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
import os
import sys
from device_variable import device

# the default value for "physical batch size", which is the largest batch size that we try to put on the GPU
DEFAULT_PHYS_BS = 1000


def get_gd_directory(dataset: str, lr: float, arch_id: str, seed: int, opt: str, loss: str, beta: float = None):
    """Return the directory in which the results should be saved."""
    #results_dir = os.environ["RESULTS"]
    results_dir = "results"
    directory = f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/{opt}/"
    if opt == "gd":
        return f"{directory}/lr_{lr}"
    elif opt == "polyak":
        return f"{directory}/polyak_lr_{lr}_beta_{beta}"
    elif opt == "nesterov":
        return f"{directory}/nesterov_lr_{lr}_beta_{beta}"

    
def get_gd_directory_linear(dataset: str, lr: float, arch_id: str, seed: int, opt: str, loss: str, beta: float = None):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]
    directory = f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/gd_linear/"
    if opt == "gd":
        return f"{directory}/lr_{lr}"
    elif opt == "polyak" or opt == "nesterov":
        return f"{directory}/lr_{lr}_beta_{beta}"


def get_flow_directory(dataset: str, arch_id: str, seed: int, loss: str, tick: float):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]
    return f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/flow/tick_{tick}"


def get_modified_flow_directory(dataset: str, arch_id: str, seed: int, loss: str, gd_lr: float, tick: float):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]
    return f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/modified_flow_lr_{gd_lr}/tick_{tick}"


def get_gd_optimizer(parameters, opt: str, lr: float, momentum: float) -> Optimizer:
    if opt == "gd":
        return SGD(parameters, lr=lr)
    elif opt == "polyak":
        return SGD(parameters, lr=lr, momentum=momentum, nesterov=False)
    elif opt == "nesterov":
        return SGD(parameters, lr=lr, momentum=momentum, nesterov=True)


def save_files(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}")


def save_files_final(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}_final")


def iterate_dataset(dataset: Dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    #print(device)
    for (batch_X, batch_y) in loader:
        yield batch_X.to(device), batch_y.to(device)


def compute_losses(network: nn.Module, loss_functions: List[nn.Module], dataset: Dataset,
                   batch_size: int = DEFAULT_PHYS_BS):
    """Compute loss over a dataset."""
    L = len(loss_functions)
    losses = [0. for l in range(L)]
    with torch.no_grad():
        for (X, y) in iterate_dataset(dataset, batch_size):
            preds = network(X)
            for l, loss_fn in enumerate(loss_functions):
                losses[l] += loss_fn(preds, y) / len(dataset)
    return losses


def get_loss_and_acc(loss: str):
    """Return modules to compute the loss and accuracy.  The loss module should be "sum" reduction. """
    if loss == "mse":
        return SquaredLoss(), SquaredAccuracy()
    elif loss == "ce":
        return nn.CrossEntropyLoss(reduction='sum'), AccuracyCE()
    elif loss == "exp":
        return ExponentialLoss(), SquaredAccuracy()
    raise NotImplementedError(f"no such loss function: {loss}")

def compute_loss_linear(network, loss_fn, X, y):
    params = parameters_to_vector(network.parameters())
    params_value = parameters_to_vector(network.parameters()).detach()
    loss = 0
    predictor = network(X)
    assert predictor.shape[1] == 1
    for i in range(predictor.shape[0]):
        jacobian_i = parameters_to_vector(torch.autograd.grad(predictor[i, 0], network.parameters(), retain_graph = True)).detach()
        loss += loss_fn(predictor[i, 0].detach() + jacobian_i @ (params - params_value), y[i])
        #loss += loss_fn(predictor[i, 0], y[i])
    return loss

def get_gradient(network, loss_fn, dataset, physical_batch_size):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    gradient = torch.zeros(p, dtype=torch.float, device=device)
    loss_derivative = []
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        predictor = network(X)
        loss = loss_fn(predictor, y) / n
        grad = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        gradient += parameters_to_vector(grad)
        loss_derivative.append((predictor - y).detach().cpu())
    loss_derivative = torch.cat(loss_derivative, axis = 0)#.squeeze()
    return gradient.detach().cpu(), loss_derivative

def compute_hessian_grad_product(network: nn.Module, loss_fn: nn.Module,
                dataset: Dataset, vector: Tensor, physical_batch_size: int = DEFAULT_PHYS_BS, sample_interval: int=100):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    hvp = torch.zeros((n // sample_interval, p), dtype=torch.float, device=device)
    vector = vector.to(device)
    sample_id = 0
    for (X, y) in iterate_dataset(dataset, 1):
        if sample_id % sample_interval == 0:
            loss = network(X)[0,0]
            grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
            dot = parameters_to_vector(grads).mul(vector).sum()
            grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
            hvp[sample_id // sample_interval, :] = parameters_to_vector(grads)
        sample_id += 1
    return hvp.cpu()

def compute_hvp(network: nn.Module, loss_fn: nn.Module,
                dataset: Dataset, vector: Tensor, physical_batch_size: int = DEFAULT_PHYS_BS):
    """Compute a Hessian-vector product."""
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    hvp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        #loss = loss_fn(network(X), y) / n
        loss = loss_fn(network(X.to(device)), torch.nn.functional.one_hot(y.to(device), num_classes=10).float()) / n
        #print(loss)
        #for param in network.parameters():
        #    print(param.data)
        """
        param_list = []
        for param in network.parameters():
            param_list.append(param)
            assert param.requires_grad
        beta = torch.square(param_list[0]) - torch.square(param_list[1])
        loss = 0.25*torch.mean((X@beta-y)**2)
        """
        #loss = loss_fn(network(X), y) / n
        #loss = 0.25 * torch.mean((network(X).squeeze()-y.squeeze())**2)
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    return hvp

def compute_hvp_smallest(network: nn.Module, loss_fn: nn.Module, alpha:float,
                dataset: Dataset, vector: Tensor, physical_batch_size: int = DEFAULT_PHYS_BS):
    """Compute a Hessian-vector product."""
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    hvp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    hvp = alpha * vector - hvp
    return hvp

def lanczos(matrix_vector, dim: int, neigs: int, which="LM"):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).to(device)
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    l_evals, l_evecs = eigsh(operator, neigs, which=which)
    #s_evals, s_evecs= eigsh(operator, neigs, which='SM')
    return torch.from_numpy(np.ascontiguousarray(l_evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(l_evecs, -1)).copy()).float()
           #torch.from_numpy(np.ascontiguousarray(s_evals[::-1]).copy()).float(), \
           #torch.from_numpy(np.ascontiguousarray(np.flip(s_evecs, -1)).copy()).float()

def compute_hvp_weight_decay(network: nn.Module, loss_fn: nn.Module, weight_decay: float,
                data_loader: torch.utils.data.DataLoader, vector: Tensor, num_classes: int, device: torch.device, use_hf_model: bool=False):
    """Compute a Hessian-vector product."""
    p = len(parameters_to_vector(network.parameters()))
    n = len(data_loader.dataset)
    hvp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)
    #for (X, y) in iterate_dataset(dataset, physical_batch_size):
    if not use_hf_model:
        for batch_idx, (data, target) in enumerate(data_loader):
            #print("hvp:", hvp)
            #print(loss_fn(network(data.to(device)), torch.nn.functional.one_hot(target.to(device), num_classes=num_classes).float()))
            #print(network(data.to(device))[0], torch.nn.functional.one_hot(target.to(device), num_classes=num_classes)[0])
            #loss = loss_fn(network(data.to(device)), torch.nn.functional.one_hot(target.to(device), num_classes=num_classes).float()) / n
            loss = loss_fn(network(data.to(device)), target.to(device)) / n
            #print("loss at analysis:", loss)
            grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
            dot = parameters_to_vector(grads).mul(vector).sum()
            #print("dot:", dot)
            grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
            #print("grads:", parameters_to_vector(grads) )
            hvp +=  parameters_to_vector(grads) 
    else:
        assert len(data_loader) == 1
        for batch_idx, input in enumerate(data_loader):
            output = network(**input)
            loss = output.loss
            grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
            dot = parameters_to_vector(grads).mul(vector).sum()
            #print("dot:", dot)
            grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
            #print("grads:", parameters_to_vector(grads) )
            hvp +=  parameters_to_vector(grads) 

    hvp += weight_decay * vector
    #print(hvp)
    return hvp

def get_hessian_eigenvalues_weight_decay(network: nn.Module, loss_fn: nn.Module, weight_decay: float, loader: torch.utils.data.DataLoader,
                            neigs=5, num_classes=10, device=torch.device('cpu'), which_eigs = "LM", use_hf_model=False):
    #vector_test = torch.ones(200)
    #print(compute_hvp(network, loss_fn, dataset, vector_test, physical_batch_size=physical_batch_size))
    #sys.exit()
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp_weight_decay(network, loss_fn, weight_decay, loader,
                                          delta, num_classes, device, use_hf_model).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    #print("lanczos starts")
    l_evals, l_evecs = lanczos(hvp_delta, nparams, neigs=neigs, which = which_eigs)
    #print("lanczos ends")
    #print(l_evals)
    return l_evals, l_evecs

def compute_hvp_weight_decay_hf(network: nn.Module, loss_fn: nn.Module, weight_decay: float,
                data_loader: torch.utils.data.DataLoader, vector: Tensor, num_classes: int, device: torch.device):
    """Compute a Hessian-vector product."""
    p = len(parameters_to_vector(network.parameters()))
    n = len(data_loader.dataset)
    hvp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)
    for batch_idx, inputs in enumerate(data_loader, start=1):
        outputs = network(**inputs)
        loss = outputs['loss']
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp +=  parameters_to_vector(grads) 
    hvp += weight_decay * vector
    return hvp

def get_hessian_eigenvalues_weight_decay_hf(network: nn.Module, loss_fn: nn.Module, weight_decay: float, loader: torch.utils.data.DataLoader,
                            neigs=5, num_classes=10, device=torch.device('cpu')):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp_weight_decay_hf(network, loss_fn, weight_decay, loader,
                                          delta, num_classes, device).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    l_evals, l_evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return l_evals, l_evecs

def get_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module, lr: float, dataset: Dataset,
                            neigs=6, physical_batch_size=1000, return_smallest = True):
    #vector_test = torch.ones(200)
    #print(compute_hvp(network, loss_fn, dataset, vector_test, physical_batch_size=physical_batch_size))
    #sys.exit()
    """ Compute the leading Hessian eigenvalues. """
    alpha = 4 / lr
    s_evals, s_evecs = 0, 0
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, dataset,
                                          delta, physical_batch_size=physical_batch_size).detach().cpu()
    if return_smallest == True:
        hvp_delta_small = lambda delta: compute_hvp_smallest(network, loss_fn, alpha, dataset,
                                            delta, physical_batch_size=physical_batch_size).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    l_evals, l_evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    if return_smallest == True:
        s_evals, s_evecs = lanczos(hvp_delta_small, nparams, neigs=neigs)
    return l_evals, l_evecs, alpha - s_evals, s_evecs

def get_hessian_eigenvalues_smallest(network: nn.Module, loss_fn: nn.Module, lr: float, dataset: Dataset,
                            neigs=6, physical_batch_size=1000):
    """ Compute the leading Hessian eigenvalues. """
    alpha = 4 / lr
    hvp_delta_small = lambda delta: compute_hvp_smallest(network, loss_fn, alpha, dataset,
                                          delta, physical_batch_size=physical_batch_size).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    s_evals, s_evecs = lanczos(hvp_delta_small, nparams, neigs=neigs)
    return alpha - s_evals, s_evecs

def get_gauss_newton_eigenvalues(loss_name: str, network: nn.Module, dataset: Dataset, neigs=6, num_class=10, device=torch.device('cpu')):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_gnvp(loss_name, network, dataset, delta, num_class, device).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    l_evals, l_evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return l_evals, l_evecs

def get_gauss_newton_w_eigenvalues(network: nn.Module, dataset: Dataset, neigs=6, num_class=10, w_shape=None):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_gnvp_w(network, dataset, delta, num_class, w_shape).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    l_evals, l_evecs = lanczos(hvp_delta, w_shape, neigs=neigs)
    return l_evals, l_evecs

def get_gauss_newton_w_class_eigenvalues(network: nn.Module, dataset: Dataset, neigs=6, num_class=10, w_shape=None):
    """ Compute the leading Hessian eigenvalues. """
    l_evals = torch.zeros(num_class, neigs)
    for class_index in range(num_class):
        hvp_delta = lambda delta: compute_gnvp_w_i(network, dataset, delta, num_class, w_shape, class_index).detach().cpu()
        nparams = len(parameters_to_vector((network.parameters())))
        l_evals[class_index, :], l_evecs = lanczos(hvp_delta, w_shape, neigs=neigs)
    return l_evals, l_evecs

def get_gauss_newton_u_eigenvalues(network: nn.Module, dataset: Dataset, neigs=6, num_class=10, w_shape=None):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_gnvp_u(network, dataset, delta, num_class, w_shape).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    l_evals, l_evecs = lanczos(hvp_delta, nparams-w_shape, neigs=neigs)
    return l_evals, l_evecs

def get_delta_c_eigenvalues(network: nn.Module, dataset: Dataset, neigs=6, num_class=10):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_delta_c_vp(network, dataset, delta, num_class).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    l_evals, l_evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return l_evals, l_evecs

def get_delta_c_c_eigenvalues(network: nn.Module, dataset: Dataset, neigs=6):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_delta_c_c_vp(network, dataset, delta).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    l_evals, l_evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return l_evals, l_evecs

def get_fld_eigenvalues(network: nn.Module, dataset: Dataset, neigs=6, num_class=10):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_fld_vp(network, dataset, delta, num_class).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    l_evals, l_evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return l_evals, l_evecs


def compute_hvp_eig(network: nn.Module, loss, vector: Tensor):
    #p = len(params)
    #hvp = torch.zeros(p, dtype=torch.float, device=device)
    grads = torch.autograd.grad(loss, network.parameters(), create_graph=True)
    dot = parameters_to_vector(grads).mul(vector).sum()
    g_list = torch.autograd.grad(dot, network.parameters(), retain_graph=True)
    grads = [g.contiguous() for g in g_list]
    hvp = parameters_to_vector(grads)  
    return hvp

def get_eig_grad(network: nn.Module, loss_fn: nn.Module, dataset: Dataset, 
                 eig_vec, physical_batch_size=1000):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    print("param:", p)
    eig_vec = eig_vec.to(device)
    eig_grad = torch.zeros(p, dtype=torch.float, device=device)
    grad_sum = torch.zeros(p, dtype=torch.float, device=device)
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(loss, network.parameters(), create_graph=True)
        grads = parameters_to_vector(grads)
        grad_sum += grads
        for i in range(p):
            eig_grad[i] += compute_hvp_eig(network, grads[i], eig_vec) @ eig_vec
    print("cosine:", grad_sum @ eig_grad)
    return eig_grad

def second_derivative_of_loss(loss_name, x=0, y=0):
    if loss_name == "MSELoss":
        return 2
    else: raise NotImplementedError

def compute_gnvp(loss_name: str, network: nn.Module, dataset: Dataset, vector: Tensor, num_class: int, device):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    #gnvp = torch.zeros(p, dtype=torch.float, device=device)
    gnvp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)
    pred_grad = torch.zeros((p, num_class), dtype=torch.float, device=device)
    for (X, y) in iterate_dataset(dataset, n):
        predictor = network(X)
        for sample_idx in range(X.shape[0]):
            for i in range(predictor.shape[1]):
                grads_i = parameters_to_vector(torch.autograd.grad(predictor[sample_idx, i], network.parameters(), retain_graph = True))
                pred_grad[:,i] = grads_i
            gnvp += (second_derivative_of_loss(loss_name, X[sample_idx], y[sample_idx]) * pred_grad) @ (pred_grad.T @ vector) / n
    return gnvp

def compute_jacobian_norm(network: nn.Module, dataset: Dataset, num_class: int):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    #gnvp = torch.zeros(p, dtype=torch.float, device=device)
    jacobian_norm = torch.zeros(n * num_class, dtype=torch.float, device='cpu')
    sample_id = 0
    for (X, _) in iterate_dataset(dataset, 1):
        predictor = network(X)
        for i in range(predictor.shape[1]):
            grads_i = parameters_to_vector(torch.autograd.grad(predictor[0, i], network.parameters(), retain_graph = True))
            jacobian_norm[sample_id * num_class + i] = torch.norm(grads_i).cpu() / n
        sample_id += 1
    return jacobian_norm

def compute_jacobian(network: nn.Module, dataset: Dataset, num_class: int, sample_interval: int):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    #gnvp = torch.zeros(p, dtype=torch.float, device=device)
    jacobian = torch.zeros(p, (n // sample_interval) * num_class, dtype=torch.float, device='cpu')
    sample_id = 0
    for (X, _) in iterate_dataset(dataset, 1):
        if sample_id % sample_interval == 0:
            predictor = network(X)
            for i in range(predictor.shape[1]):
                grads_i = parameters_to_vector(torch.autograd.grad(predictor[0, i], network.parameters(), retain_graph = True))
                jacobian[:, (sample_id // sample_interval) * num_class + i] = grads_i.cpu()
        sample_id += 1
    return jacobian

def compute_gnvp_w(network: nn.Module, dataset: Dataset, vector: Tensor, num_class: int, w_shape: int):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    #gnvp = torch.zeros(p, dtype=torch.float, device=device)
    gnvp = torch.zeros(w_shape, dtype=torch.float, device=device)
    vector = vector.to(device)
    pred_grad = torch.zeros((w_shape, num_class), dtype=torch.float, device=device)
    for (X, _) in iterate_dataset(dataset, 1):
        predictor = network(X)
        for i in range(predictor.shape[1]):
            grads_i = parameters_to_vector(torch.autograd.grad(predictor[0, i], network.parameters(), retain_graph = True))[:w_shape]
            pred_grad[:,i] = grads_i
        gnvp += pred_grad @ (pred_grad.T @ vector) / n
    return gnvp

def get_gauss_newton_matrix_u(network: nn.Module, dataset: Dataset, num_class: int, w_shape: int):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    jacobian = torch.zeros((p-w_shape, n * num_class), dtype=torch.float, device=device)
    #gnvp = torch.zeros(w_shape, dtype=torch.float, device=device)
    #vector = vector.to(device)
    #pred_grad = torch.zeros((w_shape, num_class), dtype=torch.float, device=device)
    sample_id = 0
    for (X, _) in iterate_dataset(dataset, 1):
        predictor = network(X)
        assert predictor.shape[1] == num_class
        for i in range(predictor.shape[1]):
            grads_i = parameters_to_vector(torch.autograd.grad(predictor[0, i], network.parameters(), retain_graph = True))[w_shape:]
            #pred_grad[:,i] = grads_i
            if sample_id == 0:
                print(torch.where(torch.abs(grads_i) <= 1e-6)[0].shape)
            jacobian[:,i*n+sample_id] = grads_i
        #gnvp += pred_grad @ (pred_grad.T @ vector) / n
        sample_id += 1
    gnvp = jacobian @ jacobian.T / n
    return gnvp

def compute_gnvp_multiple(network: nn.Module, dataset: Dataset, vectors: Tensor, num_class: int):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    #gnvp = torch.zeros(p, dtype=torch.float, device=device)
    gnvp = torch.zeros((n*num_class, vectors.shape[1]), dtype=torch.float, device=device)
    vectors = vectors.to(device)
    sample_id = 0
    for (X, _) in iterate_dataset(dataset, 1):
        predictor = network(X)
        for i in range(predictor.shape[1]):
            grads_i = parameters_to_vector(torch.autograd.grad(predictor[0, i], network.parameters(), retain_graph = True))
            gnvp[i*n+sample_id, :] = vectors.T @ grads_i
        sample_id += 1
    return gnvp   

def compute_gnvp_w_multiple(network: nn.Module, dataset: Dataset, vectors: Tensor, num_class: int, w_shape: int):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    #gnvp = torch.zeros(p, dtype=torch.float, device=device)
    gnvp = torch.zeros((n*num_class, vectors.shape[1]), dtype=torch.float, device=device)
    vectors = vectors.to(device)
    sample_id = 0
    for (X, _) in iterate_dataset(dataset, 1):
        predictor = network(X)
        for i in range(predictor.shape[1]):
            grads_i = parameters_to_vector(torch.autograd.grad(predictor[0, i], network.parameters(), retain_graph = True))[:w_shape]
            gnvp[i*n+sample_id, :] = vectors.T @ grads_i
        sample_id += 1
    return gnvp   

def compute_gnvp_w_i(network: nn.Module, dataset: Dataset, vector: Tensor, num_class: int, w_shape: int, class_index_true: int):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    #gnvp = torch.zeros(p, dtype=torch.float, device=device)
    gnvp = torch.zeros(w_shape, dtype=torch.float, device=device)
    vector = vector.to(device)
    pred_grad = torch.zeros((w_shape, num_class), dtype=torch.float, device=device)
    for (X, y) in iterate_dataset(dataset, 1):
        class_index = torch.where(y==1)[1][0].item()
        if (class_index_true != class_index):
            continue
        predictor = network(X)
        for i in range(predictor.shape[1]):
            grads_i = parameters_to_vector(torch.autograd.grad(predictor[0, i], network.parameters(), retain_graph = True))[:w_shape]
            pred_grad[:,i] = grads_i
        gnvp += pred_grad @ (pred_grad.T @ vector) / n
    return gnvp

def compute_gnvp_u_multiple(network: nn.Module, dataset: Dataset, vectors: Tensor, num_class: int, w_shape: int):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    #gnvp = torch.zeros(p, dtype=torch.float, device=device)
    gnvp = torch.zeros((n*num_class, vectors.shape[1]), dtype=torch.float, device=device)
    vectors = vectors.to(device)
    sample_id = 0
    for (X, _) in iterate_dataset(dataset, 1):
        predictor = network(X)
        for i in range(predictor.shape[1]):
            grads_i = parameters_to_vector(torch.autograd.grad(predictor[0, i], network.parameters(), retain_graph = True))[w_shape:]
            gnvp[i*n+sample_id, :] = vectors.T @ grads_i
        sample_id += 1
    return gnvp 

def compute_gnvp_u(network: nn.Module, dataset: Dataset, vector: Tensor, num_class: int, w_shape: int):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    #gnvp = torch.zeros(p, dtype=torch.float, device=device)
    gnvp = torch.zeros(p-w_shape, dtype=torch.float, device=device)
    vector = vector.to(device)
    pred_grad = torch.zeros((p-w_shape, num_class), dtype=torch.float, device=device)
    for (X, _) in iterate_dataset(dataset, 1):
        predictor = network(X)
        for i in range(predictor.shape[1]):
            grads_i = parameters_to_vector(torch.autograd.grad(predictor[0, i], network.parameters(), retain_graph = True))[w_shape:]
            pred_grad[:,i] = grads_i
        gnvp += pred_grad @ (pred_grad.T @ vector) / n
    return gnvp

def compute_delta_c_vp(network: nn.Module, dataset: Dataset, vector: Tensor, num_class: int):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    gnvp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)
    pred_grad = torch.zeros((p, num_class), dtype=torch.float, device=device)
    counter = [0 for _ in range(num_class)]
    for (X, y) in iterate_dataset(dataset, 1):
        
        class_num = y.shape[1]
        assert num_class == class_num
        class_index = torch.where(y==1)[1][0].item()
        counter[class_index] += 1
        predictor = network(X)
        for i in range(predictor.shape[1]):
            if i != class_index:
                grads_i = parameters_to_vector(torch.autograd.grad(predictor[0, i], network.parameters(), retain_graph = True))
                pred_grad[:,class_index] += grads_i / (class_num - 1)
    
    for class_index in range(num_class):
        if counter[class_index] > 0:
            pred_grad[:,class_index] = pred_grad[:,class_index] / counter[class_index]
        else:
            print("Some classes not visited.")
    #print(pred_grad.shape)
    #print(vector.shape)
    gnvp = pred_grad @ (pred_grad.T @ vector) * (class_num - 1)
    #gnvp = pred_grad.T @ (pred_grad @ vector) * (class_num - 1)
    return gnvp

def compute_fld_vp(network: nn.Module, dataset: Dataset, vector: Tensor, num_class: int):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    gnvp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)
    pred_grad = torch.zeros((p, num_class), dtype=torch.float, device=device)
    grad_avg = 0
    counter = [0 for _ in range(num_class)]
    for (X, y) in iterate_dataset(dataset, 1):
        
        class_num = y.shape[1]
        assert num_class == class_num
        class_index = torch.where(y==1)[1][0].item()
        counter[class_index] += 1
        predictor = network(X)
        for i in range(predictor.shape[1]):
            #if i != class_index:
            grads_i = parameters_to_vector(torch.autograd.grad(predictor[0, i], network.parameters(), retain_graph = True))
            pred_grad[:,class_index] += grads_i / class_num
            grad_avg += grads_i / class_num / n
    assert np.sum(counter) == n
    for class_index in range(num_class):
        if counter[class_index] > 0:
            pred_grad[:,class_index] = pred_grad[:,class_index] / counter[class_index]
        else:
            print("Some classes not visited.")
    #print(pred_grad.shape)
    #print(vector.shape)
    grad_avg = grad_avg.unsqueeze(-1)
    gnvp = (pred_grad - grad_avg) @ ((pred_grad - grad_avg).T @ vector) * class_num
    #gnvp = pred_grad.T @ (pred_grad @ vector) * (class_num - 1)
    return gnvp

def compute_delta_c_c_vp(network: nn.Module, dataset: Dataset, vector: Tensor):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    #delta_c_c_vp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)
    delta_c_c_grad = torch.zeros((p, 10), dtype=torch.float, device=device)
    for (X, y) in iterate_dataset(dataset, 1):
        class_index = torch.where(y==1)[1][0].item()
        predictor = network(X)
        for i in range(predictor.shape[1]):
            if i == class_index:
                grads_i = parameters_to_vector(torch.autograd.grad(predictor[0, i], network.parameters(), retain_graph = True))
                delta_c_c_grad[:, i] += grads_i / n
    delta_c_c_vp = delta_c_c_grad @ (delta_c_c_grad.T @ vector)
    return delta_c_c_vp

def compute_delta_c_cp(network: nn.Module, dataset: Dataset):
    class_list = [i for i in range(10)]

    means = []
    counters = []
    for c in class_list:
        means.append([])
        counters.append([])
        for cp in class_list:
            means[-1].append(None)
            counters[-1].append(0)
    
    for (X, y) in iterate_dataset(dataset, 1):
        predictor = network(X)
        for idx_c, c in enumerate(class_list):
            idxs = (y == c).nonzero()
            fc = predictor[idxs.squeeze(-1)]

def compute_gradient(network: nn.Module, loss_fn: nn.Module,
                     dataset: Dataset, physical_batch_size: int = DEFAULT_PHYS_BS):
    """ Compute the gradient of the loss function at the current network parameters. """
    p = len(parameters_to_vector(network.parameters()))
    average_gradient = torch.zeros(p, device=device)
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        batch_loss = loss_fn(network(X), y) / len(dataset)
        batch_gradient = parameters_to_vector(torch.autograd.grad(batch_loss, inputs=network.parameters()))
        average_gradient += batch_gradient
    return average_gradient


class AtParams(object):
    """ Within a with block, install a new set of parameters into a network.

    Usage:

        # suppose the network has parameter vector old_params
        with AtParams(network, new_params):
            # now network has parameter vector new_params
            do_stuff()
        # now the network once again has parameter vector new_params
    """

    def __init__(self, network: nn.Module, new_params: Tensor):
        self.network = network
        self.new_params = new_params

    def __enter__(self):
        self.stash = parameters_to_vector(self.network.parameters())
        vector_to_parameters(self.new_params, self.network.parameters())

    def __exit__(self, type, value, traceback):
        vector_to_parameters(self.stash, self.network.parameters())


def compute_gradient_at_theta(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                              theta: torch.Tensor, batch_size=DEFAULT_PHYS_BS):
    """ Compute the gradient of the loss function at arbitrary network parameters "theta".  """
    with AtParams(network, theta):
        return compute_gradient(network, loss_fn, dataset, physical_batch_size=batch_size)


class SquaredLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        #return 0.5 * ((input - target) ** 2).sum()
        return 0.25 * ((input - target) ** 2).sum()

class ExponentialLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        target = 2*(target - 0.5)
        return torch.mean(torch.exp(-input * target))

class SquaredAccuracy(nn.Module):
    def __init__(self):
        super(SquaredAccuracy, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target.argmax(1)).float().sum()


class AccuracyCE(nn.Module):
    def __init__(self):
        super(AccuracyCE, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target).float().sum()


class VoidLoss(nn.Module):
    def forward(self, X, Y):
        return 0

def _check_param_device(param: torch.Tensor, old_param_device) -> int:
    r"""Check if the parameters are located on the same device.

    Currently, the conversion between model parameters and single vector form is not supported
    for multiple allocations, e.g. parameters in different GPUs/PrivateUse1s, or mixture of CPU/GPU/PrivateUse1.

    Args:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """
    # Meet the first parameter
    support_device_types = ["cuda", torch._C._get_privateuse1_backend_name()]
    if old_param_device is None:
        old_param_device = param.get_device() if param.device.type in support_device_types else -1
    else:
        warn = False
        if param.device.type in support_device_types:  # Check if in same GPU/PrivateUse1
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device

def get_exp_avg(optimizer):
    state_dict = optimizer.state_dict()['state']
    exp_avgs, exp_avg_sqs = [], []
    # Flag for the device where the parameter is located
    param_device = None
    vec = []
    for name in state_dict:
        exp_avg = state_dict[name]['exp_avg']
        exp_avg_sq = state_dict[name]['exp_avg_sq']
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(exp_avg, param_device)

        exp_avgs.append(exp_avg.view(-1))
        exp_avg_sqs.append(exp_avg_sq.view(-1))
    return torch.cat(exp_avgs), torch.cat(exp_avg_sqs)



def state_dict_to_vector(state_dict) -> torch.Tensor:
    r"""Flatten an iterable of parameters into a single vector.

    Args:
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for name in state_dict:
        param = state_dict[name]
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.view(-1))
    return torch.cat(vec)

def vector_to_state_dict(vec, state_dict) -> None:
    r"""Copy slices of a vector into an iterable of parameters.

    Args:
        vec (Tensor): a single vector representing the parameters of a model.
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f'expected torch.Tensor, but got: {torch.typename(vec)}')
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for name in state_dict:
        param = state_dict[name]
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param

    return state_dict

def map_update(map1, map2, reduction="sum"):
    if map2 is None:
        return
    for key in map2:
        if reduction == "sum":
            if key in map1:
                map1[key] += map2[key]
            else:
                map1[key] = map2[key]
        elif reduction == "append":
            if key in map1:
                map1[key].append(map2[key])
            else:
                map1[key] = [map2[key]]
        else:
            raise NotImplementedError
        
def graph_update(graph, map,  normalizer):
    for key in map:
        if key in ["grad_norm", "grad_l1_norm", "ascent_grad_norm", "ascent_grad_l1_norm",\
                    "dominant_alignment", "batch_loss", "hessian_gn_align", "hessian_eig", "gn_eig", \
                    "ratio_A", "ratio_B"]:
            #print(key, map[key])
            getattr(graph, key).append(map[key])
        else:
            if key == 'ascent_step_cos':
                print(map[key])
            getattr(graph, key).append(map[key] / normalizer)

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def grads_to_vector(parameters: Iterable[torch.Tensor]) -> None:
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.grad.view(-1))
    return torch.cat(vec)

def vector_to_group_grads(vec, param_groups):
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f'expected torch.Tensor, but got: {torch.typename(vec)}')
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for group in param_groups:
        for param in group["params"]:
            param_device = _check_param_device(param, param_device)
            num_param = param.numel()
            # Slice the vector, reshape it, and replace the old data of the parameter
            param.grad = vec[pointer:pointer + num_param].view_as(param).data

            # Increment the pointer
            pointer += num_param

def vector_to_grads(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f'expected torch.Tensor, but got: {torch.typename(vec)}')
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.grad = vec[pointer:pointer + num_param].to(param).view_as(param).data
        #print(param.requires_grad, torch.norm(param.grad).item())

        # Increment the pointer
        pointer += num_param

def vector_to_grads_sq(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f'expected torch.Tensor, but got: {torch.typename(vec)}')
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.grad_sq = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param

def dict_to_(sample_dict, device):
    for key in sample_dict:
        value = sample_dict[key]
        if isinstance(value, torch.Tensor):
            sample_dict[key] = value.to(device)

def copy_graph(dest, src):
    for key in src.__dict__:
        setattr(dest, key, getattr(src, key))

def density_generate(eigenvalues,
                     weights,
                     num_bins=10000,
                     sigma_squared=1e-5,
                     overhead=0.01):

    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids


def gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x)**2 /
                  (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)

def get_esd_plot(ax, eigenvalues, weights, title, ylabel=False):
    #import matplotlib.pyplot as plt
    density, grids = density_generate(eigenvalues, weights)
    ax.semilogy(grids, density + 1.0e-7)
    #ax.set_ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
    ax.set_xlabel('Eigenvalues', fontsize=10, labelpad=6)
    if ylabel == 0:
        ax.set_ylabel('Density (Log Scale)', fontsize=8, labelpad=6)
    #ax.set_xticks(fontsize=12)
    #ax.set_yticks(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=8)
    #ax.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
    ax.set_xlim(np.min(eigenvalues) - 1, np.max(eigenvalues) + 1)
    #plt.tight_layout()
    ax.set_title("Epoch: " + str(title-1))

def get_cls_head_name_from_model(model_name):
    if model_name == "gpt2":
        return "score.weight"
    else:
        raise NotImplementedError

def project_to_orth_space(vecs, eigvecs):
    # eigvecs: d * N; vecs: d * M
    dom_vecs = eigvecs @ (eigvecs.T @ vecs)
    return dom_vecs, torch.norm(dom_vecs) / torch.norm(vecs)

def cosine_similarity_batch(u, v, ret_abs=False):
    #u,v shape: d * r
    u_norm, v_norm = torch.norm(u, dim=0), torch.norm(v, dim=0)
    inner_product = torch.sum(u*v, dim=0)
    cosine =  inner_product / (u_norm * v_norm)
    if ret_abs:
        return torch.abs(cosine).mean()
    else:
        return cosine.mean()
    
def tensor_topk(x, k):
    val, ind = torch.topk(torch.abs(x), k, sorted=False)
    masked = x.clone()
    masked[ind] = 0
    return x - masked

def tensor_randk(x, k):
    n = x.numel()
    device = x.device
    
    if k >= n:
        return x  # No compression needed
    
    # Step 1: Randomly select k indices
    indices = torch.randperm(n, device=device)[:k]
    
    # Step 2: Create sparse output (initialize as zeros)
    x_hat = torch.zeros_like(x)
    
    # Step 3: Scale selected entries to ensure unbiasedness
    x_hat[indices] = x[indices] * (n / k)
    
    return x_hat

import sys
def deepgso(ob):
    size = sys.getsizeof(ob)
    if isinstance(ob, (list,tuple,set)):
        for element in ob:
            size+=deepgso(element)
    if isinstance(ob, dict):
        for k,v in ob.items():
            size+=deepgso(k)
            size+=deepgso(v)
    return size

def get_model_size(model):
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    return size_model

def get_gpu_memory():
    """
    import subprocess as sp
    import os
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(memory_free_values)
    return memory_free_values
    """
    import nvidia_smi

    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    print("Total memory:", info.total)
    print("Free memory:", info.free)
    print("Used memory:", info.used)

    nvidia_smi.nvmlShutdown()

def safe_save_model_for_hf_trainer(trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        print("saving hf repo")
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def save_lora_module_for_hf_trainer(model, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = model.state_dict()
    cpu_state_dict = {key: value.cpu() for key, value in state_dict.items() if value.requires_grad}
    torch.save(cpu_state_dict, output_dir+"lora_module.pth")

def get_sgd_optimizer_momentum(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            print(p, torch.norm(state["momentum_buffer"]))