from utilities import get_hessian_eigenvalues_weight_decay, get_hessian_eigenvalues, get_hessian_eigenvalues_weight_decay_hf
from optimizer.sam import disable_running_stats, enable_running_stats
import sys
import torch

def compute_eigenvalues(graphs, model, criterion_summed, weight_decay, loader, loader_test, num_classes, device):
    model.train()
    disable_running_stats(model) #set momentum to 0, do not update running mean or var.
    #running mean requires_grad is set to False. No worry about having computing gradient.
    eigs, _ = get_hessian_eigenvalues_weight_decay(model, criterion_summed, weight_decay, loader, neigs=10, num_classes=num_classes, device=device)
    #eigs, _, _, _ = get_hessian_eigenvalues(model, criterion_summed, lr, analysis_dataset, neigs=10, return_smallest = False)
    graphs.eigs.append(eigs[0].item())
    
    print("train eigs:", graphs.eigs)
    

    eigs_test, _ = get_hessian_eigenvalues_weight_decay(model, criterion_summed, weight_decay, loader_test, neigs=10, num_classes=num_classes, device=device)
    #eigs, _, _, _ = get_hessian_eigenvalues(model, criterion_summed, lr, analysis_dataset, neigs=10, return_smallest = False)
    graphs.eigs_test.append(eigs_test[0].item())
    print("test eigs:", graphs.eigs_test)
    enable_running_stats(model)

def compute_eigenvalues_hf(graphs, model, criterion_summed, weight_decay, loader, loader_test, num_classes, device): 
    model.train()
    disable_running_stats(model) #set momentum to 0, do not update running mean or var.
    #running mean requires_grad is set to False. No worry about having computing gradient.
    eigs, _ = get_hessian_eigenvalues_weight_decay_hf(model, criterion_summed, weight_decay, loader, neigs=10, num_classes=num_classes, device=device)
    #eigs, _, _, _ = get_hessian_eigenvalues(model, criterion_summed, lr, analysis_dataset, neigs=10, return_smallest = False)
    graphs.eigs.append(eigs[0].item())
    
    print("train eigs:", graphs.eigs)
    

    eigs_test, _ = get_hessian_eigenvalues_weight_decay_hf(model, criterion_summed, weight_decay, loader_test, neigs=10, num_classes=num_classes, device=device)
    #eigs, _, _, _ = get_hessian_eigenvalues(model, criterion_summed, lr, analysis_dataset, neigs=10, return_smallest = False)
    graphs.eigs_test.append(eigs_test[0].item())
    print("test eigs:", graphs.eigs_test)
    enable_running_stats(model)