from utilities import get_hessian_eigenvalues_weight_decay, get_hessian_eigenvalues
from optimizer.sam import disable_running_stats, enable_running_stats
import torch
import sys

def compute_eigenvalues(graphs, model, criterion_summed, weight_decay, loader, num_classes, device):
    model.train()
    disable_running_stats(model) #set momentum to 0, do not update running mean or var.
    #running mean requires_grad is set to False. No worry about having computing gradient.

    #an ascent step on data
    for batch_idx, (data, target) in enumerate(loader, start=1):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        out = model(data)
        loss = criterion_summed(out, target)

        (loss / data.shape[0]).backward()

    adv_loader = ...
    eigs, _ = get_hessian_eigenvalues_weight_decay(model, criterion_summed, weight_decay, adv_loader, neigs=10, num_classes=num_classes, device=device)
    #eigs, _, _, _ = get_hessian_eigenvalues(model, criterion_summed, lr, analysis_dataset, neigs=10, return_smallest = False)
    graphs.eigs.append(eigs[0].item())
    
    print("train eigs:", graphs.eigs)
    enable_running_stats(model)