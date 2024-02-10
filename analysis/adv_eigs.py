from utilities import get_hessian_eigenvalues_weight_decay, get_hessian_eigenvalues
from optimizer.sam import disable_running_stats, enable_running_stats
import torch
import sys
from torch.utils.data.dataset import TensorDataset

def compute_adv_eigenvalues(graphs, model, criterion_summed, adv_eta, weight_decay, loader, num_classes, device):
    model.train()
    disable_running_stats(model) #set momentum to 0, do not update running mean or var.
    #running mean requires_grad is set to False. No worry about having computing gradient.

    #an ascent step on data
    adv_X, adv_y = [], []
    for batch_idx, (data, target) in enumerate(loader, start=1):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        out = model(data)
        loss = criterion_summed(out, target)

        (loss / data.shape[0]).backward()

        adv_X.append(data + adv_eta * data.grad)
        adv_y.append(target)

    adv_dataset = TensorDataset(torch.cat(adv_X, dim=0), torch.cat(adv_y, dim=0))
    adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=loader.batch_size, shuffle=False)

    eigs, _ = get_hessian_eigenvalues_weight_decay(model, criterion_summed, weight_decay, adv_loader, neigs=10, num_classes=num_classes, device=device)
    #eigs, _, _, _ = get_hessian_eigenvalues(model, criterion_summed, lr, analysis_dataset, neigs=10, return_smallest = False)
    if adv_eta in graphs.adv_eigs:
        graphs.adv_eigs[adv_eta].append(eigs[0].item())
    else:
        graphs.adv_eigs[adv_eta] = [eigs[0].item()]
    
    print("train adversarial eigs:", graphs.adv_eigs)
    enable_running_stats(model)