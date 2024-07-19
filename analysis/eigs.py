from utilities import get_hessian_eigenvalues_weight_decay, get_hessian_eigenvalues, get_hessian_eigenvalues_weight_decay_hf
from optimizer.sam import disable_running_stats, enable_running_stats
import sys
import torch
from torch.nn.utils import parameters_to_vector

def project_and_step(model, eigvecs, data_loader, device, loss_fn):
    model.zero_grad()
    grads = 0
    n = len(data_loader.dataset)
    for batch_idx, (data, target) in enumerate(data_loader):
        loss = loss_fn(model(data.to(device)), target.to(device)) / n
        grads_param = torch.autograd.grad(loss, inputs=model.parameters(), create_graph=True)
        grads += parameters_to_vector(grads_param)
    dom_grad = eigvecs @ (eigvecs.T @ grads.to(eigvecs))
    return grads.detach().cpu().numpy(), torch.norm(dom_grad) / torch.norm(grads)
    

def compute_eigenvalues(graphs, model, criterion_summed, weight_decay, loader, loader_test, num_classes, device, analysis_list, use_hf_model=False):
    model.train()
    disable_running_stats(model) #set momentum to 0, do not update running mean or var.
    #running mean requires_grad is set to False. No worry about having computing gradient.
    eigs, evecs = get_hessian_eigenvalues_weight_decay(model, criterion_summed, weight_decay, loader, neigs=1, num_classes=num_classes, device=device, use_hf_model=use_hf_model)
    #eigs, _, _, _ = get_hessian_eigenvalues(model, criterion_summed, lr, analysis_dataset, neigs=10, return_smallest = False)
    graphs.eigs.append(eigs[0].item())
    if "evec" in analysis_list:
        graphs.evec.append(evecs[:, 0].cpu().numpy())
    
    print("train eigs:", graphs.eigs)

    if 'grad_evec_cos' in analysis_list:
        grad, grad_evecs_cos = project_and_step(model, evecs, loader, device, criterion_summed)
        print("grad and eigenspace alignment", grad_evecs_cos.item())
        graphs.grad_evecs_cos.append(grad_evecs_cos.item())
        graphs.grads.append(grad)
        graphs.weight.append(model.w_plus.detach().cpu().numpy())

    eigs_test, _ = get_hessian_eigenvalues_weight_decay(model, criterion_summed, weight_decay, loader_test, neigs=10, num_classes=num_classes, device=device, use_hf_model=use_hf_model)
    #eigs, _, _, _ = get_hessian_eigenvalues(model, criterion_summed, lr, analysis_dataset, neigs=10, return_smallest = False)
    graphs.eigs_test.append(eigs_test[0].item())
    print("test eigs:", graphs.eigs_test)
    enable_running_stats(model)

def compute_gn_eigenvalues(graphs, loss_name, model, loader, num_classes, device):
    from utilities import get_gauss_newton_eigenvalues
    model.train()
    disable_running_stats(model) #set momentum to 0, do not update running mean or var.
    #running mean requires_grad is set to False. No worry about having computing gradient.
    eigs, _ = get_gauss_newton_eigenvalues(loss_name, model, loader.dataset, neigs=10, num_class=num_classes, device=device)
    #eigs, _, _, _ = get_hessian_eigenvalues(model, criterion_summed, lr, analysis_dataset, neigs=10, return_smallest = False)
    graphs.gn_eigs.append(eigs[0].item())
    
    print("train gn eigs:", graphs.gn_eigs)
    
    """
    eigs_test, _ = get_gn_eigenvalues_weight_decay(model, criterion_summed, weight_decay, loader_test, neigs=10, num_classes=num_classes, device=device)
    #eigs, _, _, _ = get_hessian_eigenvalues(model, criterion_summed, lr, analysis_dataset, neigs=10, return_smallest = False)
    graphs.gn_eigs_test.append(eigs_test[0].item())
    print("test eigs:", graphs.eigs_test)
    """
    enable_running_stats(model)

def compute_eigenvalues_hf(graphs, model, criterion_summed, weight_decay, loader, loader_test, num_classes, device): 
    model.train()
    disable_running_stats(model) #set momentum to 0, do not update running mean or var.
    #running mean requires_grad is set to False. No worry about having computing gradient.
    eigs, _ = get_hessian_eigenvalues_weight_decay_hf(model, criterion_summed, weight_decay, loader, neigs=10, num_classes=num_classes, device=device)
    #eigs, _, _, _ = get_hessian_eigenvalues(model, criterion_summed, lr, analysis_dataset, neigs=10, return_smallest = False)
    graphs.eigs.append(eigs[0].item())
    
    #print("train eigs:", graphs.eigs)
    

    eigs_test, _ = get_hessian_eigenvalues_weight_decay_hf(model, criterion_summed, weight_decay, loader_test, neigs=10, num_classes=num_classes, device=device)
    #eigs, _, _, _ = get_hessian_eigenvalues(model, criterion_summed, lr, analysis_dataset, neigs=10, return_smallest = False)
    graphs.eigs_test.append(eigs_test[0].item())
    #print("test eigs:", graphs.eigs_test)
    enable_running_stats(model)