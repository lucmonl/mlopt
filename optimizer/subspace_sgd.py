import torch
from utilities import get_hessian_eigenvalues_weight_decay
from optimizer.sam import disable_running_stats, enable_running_stats
import sys
from torch.utils.data.dataset import TensorDataset
from utilities import grads_to_vector, vector_to_grads

from typing import List

from backpack import backpack, extend
from backpack.utils.examples import _autograd_ggn_exact_columns
from torch.nn import Linear, MSELoss, ReLU, Sequential

from vivit.linalg.eigh import EighComputation

from torch import cat
import numpy as np


class GN_DOM_SGD(torch.optim.Optimizer):
    def __init__(self, model, params, dom_dim, criterion, criterion_summed, device, **kwargs):
        defaults = dict(**kwargs)
        super(GN_DOM_SGD, self).__init__(params, defaults)
        #self.norm_sgd_lr = norm_sgd_lr
        self.dom_dim = dom_dim
        self.model = model
        self.criterion = criterion
        self.criterion_summed = criterion_summed
        self.device = device

        self.base_optimizer = torch.optim.SGD(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    def select_top_k(self, evals) -> List[int]:
        """Select the top-k eigenvalues for the eigenvector computation.

        Args:
            evals: Eigenvalues, sorted in ascending order.
            k: Number of leading eigenvalues. Defaults to ``4``.

        Returns:
            Indices of top-k eigenvalues.
        """
        k = self.dom_dim
        return [evals.numel() - k + idx for idx in range(k)]
    
    def get_gn_eigs(self, X, y):
        model = extend(self.model)
        loss_function = extend(self.criterion)
        loss = loss_function(model(X), y)

        computation = EighComputation()
        group = {
            "params": [p for p in model.parameters() if p.requires_grad],
            "criterion": self.select_top_k,
        }
        param_groups = [group]

        extension = computation.get_extension()
        extension_hook = computation.get_extension_hook(param_groups)

        with backpack(extension, extension_hook=extension_hook):
            loss.backward()

        evals, evecs = computation.get_result(group)
        evecs_flat = cat([e.flatten(start_dim=1) for e in evecs], dim=1).T # [k * p]
        return evals, evecs_flat
    
    @torch.no_grad()
    def project_and_step(self, eigvecs):
        grads = grads_to_vector(self.model.parameters())
        dom_grad = eigvecs @ (eigvecs.T @ grads)
        return dom_grad, torch.norm(dom_grad) / torch.norm(grads)
    
    def step(self, epoch, batch_idx, batch, zero_grad=False, train_stats=False):
        disable_running_stats(self.model)

        dominant_alignment = torch.ones(1) * -1
        hessian_gn_align = torch.ones(1) * -1
        hes_eigvals, eigvals = [torch.ones(1) * -1], [torch.ones(1) * -1]

        if epoch >= 500 and batch_idx % 1 == 0: 
            eigvals, eigvecs = self.get_gn_eigs(batch[0].to(self.device), batch[1].to(self.device))    
            #eigvecs = torch.flip(eigvecs, dim=1)      
            dom_grad, dominant_alignment = self.project_and_step(eigvecs.to(self.device))
            
            #print(eigvals)
            #if dominant_alignment > 0.95:

            batch_dataset = TensorDataset(batch[0], batch[1])
            batch_loader = torch.utils.data.DataLoader(batch_dataset, batch_size=batch[0].shape[0], shuffle=True)

            hes_eigvals, hes_eigvecs = get_hessian_eigenvalues_weight_decay(self.model, self.criterion_summed, 
                                                                0.0, batch_loader, 
                                                                neigs=10, 
                                                                num_classes=10,
                                                                device=self.device, 
                                                                use_hf_model=False)
            #print(eigvals)
            #print([(hes_eigvecs[:, i] @ eigvecs[:, 9-i].cpu()).abs().item() for i in range(10)])
            #hessian_gn_align = np.mean([(hes_eigvecs[:, i] @ eigvecs[:, 9-i].cpu()).abs().item() for i in range(self.dom_dim)])
            #print((hes_eigvecs.T @ eigvecs.cpu()).abs())
            hessian_gn_align = torch.mean(torch.norm(hes_eigvecs @ (hes_eigvecs.T @ eigvecs.cpu()), dim=0))
            
            vector_to_grads(dom_grad, self.model.parameters())
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
        enable_running_stats(self.model)

        if train_stats:
            return {"dominant_alignment": dominant_alignment.item(), "hessian_gn_align": hessian_gn_align.item(),
                    "hessian_eig": hes_eigvals[0].item(), "gn_eig": eigvals[-1].item()}

class GN_BULK_SGD(torch.optim.Optimizer):
    def __init__(self, model, params, dom_dim, criterion, criterion_summed, device, **kwargs):
        defaults = dict(**kwargs)
        super(GN_BULK_SGD, self).__init__(params, defaults)
        #self.norm_sgd_lr = norm_sgd_lr
        self.dom_dim = dom_dim
        self.model = model
        self.criterion = criterion
        self.criterion_summed = criterion_summed
        self.device = device

        self.base_optimizer = torch.optim.SGD(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    def select_top_k(self, evals) -> List[int]:
        """Select the top-k eigenvalues for the eigenvector computation.

        Args:
            evals: Eigenvalues, sorted in ascending order.
            k: Number of leading eigenvalues. Defaults to ``4``.

        Returns:
            Indices of top-k eigenvalues.
        """
        k = self.dom_dim
        return [evals.numel() - k + idx for idx in range(k)]
    
    def get_gn_eigs(self, X, y):
        model = extend(self.model)
        loss_function = extend(self.criterion)
        loss = loss_function(model(X), y)

        computation = EighComputation()
        group = {
            "params": [p for p in model.parameters() if p.requires_grad],
            "criterion": self.select_top_k,
        }
        param_groups = [group]

        extension = computation.get_extension()
        extension_hook = computation.get_extension_hook(param_groups)

        with backpack(extension, extension_hook=extension_hook):
            loss.backward()

        evals, evecs = computation.get_result(group)
        evecs_flat = cat([e.flatten(start_dim=1) for e in evecs], dim=1).T # [k * p]
        return evals, evecs_flat
    
    @torch.no_grad()
    def project_and_step(self, eigvecs):
        grads = grads_to_vector(self.model.parameters())
        dom_grad = eigvecs @ (eigvecs.T @ grads)
        return grads-dom_grad, torch.norm(dom_grad) / torch.norm(grads)
    
    def step(self, epoch, batch_idx, batch, zero_grad=False, train_stats=False):
        disable_running_stats(self.model)

        dominant_alignment = torch.ones(1) * -1
        hessian_gn_align = torch.ones(1) * -1
        hes_eigvals, eigvals = [torch.ones(1) * -1], [torch.ones(1) * -1]

        if epoch >= 90 and batch_idx % 1 == 0: 
            eigvals, eigvecs = self.get_gn_eigs(batch[0].to(self.device), batch[1].to(self.device))    
            #eigvecs = torch.flip(eigvecs, dim=1)      
            dom_grad, dominant_alignment = self.project_and_step(eigvecs.to(self.device))
            
            #print(eigvals)
            #if dominant_alignment > 0.95:

            batch_dataset = TensorDataset(batch[0], batch[1])
            batch_loader = torch.utils.data.DataLoader(batch_dataset, batch_size=batch[0].shape[0], shuffle=True)

            hes_eigvals, hes_eigvecs = get_hessian_eigenvalues_weight_decay(self.model, self.criterion_summed, 
                                                                0.0, batch_loader, 
                                                                neigs=10, 
                                                                num_classes=10,
                                                                device=self.device, 
                                                                use_hf_model=False)
            #print(eigvals)
            #print([(hes_eigvecs[:, i] @ eigvecs[:, 9-i].cpu()).abs().item() for i in range(10)])
            #hessian_gn_align = np.mean([(hes_eigvecs[:, i] @ eigvecs[:, 9-i].cpu()).abs().item() for i in range(self.dom_dim)])
            #print((hes_eigvecs.T @ eigvecs.cpu()).abs())
            hessian_gn_align = torch.mean(torch.norm(hes_eigvecs @ (hes_eigvecs.T @ eigvecs.cpu()), dim=0))
            
            vector_to_grads(dom_grad, self.model.parameters())
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
        enable_running_stats(self.model)

        if train_stats:
            return {"dominant_alignment": dominant_alignment.item(), "hessian_gn_align": hessian_gn_align.item(),
                    "hessian_eig": hes_eigvals[0].item(), "gn_eig": eigvals[-1].item()}


class DOM_SGD(torch.optim.Optimizer):
    def __init__(self, model, params, dom_dim, criterion_summed, batch_size, num_classes, device, use_hf_model=False, start=0, end=-1, eigs_pattern="LM", **kwargs):
        defaults = dict(**kwargs)
        super(DOM_SGD, self).__init__(params, defaults)
        #self.norm_sgd_lr = norm_sgd_lr
        self.dom_dim = dom_dim
        self.model = model
        self.criterion_summed = criterion_summed
        self.batch_size = batch_size
        self.weight_decay = kwargs["weight_decay"]
        self.num_classes = num_classes
        self.device = device
        self.use_hf_model = use_hf_model

        self.start_eigs = start
        self.end_eigs = num_classes if end == -1 else end
        self.which_eigs = eigs_pattern

        self.base_optimizer = torch.optim.SGD(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def project_and_step(self, eigvecs):
        grads = grads_to_vector(self.model.parameters())
        dom_grad = eigvecs @ (eigvecs.T @ grads)
        return dom_grad, torch.norm(dom_grad) / torch.norm(grads)
        

    #@torch.no_grad()
    def step(self, epoch, batch_id, batch, zero_grad=False, train_stats=False):
        disable_running_stats(self.model)

        dominant_alignment = torch.ones(1) * -1

        if epoch >= 90 and batch_id % 1 == 0: 
            batch_dataset = TensorDataset(batch[0], batch[1])
            batch_loader = torch.utils.data.DataLoader(batch_dataset, batch_size=batch[0].shape[0], shuffle=True)

            _, eigvecs = get_hessian_eigenvalues_weight_decay(self.model, self.criterion_summed, 
                                                                self.weight_decay, batch_loader, 
                                                                neigs=self.end_eigs, 
                                                                num_classes=self.num_classes,
                                                                device=self.device, 
                                                                which_eigs= self.which_eigs,
                                                                use_hf_model=self.use_hf_model) 
            #eigvecs shape [p, neigs]
            eigvecs = eigvecs[:, self.start_eigs: self.end_eigs]
          
            dom_grad, dominant_alignment = self.project_and_step(eigvecs.to(self.device))
            #if dominant_alignment > 0.95:
            vector_to_grads(dom_grad, self.model.parameters())
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
        enable_running_stats(self.model)

        if train_stats:
            return {"dominant_alignment": dominant_alignment.item()}

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        (p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class BULK_SGD(torch.optim.Optimizer):
    def __init__(self, model, params, dom_dim, criterion_summed, batch_size, num_classes, device, use_hf_model=False, end=-1, eigs_pattern="LM", **kwargs):
        defaults = dict(**kwargs)
        super(BULK_SGD, self).__init__(params, defaults)
        #self.norm_sgd_lr = norm_sgd_lr
        self.dom_dim = dom_dim
        self.model = model
        self.criterion_summed = criterion_summed
        self.batch_size = batch_size
        self.weight_decay = kwargs["weight_decay"]
        self.num_classes = num_classes
        self.device = device
        self.use_hf_model = use_hf_model
        self.end_eigs = num_classes if end == -1 else end
        self.eigs_pattern = eigs_pattern

        self.base_optimizer = torch.optim.SGD(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def project_and_step(self, eigvecs):
        grads = grads_to_vector(self.model.parameters())
        dom_grad = eigvecs @ (eigvecs.T @ grads)
        return grads - dom_grad, torch.norm(dom_grad) / torch.norm(grads)
        

    #@torch.no_grad()
    def step(self, epoch, batch_id, batch, zero_grad=False, train_stats=False):
        disable_running_stats(self.model)

        dominant_alignment = torch.ones(1) * -1

        if epoch >= 90 and batch_id % 1 == 0: 
            batch_dataset = TensorDataset(batch[0], batch[1])
            batch_loader = torch.utils.data.DataLoader(batch_dataset, batch_size=batch[0].shape[0], shuffle=True)

            _, eigvecs = get_hessian_eigenvalues_weight_decay(self.model, self.criterion_summed, 
                                                                self.weight_decay, batch_loader, 
                                                                neigs=self.end_eigs,
                                                                num_classes=self.num_classes,
                                                                device=self.device,
                                                                which_eigs = self.eigs_pattern,
                                                                use_hf_model=self.use_hf_model) 
        #eigvecs shape [p, neigs]
        
            dom_grad, dominant_alignment = self.project_and_step(eigvecs.to(self.device))
            #if dominant_alignment > 0.95:
            vector_to_grads(dom_grad, self.model.parameters())
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
        enable_running_stats(self.model)

        if train_stats:
            return {"dominant_alignment": dominant_alignment.item()}

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        (p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups