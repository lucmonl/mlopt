import torch
from utilities import get_hessian_eigenvalues_weight_decay
from optimizer.sam import disable_running_stats, enable_running_stats
import sys
from torch.utils.data.dataset import TensorDataset
from utilities import grads_to_vector, vector_to_grads


class DOM_SGD(torch.optim.Optimizer):
    def __init__(self, model, params, dom_dim, criterion_summed, batch_size, num_classes, device, use_hf_model=False, **kwargs):
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

        self.base_optimizer = torch.optim.SGD(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)



    def set_lr(self, lr):
        for param_group in self.param_groups:
            param_group['lr'] = lr

    @torch.no_grad()
    def project_and_step(self, eigvecs):
        grads = grads_to_vector(self.model.parameters())
        dom_grad = eigvecs @ (eigvecs.T @ grads)
        return dom_grad, torch.norm(dom_grad) / torch.norm(grads)
        

    #@torch.no_grad()
    def step(self, batch, zero_grad=False, train_stats=False):
        disable_running_stats(self.model)
        batch_dataset = TensorDataset(batch[0], batch[1])
        batch_loader = torch.utils.data.DataLoader(batch_dataset, batch_size=batch[0].shape[0], shuffle=True)

        _, eigvecs = get_hessian_eigenvalues_weight_decay(self.model, self.criterion_summed, 
                                                             self.weight_decay, batch_loader, 
                                                             neigs=self.num_classes, 
                                                             num_classes=self.num_classes,
                                                             device=self.device, 
                                                             use_hf_model=self.use_hf_model) 
        #eigvecs shape [p, neigs]        
        dom_grad, dominant_alignment = self.project_and_step(eigvecs.to(self.device))
        if dominant_alignment > 0.95:
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