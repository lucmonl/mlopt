import torch
from torch.optim import SGD, Adam
import sys
from utilities import vector_to_group_grads
from torch.nn.utils import parameters_to_vector
from csvec import CSVec


class fetchSGD(torch.optim.Optimizer):
    def __init__(self, model, params, lr, sketch_size, momentum, **kwargs):
        self.p = len(parameters_to_vector(model.parameters()))
        defaults = dict(**kwargs)
        super(fetchSGD, self).__init__(params, defaults)

        self.lr = lr
        self.momentum = momentum
        self.sketch_size = sketch_size
        self.base_optimizer = SGD(self.param_groups, lr=1.0, momentum=0.0, weight_decay=0.0)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        
        self.moment, self.error_feedback = torch.zeros(5, sketch_size), torch.zeros(5, sketch_size)
        self.model = model
        #self.avg_sketch = torch.empty(sketch_size)
        #self.desketch_operator = None
        

    @torch.no_grad()
    def step(self, zero_grad=False):
        from hadamard_transform import hadamard_transform, pad_to_power_of_2 
        # momentum
        self.moment = self.momentum * self.moment.to(self.model.avg_sketch) + self.model.avg_sketch

        # error feedback
        self.error_feedback = self.error_feedback.to(self.moment) + self.lr * self.moment
        #self.error_feedback = self.lr * self.moment

        # unsketch
        #sketch = args2sketch(args)
        sketch = CSVec(d=self.p, c=self.sketch_size, r=5, device=self.model.avg_sketch.device, numBlocks=20)
        sketch.accumulateTable(self.error_feedback)
        topk_update = sketch.unSketch(k=self.sketch_size)
        """
        unsketch = self.model.desketch_operator @ self.error_feedback
        unsketch = hadamard_transform(unsketch) * self.model.D
        unsketch = unsketch[:self.p]
        _, topk_indx = torch.topk(unsketch**2, k=self.sketch_size, sorted=False)
        topk_update = torch.zeros_like(unsketch)
        topk_update[topk_indx] = unsketch[topk_indx]
        """
        # Error Accumulations
        sketch.zero()
        sketch.accumulateVec(topk_update)
        sketched_update = sketch.table
        #self.error_feedback -= sketched_update
        nz = sketched_update.nonzero()
        self.error_feedback[nz[:,0], nz[:,1]] = 0
        self.moment[nz[:,0], nz[:,1]] = 0
        
        """
        sketch_update = pad_to_power_of_2(topk_update)
        sketch_update = hadamard_transform(self.model.D*sketch_update)
        sketch_update = self.model.desketch_operator.T @ sketch_update
        self.error_feedback = self.error_feedback - sketch_update
        """
        # assign gradients
        #vector_to_group_grads(topk_update, self.base_optimizer.param_groups)
        vector_to_group_grads(topk_update, self.base_optimizer.param_groups)
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _get_attr_norm(self, attr):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                torch.stack([
                    (self.state[p][attr]).norm(p=2).to(shared_device)
                    for group in self.param_groups for p in group["params"]
                    if attr in self.state[p]
                ]),
                p=2
               )
        return norm

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        #print(state_dict['state'][0].keys())
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        #print(self.adam_optimizer.state_dict()['state'])