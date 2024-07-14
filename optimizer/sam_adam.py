import torch
from torch.optim import SGD, Adam
import sys


class AdamS_v1(torch.optim.Optimizer):
    def __init__(self, params, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(AdamS_v1, self).__init__(params, defaults)

        self.lr = kwargs["lr"]
        self.base_optimizer = Adam(self.param_groups, lr=self.lr, betas=(kwargs["momentum"], 0.999), weight_decay=kwargs["weight_decay"])
        #self.sgd_optimizer = SGD(self.param_groups, lr=self.lr, momentum=0.0, weight_decay=0.0)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["old_p"] = p.data.clone()

        self.base_optimizer.step()
        #print(self.adam_optimizer.state_dict()['state'][0].keys())
        #sys.exit()

        #self.sgd_optimizer.step()
        #for group in self.param_groups:
        #    for p in group["params"]:
        #        print(self.state[p]['exp_avg'])

        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["adam"] = (self.state[p]["old_p"] - p.data).clone() / self.lr
                #print(torch.norm(self.state[p]["adam"] - p.grad), torch.norm(self.state[p]["adam"]), torch.norm(p.grad))
        

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"] # recover to original parameter
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        adam_norm = self._get_attr_norm("adam")
        grad_norm = self._grad_norm()
        #print(self._get_attr_norm("adam"), )
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                p.grad = self.state[p]["adam"] + 1.0 * adam_norm * p.grad / grad_norm

                p.add_(p.grad, alpha=-self.lr) 

                #print(self.state[p]['exp_avg'])

        #self.adam_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

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
        
        """
        for group in self.param_groups:
            for p in group["params"]:
                print(self.state[p]['exp_avg'])
        sys.exit()
        """