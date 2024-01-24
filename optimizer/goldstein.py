import torch


class Goldstein(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, delta=0.05, adaptive=False, **kwargs):
        assert delta >= 0.0, f"Invalid delta, should be non-negative: {delta}"

        defaults = dict(delta=delta, adaptive=adaptive, **kwargs)
        super(Goldstein, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        r = grad_norm / 100
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["old_grad"] = p.grad.clone()
                #e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                #p.add_(e_w)  # climb to the local maximum "w + e(w)"
                
                #sample zeta uniformly from B_r(g_k)
                p.grad = (torch.rand_like(p.data)-0.5)*2*r + self.state[p]["old_grad"]

        #choose y_k uniformly from [x,x-delta hat zeta_k]
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["delta"]
            for p in group["params"]:
                if p.grad is None: continue
                p = self.state[p]["old_p"] - scale*torch.rand_like(p.data)*p.grad/grad_norm

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        #compute lambda
        coef_a, coef_b = 0, 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                coef_a += torch.sum((p.grad - self.state[p]["old_grad"])**2)
                coef_b += torch.sum((p.grad * (self.state[p]["old_grad"]-p.grad)))
                #p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        lam = -coef_b / coef_a
        if lam > 1: lam=1
        elif lam <0: lam=0
        #update grads
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.grad = lam*self.state[p]["old_grad"] + (1-lam)*p.grad
        #if zero_grad: self.zero_grad()

    @torch.no_grad()
    def third_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.grad = p.grad/(grad_norm + 1e-12)
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
        

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

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
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)