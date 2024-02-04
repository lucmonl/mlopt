import torch


class Normalized_Optimizer(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, norm_sgd_lr, **kwargs):
        defaults = dict(**kwargs)
        super(Normalized_Optimizer, self).__init__(params, defaults)
        self.norm_sgd_lr = norm_sgd_lr

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    def set_lr(self, lr):
        for param_group in self.param_groups:
            param_group['lr'] = lr

    @torch.no_grad()
    def step(self, loss=-1, accuracy=-1, zero_grad=False):
        if loss != -1 and loss > 0.01:
            self.base_optimizer.step()
            if zero_grad: self.zero_grad()
            return
        if accuracy !=-1 and accuracy < 0.95:
            self.base_optimizer.step()
            if zero_grad: self.zero_grad()
            return
        self.set_lr(self.norm_sgd_lr)
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = 1 / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                p.grad = p.grad * scale
                #e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                #p.add_(-e_w)  # climb to the local maximum "w + e(w)"

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

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