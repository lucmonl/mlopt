import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, train_stats=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

        self.track_cos_descent_ascent = train_stats
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

                if self.track_cos_descent_ascent:
                    #cos_descent_ascent += self.state[p]["descent_grad"] @ (p.grad.clone() / (grad_norm + 1e-12))
                    self.state[p]["ascent_grad"] = p.grad.clone().reshape(-1) / (grad_norm + 1e-12)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        cos_descent_ascent = 0
        if self.track_cos_descent_ascent:
            grad_norm = self._grad_norm()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

                if self.track_cos_descent_ascent:
                    cos_descent_ascent += (self.state[p]["ascent_grad"] @ (p.grad.clone().reshape(-1) / (grad_norm + 1e-12))).item()

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

        return cos_descent_ascent

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


class Replay_SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, train_stats=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(Replay_SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.replay_gradient = None
        self.track_cos_descent_ascent = train_stats

        self.defaults.update(self.base_optimizer.defaults)


    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()
        """
        """
        if self.replay_gradient:
            for (param_group, replay_group) in zip(self.param_groups, self.replay_gradient):
                for (p, g) in zip(group["params"]. replay_group["params"]):
                    p.add_(g)
        """
        cos_descent_ascent = 0
        if self.track_cos_descent_ascent:
            grad_norm = self._grad_norm()

        replay_norm = self._replay_norm()
        for group in self.param_groups:
            scale = group["rho"] / (replay_norm + 1e-12)
            for p in group["params"]:
                self.state[p]["old_p"] = p.data.clone()
                p.add_(self.state[p]["replay_gradient"] * scale.to(p))

                if self.track_cos_descent_ascent:
                    cos_descent_ascent += (self.state[p]["replay_gradient"]).reshape(-1) * scale.to(p) @ p.grad.reshape(-1) / (grad_norm + 1e-12).to(p)

        if zero_grad:
            self.zero_grad()
        return cos_descent_ascent.item()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                # update replay_gradient
                #self.state[p]["replay_gradient"] = torch.normal(mean = torch.zeros_like(p.grad), 
                #                                                std = torch.abs(p.grad.clone()))
                self.state[p]["replay_gradient"] = torch.normal(mean = 0, std = 1, size=p.grad.shape).to(p)
                                                               

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

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

    def _replay_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                torch.stack([
                    (self.state[p]["replay_gradient"]).norm(p=2).to(shared_device)
                    for group in self.param_groups for p in group["params"]
                    if "replay_gradient" in self.state[p]
                ]),
                p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class LookSAM(torch.optim.Optimizer):
    "from https://github.com/rollovd/LookSAM/blob/master/looksam.py"
    def __init__(self, alpha, params, base_optimizer, rho, **kwargs):

        """
        LookSAM algorithm: https://arxiv.org/pdf/2203.02714.pdf
        Optimization algorithm that capable of simultaneously minimizing loss and loss sharpness to narrow
        the generalization gap.

        :param k: frequency of SAM's gradient calculation (default: 10)
        :param model: your network
        :param criterion: your loss function
        :param base_optimizer: optimizer module (SGD, Adam, etc...)
        :param alpha: scaling factor for the adaptive ratio (default: 0.7)
        :param rho: radius of the l_p ball (default: 0.1)

        :return: None

        Usage:
            model = YourModel()
            criterion = YourCriterion()
            base_optimizer = YourBaseOptimizer
            optimizer = LookSAM(k=k,
                                alpha=alpha,
                                model=model,
                                base_optimizer=base_optimizer,
                                criterion=criterion,
                                rho=rho,
                                **kwargs)

            ...

            for train_index, data in enumerate(loader):
                loss = criterion(model(samples), targets)
                loss.backward()
                optimizer.step(t=train_index, samples=samples, targets=targets, zero_grad=True)

            ...

        """

        defaults = dict(alpha=alpha, rho=rho, **kwargs)
        super(LookSAM, self).__init__(params, defaults)

        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @staticmethod
    def normalized(g):
        return g / (g.norm(p=2) + 1e-8)

    def first_step(self, zero_grad=True):
        # do actual sharpness-aware update
        group = self.param_groups[0]
        scale = group['rho'] / (self._grad_norm() + 1e-8)

        for index_p, p in enumerate(group['params']):
            if p.grad is None:
                continue

            self.state[p]['old_p'] = p.data.clone()
            self.state[f'old_grad_p_{index_p}']['old_grad_p'] = p.grad.clone()

            with torch.no_grad():
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    def second_step(self, zero_grad=True):
        # update gv
        group = self.param_groups[0]
        for index_p, p in enumerate(group['params']):
            if p.grad is None:
                continue
            old_grad_p = self.state[f'old_grad_p_{index_p}']['old_grad_p']
            g_grad_norm = LookSAM.normalized(old_grad_p)
            g_s_grad_norm = LookSAM.normalized(p.grad)
            self.state[f'gv_{index_p}']['gv'] = torch.sub(p.grad, p.grad.norm(p=2) * torch.sum(
                g_grad_norm * g_s_grad_norm) * g_grad_norm)
            # recover data
            p.data = self.state[p]['old_p']
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _ip_g_g_s(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.sum(
            torch.stack([
                self.state[f'old_grad_p_{index_p}']['old_grad_p'].to(shared_device).reshape(-1) @ p.grad.to(shared_device).reshape(-1)
                for group in self.param_groups for index_p, p in enumerate(group['params'])
                if p.grad is not None
            ]),
        )
        return norm

    def _ip_g_gv(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.sum(
            torch.stack([
                self.state[f'gv_{index_p}']['gv'].to(shared_device).reshape(-1) @ self.state[f'old_grad_p_{index_p}']['old_grad_p'].to(shared_device).reshape(-1)
                for group in self.param_groups for index_p, p in enumerate(group['params'])
                if p.grad is not None
            ])
        )
        norm_1 = torch.norm(
            torch.stack([
                self.state[f'gv_{index_p}']['gv'].to(shared_device).norm(p=2)
                for group in self.param_groups for index_p, p in enumerate(group['params'])
                if p.grad is not None
            ]),
            p=2
        )
        norm_2 = torch.norm(
            torch.stack([
                self.state[f'old_grad_p_{index_p}']['old_grad_p'].to(shared_device).norm(p=2)
                for group in self.param_groups for index_p, p in enumerate(group['params'])
                if p.grad is not None
            ]),
            p=2
        )
        return norm, norm_1, norm_2

    def _old_grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                self.state[f'old_grad_p_{index_p}']['old_grad_p'].norm(p=2).to(shared_device) for group in self.param_groups for index_p, p in enumerate(group['params'])
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def second_step_v2(self, zero_grad = True):
        "what described in the paper"
        group = self.param_groups[0]
        scale = self._ip_g_g_s() / (self._old_grad_norm()**2 + 1e-8)

        for index_p, p in enumerate(group['params']):
            if p.grad is None:
                continue
            old_grad_p = self.state[f'old_grad_p_{index_p}']['old_grad_p']
            #g_grad_norm = LookSAM.normalized(old_grad_p)
            #g_s_grad_norm = LookSAM.normalized(p.grad)
            self.state[f'gv_{index_p}']['gv'] = torch.sub(p.grad, scale * old_grad_p)
            # recover data
            p.data = self.state[p]['old_p']

        #print(self._ip_g_gv())
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def normal_step(self, zero_grad=True):
        group = self.param_groups[0]
        for index_p, p in enumerate(group['params']):
            if p.grad is None:
                continue
            # retrieve gv and update grad
            with torch.no_grad():
                gv = self.state[f'gv_{index_p}']['gv']
                p.grad.add_(self.alpha.to(p) * (p.grad.norm(p=2) / (gv.norm(p=2) + 1e-8) * gv))
                #print((gv).norm(p=2), p.grad.norm(p=2))
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
            
    def normal_step_v2(self, zero_grad=True):
        group = self.param_groups[0]
        scale = self._grad_norm() / (self._gv_norm() + 1e-8)
        for index_p, p in enumerate(group['params']):
            if p.grad is None:
                continue
            # retrieve gv and update grad
            with torch.no_grad():
                gv = self.state[f'gv_{index_p}']['gv']
                #p.grad.add_(self.alpha.to(p) * scale * gv)
                p.grad.add_(0.1 * scale * gv)
                #print((gv).norm(p=2), p.grad.norm(p=2))
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, t, samples, targets, zero_grad=False):
        if not t % self.k:
            group = self.param_groups[0]
            scale = group['rho'] / (self._grad_norm() + 1e-8)

            for index_p, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                self.state[p]['old_p'] = p.data.clone()
                self.state[f'old_grad_p_{index_p}']['old_grad_p'] = p.grad.clone()

                with torch.no_grad():
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)

            self.criterion(self.model(samples), targets).backward()

        group = self.param_groups[0]
        for index_p, p in enumerate(group['params']):
            if p.grad is None:
                continue
            if not t % self.k:
                old_grad_p = self.state[f'old_grad_p_{index_p}']['old_grad_p']
                g_grad_norm = LookSAM.normalized(old_grad_p)
                g_s_grad_norm = LookSAM.normalized(p.grad)
                self.state[f'gv_{index_p}']['gv'] = torch.sub(p.grad, p.grad.norm(p=2) * torch.sum(
                    g_grad_norm * g_s_grad_norm) * g_grad_norm)

            else:
                with torch.no_grad():
                    gv = self.state[f'gv_{index_p}']['gv']
                    p.grad.add_(self.alpha.to(p) * (p.grad.norm(p=2) / (gv.norm(p=2) + 1e-8) * gv))

            p.data = self.state[p]['old_p']

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device) for group in self.param_groups for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )

        return norm

    def _gv_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                self.state[f'gv_{index_p}']['gv'].norm(p=2).to(shared_device)
                for group in self.param_groups for index_p, p in enumerate(group['params'])
                if p.grad is not None
            ]),
            p=2
        )

        return norm