import torch
from collections import defaultdict


class LORA_RITE(torch.optim.Optimizer):
    def __init__(self, model, lr=0.001, beta=0.9, output_layer_name=None, version="v1"):
        defaults = dict(lr=lr, beta=beta)
        super(LORA_RITE, self).__init__(model.parameters(), defaults)
        self.model = model
        self.lr = lr
        self.beta = beta
        self.output_layer_name = output_layer_name
        self.version = version

        from torch.optim import Adam
        output_params_list = []
        for n, p in model.named_parameters():
            if p.requires_grad and self.output_layer_name and self.output_layer_name in n:
                output_params_list.append(p)
        self.output_layer_optimizer = Adam(output_params_list, lr=self.lr, betas=(beta, 0.999), weight_decay=0.0)
        

    @torch.no_grad()
    def zero_grad(self):
        self.model.zero_grad()
        return
    
    @torch.no_grad()
    def lora_rite(self, lora_A_param, lora_B_param, update_B=True):
        #compute transformation invariant gradient
        print(torch.norm(lora_A_param))
        if update_B:
            U_A, R_A = torch.linalg.qr(lora_A_param.T, mode='reduced') #U_A n*r, R_A r*r
            R_A_inv = torch.linalg.pinv(R_A)
            unmagnified_B_grad = lora_B_param.grad @ R_A_inv #m*r
        else:
            U_A, R_A = torch.linalg.qr(lora_A_param, mode='reduced') #U_A m*r, R_A r*r
            R_A_inv = torch.linalg.pinv(R_A)
            unmagnified_B_grad = lora_B_param.grad.T @ R_A_inv #n*r
        
        # subspace projection
        if "U" in self.state[lora_A_param]:
            P_B = U_A.T @ self.state[lora_A_param]["U"] #r*r
        else:
            P_B = U_A.T @ U_A

        #update U
        self.state[lora_A_param]["U"] = torch.clone(U_A)

        # projected second moment
        if "sq_avg" in self.state[lora_B_param]:
            projected_sq_avg_B = P_B @ self.state[lora_B_param]["sq_avg"] @ P_B.T #r*r
            eigvals_sq_avg = torch.linalg.eigvals(self.state[lora_B_param]["sq_avg"])
            eigvals_proj_sq_avg = torch.linalg.eigvals(projected_sq_avg_B)
            d_lambda_B = torch.max(torch.abs(eigvals_sq_avg - eigvals_proj_sq_avg))
        else:
            projected_sq_avg_B = 0
            d_lambda_B = 0

        #update unmagnified second moment
        momentum_v = 0.999
        if not isinstance(projected_sq_avg_B, int):
            self.state[lora_B_param]["sq_avg"] = momentum_v * projected_sq_avg_B + (1- momentum_v) * unmagnified_B_grad.T @ unmagnified_B_grad #r*r
        else:
            self.state[lora_B_param]["sq_avg"] = unmagnified_B_grad.T @ unmagnified_B_grad
        #update escaped mass
        if "escape_mass" in self.state[lora_B_param]:
            self.state[lora_B_param]["escape_mass"] += d_lambda_B
        else:
            self.state[lora_B_param]["escape_mass"] = d_lambda_B

        #unmagnified precondition step
        print(torch.norm(self.state[lora_B_param]["sq_avg"]))
        if torch.norm(self.state[lora_B_param]["sq_avg"]) > 1e-6:
            sq_avg_eigs, sq_avg_eigvecs = torch.linalg.eigh(self.state[lora_B_param]["sq_avg"]) #eigs L: r; eigvecs Q: r*r Q diag(L) Q^T = RHS
            # avoid 0 in diags
            damped_sq_avg_eigs = sq_avg_eigs + self.state[lora_B_param]["escape_mass"] #r
            sq_avg_eigs_pow = torch.zeros_like(damped_sq_avg_eigs)
            mask = damped_sq_avg_eigs > 1e-8
            sq_avg_eigs_pow[mask] =  damped_sq_avg_eigs[mask].pow(-0.5)
            #damped_sq_avg_eigs = damped_sq_avg_eigs * (torch.abs(damped_sq_avg_eigs) > 1e-8)
            if self.version == "v1":
                precondition_step = unmagnified_B_grad @ sq_avg_eigvecs @ torch.diag(sq_avg_eigs_pow) @ sq_avg_eigvecs.T #m*r
            elif self.version == "v2":
                precondition_step = unmagnified_B_grad #m*r
                sq_avg = sq_avg_eigvecs @ torch.diag(sq_avg_eigs_pow) @ sq_avg_eigvecs.T
        else:
            precondition_step = unmagnified_B_grad
            sq_avg = torch.eye(self.state[lora_B_param]["sq_avg"].shape[0]).to(self.state[lora_B_param]["sq_avg"])

        #unmaginified first moment
        if "exp_avg" in self.state[lora_B_param]:
            self.state[lora_B_param]["exp_avg"] = self.beta * self.state[lora_B_param]["exp_avg"] @ P_B.T + (1 - self.beta) * precondition_step
        else:
            self.state[lora_B_param]["exp_avg"] = precondition_step

        #update parameters
        if self.version == "v1":
            if update_B:
                #lora_B_param.add_(-self.lr * self.state[lora_B_param]["exp_avg"] @ R_A_inv.T)
                lora_B_update = -self.lr * self.state[lora_B_param]["exp_avg"] @ R_A_inv.T
                return lora_B_update
            else:
                #lora_B_param.add_(-self.lr * (self.state[lora_B_param]["exp_avg"] @ R_A_inv.T).T)
                lora_A_update = -self.lr * (self.state[lora_B_param]["exp_avg"] @ R_A_inv.T).T
                return lora_A_update
        elif self.version == "v2":
            if update_B:
                #lora_B_param.add_(-self.lr * self.state[lora_B_param]["exp_avg"] @ R_A_inv.T)
                lora_B_update = -self.lr * self.state[lora_B_param]["exp_avg"] @ sq_avg @ R_A_inv.T
                return lora_B_update
            else:
                #lora_B_param.add_(-self.lr * (self.state[lora_B_param]["exp_avg"] @ R_A_inv.T).T)
                lora_A_update = -self.lr * (self.state[lora_B_param]["exp_avg"] @ sq_avg @ R_A_inv.T).T
                return lora_A_update
    
    @torch.no_grad()
    def step(self, zero_grad=True):
        adapter_names = []
        adapter_weights = {}
        #output_weights = {}
        #output_layer_name = opt_params["output_layer_name"]
        for name, param in self.model.named_parameters():  
            # select lora_A and lora_B
            if param.requires_grad:
                if self.output_layer_name and self.output_layer_name in name:
                    pass
                    #output_weights[name]= 0
                else:
                    adapter_names.append(name)
                    adapter_weights[name] = param

        for i in range(0, len(adapter_names), 2):
            lora_A_name, lora_B_name = adapter_names[i], adapter_names[i+1]
            lora_A_param, lora_B_param = adapter_weights[lora_A_name], adapter_weights[lora_B_name]
            print(lora_A_name, lora_B_name)

            lora_B_update = self.lora_rite(lora_A_param, lora_B_param, update_B = True)
            lora_A_update = self.lora_rite(lora_B_param, lora_A_param, update_B = False)
            lora_B_param.add_(lora_B_update)
            lora_A_param.add_(lora_A_update)
        
        if self.output_layer_name:
            self.output_layer_optimizer.step()

        if zero_grad:   
            self.output_layer_optimizer.zero_grad()
            self.model.zero_grad()
        