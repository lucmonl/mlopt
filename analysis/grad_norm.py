import torch


def get_grad_norm(model, ascent=False):
    #shared_device = model.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
    norm = torch.norm(
            torch.stack([
                (p.grad).norm(p=2)  #.to(shared_device)
                #for group in model.param_groups for p in group["params"]
                for p in model.parameters()
                if p.grad is not None
            ]),
            p=2
        )
    l1_norm = torch.norm(
            torch.stack([
                (p.grad).norm(p=1)  #.to(shared_device)
                #for group in model.param_groups for p in group["params"]
                for p in model.parameters()
                if p.grad is not None
            ]),
            p=1
        )
    if ascent:
       return {"ascent_grad_norm": norm.item(), "ascent_grad_l1_norm": l1_norm.item()} 
    return {"grad_norm": norm.item(), "grad_l1_norm": l1_norm.item()}