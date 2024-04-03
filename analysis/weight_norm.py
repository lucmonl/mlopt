import torch
#from sklearn.preprocessing import normalize

def get_min_weight_norm(graphs, model, C, model_name):
    norm_min = 1e10
    for name, param in model.named_parameters():
        """
        if 'weight_g' in name:  #if param.shape[1] == 1:
            continue
        if 'output_layer' in name:
            continue
        print(name)
        """
        if model_name in ["weight_norm", 'weight_norm_width_scale']:
            if 'output_layer' in name:
                continue
            norm_min = min(norm_min, torch.min(torch.norm(param, dim=-1)).cpu().detach().numpy())
        elif model_name in ['weight_norm_torch']:
            if 'weight_v' in name:
                norm_min = min(norm_min, torch.min(torch.norm(param, dim=-1)).cpu().detach().numpy())
        elif model_name in ['WideResNet_WN_woG']:
            if 'weight_v' in name:
                assert param.ndim == 4
                param_norm = torch.norm(param, dim=[1,2])
                norm_min = min(norm_min, torch.min(torch.norm(param_norm, dim=-1)).cpu().detach().numpy())
        else:
            raise NotImplementedError
    graphs.wn_norm_min.append(norm_min)

def get_grad_loss_ratio(graphs, model, loss_name, loader, criterion, criterion_summed, num_classes, device):
    loss_sum = 0
    model.zero_grad()
    for batch_idx, (data, target) in enumerate(loader, start=1):
        data, target = data.to(device), target.to(device)
        out = model(data)
        #if loss_name == 'CrossEntropyLoss':
        loss = criterion_summed(out, target)
        #elif loss_name == 'MSELoss':
        #loss = criterion_summed(out, target)
        loss_sum += loss
    loss_mean = loss_sum / len(loader.dataset)
    loss_mean.backward()
    grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += torch.norm(param.grad)**2
    graphs.wn_grad_loss_ratio.append((grad_norm/loss_mean).item())
    print("Grad Loss Ratio:", graphs.wn_grad_loss_ratio[-1], "\t Minimum Weight Norm:", graphs.wn_norm_min[-1])

def get_min_weight_norm_with_g(graphs, model, C, model_name):
    norm_min = 1e10
    for name, param in model.named_parameters():
        #print(name)
        #print(name, param.shape)
        """
        if 'weight_g' in name:  #if param.shape[1] == 1:
            continue
        if 'output_layer' in name:
            continue
        print(name)
        """
        if model_name == "weight_norm":
            if 'output_layer' in name:
                continue
            norm_min = min(norm_min, torch.min(torch.norm(param, dim=-1)).cpu().detach().numpy())
        if model_name == 'weight_norm_torch' and 'weight_g' in name:
            weight_g = param
        if model_name == 'weight_norm_torch' and 'weight_v' in name:
            print((weight_g * param).shape)
            norm_min = min(norm_min, torch.min(torch.norm(weight_g * param, dim=-1)).cpu().detach().numpy())
    graphs.wn_norm_min_with_g.append(norm_min)