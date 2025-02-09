import torch
from optimizer.load_optimizer import load_optimizer
import copy
import torch.nn.functional as F

#def federated_lora_avg(model, loss_name, criterion, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, opt_params, server_epoch):
def federated_lora_avg(model, loss_name, criterion, lora_rank, train_graphs, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, opt_params, model_params, server_epoch):
    client_num, client_opt_name, client_epoch = opt_params["client_num"], opt_params["client_opt_name"], opt_params["client_epoch"]
    import copy
    import math
    from utilities import vector_to_grads, vector_to_grads_sq
    from main import train

    adapter_names = []
    adapter_weights = {}
    output_weights = {}
    output_layer_name = opt_params["output_layer_name"]
    for name, param in model.named_parameters():
        # select lora_A and lora_B
        if param.requires_grad:
            if output_layer_name and output_layer_name in name:
                output_weights[name]= 0
            else:
                adapter_names.append(name)
                adapter_weights[name] = torch.zeros_like(param)

    #from utilities import state_dict_to_vector, vector_to_state_dict
    # initialize client models, optimizers
        
    #running_stats = {}
    client_opt_params = copy.deepcopy(opt_params)
    client_opt_params["train_stats"] = False
    for client_id in range(client_num):
        # update client models
        client_model = copy.deepcopy(model)

        client_model.train()
        optimizer, lr_scheduler, _= load_optimizer(client_opt_name, client_model, client_lr, opt_params["client_momentum"], opt_params["client_weight_decay"], opt_params["lr_decay"], opt_params["epochs_lr_decay"], False, model_params, opt_params)
        #vector_to_parameters(old_params, client_model.parameters())
        for epoch in range(client_epoch):
            train(client_model, loss_name, criterion, device, train_loaders[client_id], optimizer, lr_scheduler, server_epoch, client_opt_params)
            
        for name, param in client_model.named_parameters():
            #print(name, param.shape)
            if param.requires_grad:
                #param_names.append(name)
                if output_layer_name and output_layer_name in name:
                    output_weights[name] += param.data / client_num
                elif name in adapter_weights:
                    #lora_params[name].append(param.data)
                    if 'lora_A' in name:
                        base_name = name.replace("lora_A.default", "base_layer")
                        adapter_weights[name] += param.data/client_num
                    elif 'lora_B' in name:
                        base_name = name.replace("lora_B.default", "base_layer")
                        adapter_weights[name] += param.data/client_num
                    else: assert False
                else:
                    assert False

    server_optimizer.zero_grad()

    if opt_params["train_stats"]:
        grad_norm = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            if output_layer_name and output_layer_name in name:
                param.grad = param.data - output_weights[name]
            elif name in adapter_weights:
                param.grad = param.data - adapter_weights[name]
            else:
                assert False

            if opt_params["train_stats"]:
                grad_norm += torch.norm(param.grad).item()**2
    
    if opt_params["train_stats"]:
        train_graphs.grad_norm.append(grad_norm ** 0.5)
        print("grad norm:", train_graphs.grad_norm[-1])

    server_optimizer.step()

    if opt_params["fedlora_uba"] >= 0:
        # rescale A and B matrices
        base_adapter_weights = {}
        for name, param in model.named_parameters():
            if 'lora_A' in name:
                base_name = name.replace("lora_A.default", "base_layer")
                base_adapter_weights[base_name] = {}
                base_adapter_weights[base_name]["A"] = param.data
            elif 'lora_B' in name:
                base_name = name.replace("lora_B.default", "base_layer")
                base_adapter_weights[base_name]["B"] = param.data
            
                
                base_full = (base_adapter_weights[base_name]["B"] @ base_adapter_weights[base_name]["A"]).T
            
                U, S, Vh = torch.linalg.svd(base_full, full_matrices=False)
                U_truncate, S_truncate, Vh_truncate = U[:, :lora_rank], torch.sqrt(S[:lora_rank]), Vh[:lora_rank, :]
                lora_A_name, lora_B_name = name.replace("lora_B.default", "lora_A.default"), name
                adapter_weights[lora_A_name].data = (U_truncate * S_truncate).T * opt_params["fedlora_uba"]
                adapter_weights[lora_B_name].data = Vh_truncate.T * S_truncate / opt_params["fedlora_uba"]
        
        # assign new param to model
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.data = adapter_weights[name].data
                #print(torch.norm(param.data))
    

    if server_lr_scheduler is not None:
        server_lr_scheduler.step()

    for group in server_optimizer.param_groups:
        print("server lr", group['lr'])

def federated_lora_fedex(model, loss_name, criterion, lora_rank, train_graphs, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, opt_params, model_params, server_epoch):
    client_num, client_opt_name, client_epoch = opt_params["client_num"], opt_params["client_opt_name"], opt_params["client_epoch"]
    import copy
    import math
    from utilities import vector_to_grads, vector_to_grads_sq
    from main import train

    base_weights = {}
    for name, param in model.named_parameters():
        if "base_layer" in name:
            base_weights[name] = torch.clone(param.data)

    adapter_names = []
    adapter_weights = {}
    output_weights = {}
    output_layer_name = opt_params["output_layer_name"]
    for name, param in model.named_parameters():
        # select lora_A and lora_B
        if param.requires_grad:
            if output_layer_name and output_layer_name in name:
                output_weights[name]= 0
            else:
                adapter_names.append(name)
                adapter_weights[name] = torch.zeros_like(param)

    #from utilities import state_dict_to_vector, vector_to_state_dict
    # initialize client models, optimizers
        
    #running_stats = {}
    client_opt_params = copy.deepcopy(opt_params)
    client_opt_params["train_stats"] = False
    for client_id in range(client_num):
        # update client models
        client_model = copy.deepcopy(model)

        client_model.train()
        optimizer, lr_scheduler, _= load_optimizer(client_opt_name, client_model, client_lr, opt_params["client_momentum"], opt_params["client_weight_decay"], opt_params["lr_decay"], opt_params["epochs_lr_decay"], False, model_params, opt_params)
        #vector_to_parameters(old_params, client_model.parameters())
        for epoch in range(client_epoch):
            train(client_model, loss_name, criterion, device, train_loaders[client_id], optimizer, lr_scheduler, server_epoch, client_opt_params)
            
        for name, param in client_model.named_parameters():
            #print(name, param.shape)
            if param.requires_grad:
                #param_names.append(name)
                if output_layer_name and output_layer_name in name:
                    output_weights[name] += param.data / client_num
                elif name in adapter_weights:
                    #lora_params[name].append(param.data)
                    if 'lora_A' in name:
                        base_name = name.replace("lora_A.default", "base_layer")
                        adapter_weights[name] += param.data/client_num
                        base_name_A, base_A_param = base_name, param.data
                    elif 'lora_B' in name:
                        base_name = name.replace("lora_B.default", "base_layer")
                        assert base_name == base_name_A #ensure this module to be the sequel of A
                        adapter_weights[name] += param.data/client_num
                        base_B_param = param.data

                        scaling = model_params["lora_alpha"] / model_params["lora_rank"]
                        base_weights[base_name] +=  scaling * (base_B_param @ base_A_param) / client_num
                    else: assert False
                else:
                    assert False

    #assign new adapters  and integrate Delta_W
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora_A' in name:
                base_name = name.replace("lora_A.default", "base_layer")
                param.data = adapter_weights[name].data
                base_name_A, base_A_param = base_name, param.data
            elif 'lora_B' in name:
                base_name = name.replace("lora_B.default", "base_layer")
                assert base_name == base_name_A #ensure this module to be the sequel of A
                param.data = adapter_weights[name].data
                base_B_param = param.data

                scaling = model_params["lora_alpha"] / model_params["lora_rank"]
                base_weights[base_name] -= scaling * (base_B_param @ base_A_param)
                #print(name, torch.norm(base_weights[base_name]).item())
            elif output_layer_name and output_layer_name in name:
                #print(name, torch.norm(param.data).item(), torch.norm(output_weights[name].data).item())
                param.data = output_weights[name].data

    # assign new param to model
    for name, param in model.named_parameters():
        if name in base_weights:
            param.data = base_weights[name]

    if server_lr_scheduler is not None:
        server_lr_scheduler.step()

    for group in server_optimizer.param_groups:
        print("server lr", group['lr'])

def federated_lora_flora(model, loss_name, criterion, lora_rank, train_graphs, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, opt_params, model_params, server_epoch):
    client_num, client_opt_name, client_epoch = opt_params["client_num"], opt_params["client_opt_name"], opt_params["client_epoch"]
    import copy
    import math
    from utilities import vector_to_grads, vector_to_grads_sq
    from main import train

    base_weights = {}
    for name, param in model.named_parameters():
        if "base_layer" in name:
            base_weights[name] = torch.clone(param.data)
    
    untouch_base_weights = {}
    for name, param in model.named_parameters():
        if "base_layer" in name:
            untouch_base_weights[name] = torch.clone(param.data)

    output_weights = {}
    output_layer_name = opt_params["output_layer_name"]
    for name, param in model.named_parameters():
        # select lora_A and lora_B
        if param.requires_grad:
            if output_layer_name and output_layer_name in name:
                output_weights[name]= 0

    #from utilities import state_dict_to_vector, vector_to_state_dict
    # initialize client models, optimizers
        
    #running_stats = {}
    client_opt_params = copy.deepcopy(opt_params)
    client_opt_params["train_stats"] = False
    for client_id in range(client_num):
        # update client models
        client_model = copy.deepcopy(model)

        client_model.train()
        optimizer, lr_scheduler, _= load_optimizer(client_opt_name, client_model, client_lr, opt_params["client_momentum"], opt_params["client_weight_decay"], opt_params["lr_decay"], opt_params["epochs_lr_decay"], False, model_params, opt_params)
        #vector_to_parameters(old_params, client_model.parameters())
        for epoch in range(client_epoch):
            train(client_model, loss_name, criterion, device, train_loaders[client_id], optimizer, lr_scheduler, server_epoch, client_opt_params)

        for name, param in client_model.named_parameters():
            #print(name, param.shape)
            if param.requires_grad:
                #param_names.append(name)
                if output_layer_name and output_layer_name in name:
                    output_weights[name] += param.data / client_num
                else: #lora modules
                    #lora_params[name].append(param.data)
                    if 'lora_A' in name:
                        base_name = name.replace("lora_A.default", "base_layer")
                        base_name_A, base_A_param = base_name, param.data
                    elif 'lora_B' in name:
                        base_name = name.replace("lora_B.default", "base_layer")
                        assert base_name == base_name_A #ensure this module to be the sequel of A
                        base_B_param = param.data

                        scaling = model_params["lora_alpha"] / model_params["lora_rank"]
                        original_base_weight_norm = torch.norm(base_weights[base_name])
                        base_weights[base_name] +=  scaling * (base_B_param @ base_A_param) / client_num
                        #print(base_name, torch.norm(base_weights[base_name]), original_base_weight_norm, torch.norm(base_weights[base_name] - untouch_base_weights[base_name]))
                        #print(base_weights[base_name])
                        #print(base_weights[base_name] - untouch_base_weights[base_name])
                    else: assert False

    print("original base layer")
    from arch.lora import get_base_layer_norm
    get_base_layer_norm(model)

    # assign new param to model and integrate Delta_W
    for name, param in model.named_parameters():
        if param.requires_grad:
            if output_layer_name and output_layer_name in name:
                param.data = output_weights[name]  

    for name, param in model.named_parameters():
        if name in base_weights:
            #print(name, torch.norm(base_weights[name] - param.data))
            param.data = base_weights[name]

    if server_lr_scheduler is not None:
        server_lr_scheduler.step()

    for group in server_optimizer.param_groups:
        print("server lr", group['lr'])

    print("update base layer")
    from arch.lora import get_base_layer_norm
    get_base_layer_norm(model)
    unload_model = model.unload()
    #return unload_model.state_dict()
    #model.delete_adapter("default")
    """
    print("merged model")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    
    from arch.lora import add_adapters_dataset
    client_model_merge = client_model.merge_and_unload()
    client_model_merge , _, _ = add_adapters_dataset(opt_params["model_name"], client_model_merge, model_params["lora_rank"], model_params["lora_alpha"], lora_freeze_a=opt_params["lora_freeze_a"])
    print("merged client model base layer")
    get_base_layer_norm(client_model_merge)
    for name,param in client_model_merge.named_parameters():
        if 'base_layer' in name:
            print(name, torch.norm(param), torch.norm(untouch_base_weights[name]), torch.norm(param-untouch_base_weights[name]))
            #print(param)
            #print(param-untouch_base_weights[name])
    """
    from arch.lora import add_adapters_dataset
    model , _, _ = add_adapters_dataset(opt_params["model_name"], unload_model, model_params["lora_rank"], model_params["lora_alpha"], lora_freeze_a=opt_params["lora_freeze_a"])
    
    """
    print("reinitialize lora module")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    """
    print("a new peft layer")
    from arch.lora import get_base_layer_norm
    get_base_layer_norm(model)
    return model

def get_topk_mask(x, density):
    mask = torch.zeros_like(x).bool()
    k = int(x.numel()*density)
    _, keep_idx = torch.topk(x, k=k)
    mask[keep_idx] = 1
    return mask

def federated_lora_flasc(model, loss_name, criterion, lora_rank, train_graphs, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, opt_params, model_params, server_epoch):
    client_num, client_opt_name, client_epoch = opt_params["client_num"], opt_params["client_opt_name"], opt_params["client_epoch"]
    import copy
    import math
    from utilities import vector_to_grads, vector_to_grads_sq
    from main import train


    server_params = {}
    output_weights = {}
    output_layer_name = opt_params["output_layer_name"]
    for name, param in model.named_parameters():
        # select lora_A and lora_B
        if param.requires_grad:
            if output_layer_name and output_layer_name in name:
                output_weights[name]= param
            else:
                server_params[name] = param

    #server_params = {n:p for n,p in model.named_parameters() if p.requires_grad}
    server_mask = {n:torch.ones_like(p) for n,p in server_params.items()}
    
    if model_params["dl_density"] < 1 or server_epoch == 1 : # one round of dense FT
        server_params_flat = torch.cat([p.flatten() for p in server_params.values()])
        server_mask_flat = get_topk_mask(x=server_params_flat.abs(), density=model_params["dl_density"])
        curr = 0
        for n,m in server_mask.items():
            server_mask[n] = server_mask_flat[curr:curr+m.numel()].reshape(m.shape)
            curr += m.numel()
    
    client_opt_params = copy.deepcopy(opt_params)
    client_opt_params["train_stats"] = False
    aggregate = None
    for client_id in range(client_num):
        # update client models
        neg_client_delta = {}
        client_model = copy.deepcopy(model)

        # Download Sparsity
        if model_params["dl_density"] < 1:
            for n,p in client_model.named_parameters():
                if p.requires_grad:
                    if output_layer_name and output_layer_name in n:
                        pass
                    else:
                        p.data = p.data*server_mask[n]
        

        client_model.train()
        optimizer, lr_scheduler, _= load_optimizer(client_opt_name, client_model, client_lr, opt_params["client_momentum"], opt_params["client_weight_decay"], opt_params["lr_decay"], opt_params["epochs_lr_decay"], False, model_params, opt_params)
        #vector_to_parameters(old_params, client_model.parameters())
        for epoch in range(client_epoch):
            train(client_model, loss_name, criterion, device, train_loaders[client_id], optimizer, lr_scheduler, server_epoch, client_opt_params)

        if model_params["dl_density"] < 1:
            #neg_client_delta = {n: (server_params[n].data*server_mask[n]) - cp.data for n,cp 
            #                    in client_model.named_parameters() if cp.requires_grad}
            for n, cp in client_model.named_parameters():
                # select lora_A and lora_B
                if cp.requires_grad:
                    if output_layer_name and output_layer_name in n:
                        pass
                    else:
                        neg_client_delta =  neg_client_delta | {n: (server_params[n].data*server_mask[n]) - cp.data}
        else:
            neg_client_delta = {n: server_params[n].data - cp.data for n,cp 
                                in client_model.named_parameters() if cp.requires_grad}
        
        #Upload Sparsity
        
        if model_params["ul_density"] < 1:
            # why not log this?
            client_delta_flat = torch.cat([p.flatten() for p in neg_client_delta.values()])
            client_mask_flat = get_topk_mask(x=client_delta_flat.abs(), density=model_params["ul_density"])
            curr = 0
            for n,p in neg_client_delta.items():
                p *= client_mask_flat[curr:curr+p.numel()].reshape(p.shape)
                curr += p.numel()

        for n, cp in client_model.named_parameters():
            # select output weights
            if cp.requires_grad:
                if output_layer_name and output_layer_name in n:
                    neg_client_delta = neg_client_delta | {n: (output_weights[n].data) - cp.data}        
        
        if aggregate is None:
            aggregate = neg_client_delta
        else:
            for n, delta in neg_client_delta.items():
                aggregate[n] += delta
    
    server_optimizer.zero_grad()
    for n, sp in server_params.items():
        sp.grad = aggregate[n] / client_num
    for n, sp in output_weights.items():
        sp.grad = aggregate[n] / client_num
    server_optimizer.step()


def federated_lora_het(model, loss_name, criterion, lora_rank, train_graphs, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, opt_params, model_params, server_epoch):
    from main import train
    
    adapter_names = []
    adapter_weights = {}
    output_weights = {}

    output_layer_name = opt_params["output_layer_name"]
    for name, param in model.named_parameters():
        # select lora_A and lora_B
        if param.requires_grad and output_layer_name not in name: # exclude the cls_head
            adapter_names.append(name)
            adapter_weights[name] = param
        if output_layer_name in name:
            output_weights[name]= 0
    
    if opt_params["train_stats"]:
        norm_A, norm_B = 0, 0 
        for name in adapter_weights:
            if 'lora_A' in name:
                norm_A += torch.norm(adapter_weights[name]) ** 2
            elif 'lora_B' in name:
                norm_B += torch.norm(adapter_weights[name]) ** 2
        #print("param norms: ", norm_A.item(), norm_B.item())
        train_graphs.lora_A_norm.append(norm_A.item())
        train_graphs.lora_B_norm.append(norm_B.item())
    
    client_num, client_opt_name, client_epoch = opt_params["client_num"], opt_params["client_opt_name"], opt_params["client_epoch"]
    
    base_names = []
    base_weights = {}
    base_adapter_weights = {}
    base_adapter_names = {}
    if "lora_param" not in opt_params:
        opt_params["lora_param"] = {}
        opt_params["lora_idx"] = {}
    for i in range(0, len(adapter_names), 2):
        lora_A_name, lora_B_name = adapter_names[i], adapter_names[i+1]
        lora_A_param, lora_B_param = adapter_weights[lora_A_name], adapter_weights[lora_B_name]
        base_weight_name = lora_A_name.replace("lora_A.default", "base_layer")
        base_adapter_weights[base_weight_name] = lora_B_param @ lora_A_param
        base_names.append(base_weight_name)
        base_adapter_names[base_weight_name] = [lora_A_name, lora_B_name]


    for name, param in model.named_parameters():
        if output_layer_name in name:
            base_weights[name] = torch.clone(param.data)

    #print(1)
    #for name in opt_params["server_params"]:
    #    print(name, torch.norm(opt_params["server_params"][name]).item())

    aggregated_weights = {}
    idx_cnt = {}
    for client_id in range(client_num):
        from main import train
        # update client models
        if server_epoch != 1:
            for name, param in model.named_parameters():
                if name in base_adapter_names.keys():
                    U, S, Vh = opt_params["U"][name], opt_params["S"][name], opt_params["Vh"][name]

                    #samp_dist = torch.distributions.Categorical(logits=S)
                    #client_idx = samp_dist.sample(sample_shape=(lora_rank))
                    #client_idx = torch.multinomial(input=S, replacement=False, num_samples=lora_rank)
                    client_idx = torch.arange(lora_rank).to(U).long()
                    #count the number of indices used for clients
                    opt_params["lora_idx"][name] = client_idx
                    if name not in idx_cnt:
                        idx_cnt[name] = 0
                    idx_cnt[name] += torch.sum(F.one_hot(client_idx, num_classes=S.shape[0]), dim=0)
                    
                    lora_A_name, lora_B_name = base_adapter_names[name]
                    
                    adapter_weights[lora_A_name].data = (U[:, client_idx] * S[client_idx]).T
                    adapter_weights[lora_B_name].data = Vh[client_idx, :].T * S[client_idx]
            #print("Client ID:", client_id, " ", idx_cnt[name][:lora_rank+5], " from ", S.shape[0])
        client_model = copy.deepcopy(model)
        client_model.train()

        optimizer, lr_scheduler, _= load_optimizer(client_opt_name, client_model, client_lr, opt_params["client_momentum"], opt_params["client_weight_decay"], opt_params["lr_decay"], opt_params["epochs_lr_decay"], False, model_params, opt_params)
        client_opt_params = copy.deepcopy(opt_params)
        client_opt_params["train_stats"] = False
        for epoch in range(client_epoch):
            train(client_model, loss_name, criterion, device, train_loaders[client_id], optimizer, lr_scheduler, server_epoch, client_opt_params)      
        
        if server_epoch != 1:
            for name, param in client_model.named_parameters():
                #print(name, param.shape)
                if param.requires_grad:
                    #param_names.append(name)
                    if output_layer_name in name:
                        output_weights[name] += param.data / client_num
                    elif name in opt_params["lora_param"]:
                        #lora_params[name].append(param.data)
                        if 'lora_A' in name:
                            base_name = name.replace("lora_A.default", "base_layer")
                            client_idx = opt_params["lora_idx"][base_name]
                            opt_params["lora_param"][name][client_idx,:] += param.data
                        elif 'lora_B' in name:
                            base_name = name.replace("lora_B.default", "base_layer")
                            client_idx = opt_params["lora_idx"][base_name]
                            opt_params["lora_param"][name][:, client_idx] += param.data
                        else: assert False
                    else:
                        assert False
        else:
            for name, param in client_model.named_parameters():
                #print(name, param.shape)
                if param.requires_grad:
                    if output_layer_name in name:
                        if name in output_weights:
                            output_weights[name] += param.data / client_num
                        else:
                            output_weights[name] = param.data / client_num
                    elif name in adapter_weights:
                        if name in opt_params["lora_param"]:
                            opt_params["lora_param"][name] += param.data / client_num
                        else:
                            opt_params["lora_param"][name] = param.data / client_num
                    else:
                        print(name)
                        assert False

    # normalize
    for i in range(0, len(adapter_names), 2):
        # A: r * input; B: output * r
        lora_A_name, lora_B_name = adapter_names[i], adapter_names[i+1]
        base_weight_name = lora_A_name.replace("lora_A.default", "base_layer")
        if server_epoch != 1:
            #print(torch.norm(opt_params["lora_param"][lora_A_name]))
            #print(torch.norm(opt_params["lora_param"][lora_A_name].T[:,16:]))
            nz_idx = torch.where(idx_cnt[base_weight_name] >0.1)[0]
            # only preserve the non-zero index
            opt_params["lora_param"][lora_A_name] = (opt_params["lora_param"][lora_A_name][nz_idx, :].T / (idx_cnt[base_weight_name][nz_idx] + 1e-6)).T
            opt_params["lora_param"][lora_B_name] = opt_params["lora_param"][lora_B_name][:, nz_idx] / (idx_cnt[base_weight_name][nz_idx] + 1e-6)
            print(idx_cnt[base_weight_name][:20])
            print(torch.norm(opt_params["lora_param"][lora_A_name]))
            #print(torch.norm(opt_params["lora_param"][lora_A_name].T[:,16:]))
            #print("===")
        aggregated_weights[base_weight_name] = opt_params["lora_param"][lora_B_name] @ opt_params["lora_param"][lora_A_name]


    server_optimizer.zero_grad()
    #for name, param in model.named_parameters():
    grad_norm = 0
    for name in opt_params["server_params"]:
        if name in aggregated_weights.keys():
            #param.requires_grad = True # going to update dense weight
            opt_params["server_params"][name].grad = (base_adapter_weights[name] - aggregated_weights[name]).T
            grad_norm += torch.linalg.norm(opt_params["server_params"][name].grad) ** 2
        elif output_layer_name in name:
            opt_params["server_params"][name].grad = base_weights[name].data - output_weights[name]
    
    server_optimizer.step()
    server_optimizer.zero_grad()

    for name, param in model.named_parameters():
        if name in aggregated_weights.keys():
            U, S, Vh = torch.linalg.svd(opt_params["server_params"][name].data, full_matrices=False)
            if "U" not in opt_params:
                opt_params["U"], opt_params["S"], opt_params["Vh"] = {}, {}, {}
            opt_params["U"][name], opt_params["S"][name], opt_params["Vh"][name] = U, torch.sqrt(S), Vh
            lora_A_name, lora_B_name = base_adapter_names[name]
            opt_params["lora_param"][lora_A_name] = torch.zeros_like(U.T)
            opt_params["lora_param"][lora_B_name] = torch.zeros_like(Vh.T)
        elif output_layer_name in name:
            param.data = opt_params["server_params"][name]