import torch
from optimizer.load_optimizer import load_optimizer
import copy
import torch.nn.functional as F
from arch.lora import synchronize_lora, merge_to_base
import numpy as np
import math

def compute_adapter_weight(model_name, lora_A_param, lora_B_param):
    if model_name in ["google/vit-base-patch16-224-in21k", "roberta-base", "meta-llama/Llama-3.2-3B"]:
        return lora_B_param @ lora_A_param
    elif model_name in ["gpt2"]:
        return (lora_B_param @ lora_A_param).T
    else:
        raise NotImplementedError


def compute_truncate_err(model, adapter_weights, client_num, model_name, server_name):
    truncate_err_list, truncate_err_ratio_list = [], []
    client_weights = {}
    import re
    # reorganize client adapter weights in one dict
    for name, param in model.named_parameters():
        if 'client_' not in name:
            # skip server adapter
            continue
        if 'lora' not in name:
            #skip output layer
            continue
        client_id = int(re.search(r"client_(\d+)\.", name).group(1))
        adapter_name = "client_{}".format(client_id)
        server_adapter_name = name.replace("{}".format(adapter_name), server_name)
        if server_adapter_name not in client_weights:
            client_weights[server_adapter_name] = {}
        """
        if 'lora_A' in name:
            if client_id not in client_weights[server_adapter_name]:
                client_weights[server_adapter_name][client_id] = {}
            client_weights[server_adapter_name][client_id]["A"] = param
        elif 'lora_B' in name:
            if client_id not in client_weights[server_adapter_name]:
                client_weights[server_adapter_name][client_id] = {}
            client_weights[server_adapter_name][client_id]["B"] = param
        """
        client_weights[server_adapter_name][client_id] = param

    #compute truncate error layer by layer
    for server_adapter_name in client_weights:
        assert len(client_weights[server_adapter_name].keys()) == client_num
        if 'lora_A' in server_adapter_name:
            pass
        else:
            continue
        layer_matrix = 0
        for client_id in client_weights[server_adapter_name]:
            lora_A_param = client_weights[server_adapter_name][client_id]
            server_adapter_name_B = server_adapter_name.replace("lora_A", "lora_B")
            lora_B_param = client_weights[server_adapter_name_B][client_id]
            layer_matrix += compute_adapter_weight(model_name, lora_A_param, lora_B_param) / client_num
        layer_matrix_gt_norm = torch.norm(layer_matrix, p='nuc')
        layer_matrix -= compute_adapter_weight(model_name, adapter_weights[server_adapter_name], adapter_weights[server_adapter_name_B])
        truncate_err = torch.norm(layer_matrix, p='nuc').item()
        truncate_err_ratio = (truncate_err / layer_matrix_gt_norm).item()
        truncate_err_list.append(truncate_err)
        truncate_err_ratio_list.append(truncate_err_ratio)

    print("Truncation Error List: ", truncate_err_list)
    print("Truncation Error Ratio List: ", truncate_err_ratio_list)
    return np.mean(truncate_err_list), np.mean(truncate_err_ratio_list)

def federated_lora_avg_v1(model, loss_name, criterion, lora_rank, train_graphs, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, opt_params, model_params, server_epoch):
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
        print(name, torch.norm(param).item())

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
                    print(name, torch.norm(param.data / client_num).item())
                elif name in adapter_weights:
                    #lora_params[name].append(param.data)
                    if 'lora_A' in name:
                        base_name = name.replace("lora_A.{}".format(opt_params["server_name"]), "base_layer")
                        adapter_weights[name] += param.data/client_num
                        print(name, torch.norm(param.data / client_num).item())
                    elif 'lora_B' in name:
                        base_name = name.replace("lora_B.{}".format(opt_params["server_name"]), "base_layer")
                        adapter_weights[name] += param.data/client_num
                        print(name, torch.norm(param.data / client_num).item())
                    else: assert False
                else:
                    assert False
    print("====== client ends ======")
    server_optimizer.zero_grad()

    if opt_params["train_stats"]:
        grad_norm = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            if output_layer_name and output_layer_name in name:
                param.grad = param.data - output_weights[name]
                print(name, torch.norm(param.grad).item(), torch.norm(param.data).item(), torch.norm(output_weights[name]).item())
            elif name in adapter_weights:
                param.grad = param.data - adapter_weights[name]
                print(name, torch.norm(param.grad).item(), torch.norm(param.data).item(), torch.norm(adapter_weights[name]).item())
            else:
                assert False

            if opt_params["train_stats"]:
                grad_norm += torch.norm(param.grad).item()**2
            #print(name, torch.norm(param.grad).item())
    print("======= pseudo grad ends =======")
    
    if opt_params["train_stats"]:
        train_graphs.grad_norm.append(grad_norm ** 0.5)
        print("grad norm:", train_graphs.grad_norm[-1])

    server_optimizer.step()

    if opt_params["fedlora_uba"] >= 0:
        # rescale A and B matrices
        base_adapter_weights = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'lora_A' in name:
                base_name = name.replace("lora_A.{}".format(opt_params["server_name"]), "base_layer")
                base_adapter_weights[base_name] = {}
                base_adapter_weights[base_name]["A"] = param.data
            elif 'lora_B' in name:
                base_name = name.replace("lora_B.{}".format(opt_params["server_name"]), "base_layer")
                base_adapter_weights[base_name]["B"] = param.data
            
                B_norm, A_norm = torch.norm(base_adapter_weights[base_name]["B"]), torch.norm(base_adapter_weights[base_name]["A"])
                base_full = (base_adapter_weights[base_name]["B"] @ base_adapter_weights[base_name]["A"]).T
            
                U, S, Vh = torch.linalg.svd(base_full, full_matrices=False)
                U_truncate, S_truncate, Vh_truncate = U[:, :lora_rank], torch.sqrt(S[:lora_rank]), Vh[:lora_rank, :]
                lora_A_name, lora_B_name = name.replace("lora_B.{}".format(opt_params["server_name"]), "lora_A.{}".format(opt_params["server_name"])), name

                S_norm = torch.norm(S_truncate)
                print("uba mode is " + opt_params["uba_mode"])
                if opt_params["uba_mode"] == "ada":
                    print("B_norm", B_norm, "A_norm", A_norm, "S_norm", S_norm)
                    ratio = (A_norm + opt_params["uba_weight"] * opt_params["fedlora_uba"]**2*S_norm) / (B_norm + opt_params["uba_weight"] * S_norm)
                    ratio = ratio**0.5
                else:
                    ratio = opt_params["fedlora_uba"]
                    
                adapter_weights[lora_A_name].data = (U_truncate * S_truncate).T * ratio
                adapter_weights[lora_B_name].data = Vh_truncate.T * S_truncate / ratio
        
        # assign new param to model
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'lora_A' in name or 'lora_B' in name:
                param.data = adapter_weights[name].data
                #print(torch.norm(param.data))
    

    if server_lr_scheduler is not None:
        server_lr_scheduler.step()

    for group in server_optimizer.param_groups:
        print("server lr", group['lr'])


#def federated_lora_avg(model, loss_name, criterion, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, opt_params, server_epoch):
def federated_lora_avg(model, loss_name, criterion, lora_rank, train_graphs, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, opt_params, model_params, server_epoch):
    client_num, client_opt_name, client_epoch = opt_params["client_num"], opt_params["client_opt_name"], opt_params["client_epoch"]
    import copy
    import math
    from utilities import vector_to_grads, vector_to_grads_sq
    from main import train
    shared_train_loader = (
        len(train_loaders) == 2
        and iter(train_loaders[0]) is train_loaders[0]
        and iter(train_loaders[1]) is not train_loaders[1]
    )

    adapter_names = []
    adapter_weights = {}
    output_weights = {}
    output_layer_name = opt_params["output_layer_name"]

    model.set_adapter(opt_params["server_name"])
    for name, param in model.named_parameters():
        # select lora_A and lora_B
        if param.requires_grad:
            if output_layer_name and output_layer_name in name:
                output_weights[name]= 0
            else:
                adapter_names.append(name)
                adapter_weights[name] = torch.zeros_like(param)
        #print(name, torch.norm(param).item())

    #print(adapter_weights.keys())
    #print(output_weights.keys())
    #print("==== initalize ends =====")
    #from utilities import state_dict_to_vector, vector_to_state_dict
    # initialize client models, optimizers
        
    #running_stats = {}
    client_opt_params = copy.deepcopy(opt_params)
    client_opt_params["train_stats"] = False
    for client_id in range(client_num):
        # update client models
        adapter_name = "client_{}".format(client_id)
        #client_model = copy.deepcopy(model)
        model.set_adapter(adapter_name)
        client_model = model #alias

        client_model.train()
        optimizer, lr_scheduler, _= load_optimizer(client_opt_name, client_model, client_lr, opt_params["client_momentum"], opt_params["client_weight_decay"], opt_params["lr_decay"], opt_params["epochs_lr_decay"], False, model_params, opt_params)
        #vector_to_parameters(old_params, client_model.parameters())
        for epoch in range(client_epoch):
            if shared_train_loader:
                try:
                    train_graphs.loader_iter += 1
                    train(client_model, loss_name, criterion, device, train_loaders[0], optimizer, lr_scheduler, server_epoch, client_opt_params)
                except StopIteration:
                    print("\nData Iterator is reloaded")
                    train_graphs.loader_iter += 1
                    train_loaders[0] = iter(train_loaders[1])
                    train(client_model, loss_name, criterion, device, train_loaders[0], optimizer, lr_scheduler, server_epoch, client_opt_params)
            else:
                train(client_model, loss_name, criterion, device, train_loaders[client_id], optimizer, lr_scheduler, server_epoch, client_opt_params)
            
        for name, param in client_model.named_parameters():
            if param.requires_grad:
                #param_names.append(name)
                server_adapter_name = name.replace("{}".format(adapter_name), opt_params["server_name"])
                if output_layer_name and output_layer_name in name:
                    output_weights[server_adapter_name] += param.data / client_num
                else:
                    if server_adapter_name in adapter_weights:
                        row, col = param.data.shape
                        adapter_weights[server_adapter_name][:row, :col] += param.data/client_num
                    else:
                        assert False
                """
                elif name in adapter_weights:
                    #lora_params[name].append(param.data)
                    if 'lora_A' in name:
                        #base_name = name.replace("lora_A.{}".format(adapter_name), "base_layer")
                        #adapter_weights[name] += param.data/client_num
                        server_adapter_name = name.replace("{}".format(adapter_name), "server")
                        adapter_weights[server_adapter_name] += param.data/client_num
                    elif 'lora_B' in name:
                        #base_name = name.replace("lora_B.{}".format(adapter_name), "base_layer")
                        #adapter_weights[name] += param.data/client_num
                        server_adapter_name = name.replace("{}".format(adapter_name), "server")
                        adapter_weights[server_adapter_name] += param.data/client_num
                    else: assert False
                else:
                    assert False
                """
    model.set_adapter(opt_params["server_name"])
    #truncate_err, truncate_err_ratio = compute_truncate_err(model, adapter_weights, client_num, opt_params["model_name"], opt_params["server_name"])
    server_optimizer.zero_grad()

    #train_graphs.truncate_err.append(truncate_err)
    #train_graphs.truncate_err_ratio.append(truncate_err_ratio)
    #print("Truncation Error: ", train_graphs.truncate_err[-1])
    #print("Truncation Error Ratio: ", train_graphs.truncate_err_ratio[-1])

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
            if param.requires_grad:
                if 'lora_A' in name:
                    base_name = name.replace("lora_A.{}".format(opt_params["server_name"]), "base_layer")
                    base_adapter_weights[base_name] = {}
                    base_adapter_weights[base_name]["A"] = param.data
                elif 'lora_B' in name:
                    base_name = name.replace("lora_B.{}".format(opt_params["server_name"]), "base_layer")
                    base_adapter_weights[base_name]["B"] = param.data
                
                    B_norm, A_norm = torch.norm(base_adapter_weights[base_name]["B"]), torch.norm(base_adapter_weights[base_name]["A"])
                    base_full = (base_adapter_weights[base_name]["B"] @ base_adapter_weights[base_name]["A"]).T
                
                    U, S, Vh = torch.linalg.svd(base_full, full_matrices=False)
                    U_truncate, S_truncate, Vh_truncate = U[:, :lora_rank], torch.sqrt(S[:lora_rank]), Vh[:lora_rank, :]
                    lora_A_name, lora_B_name = name.replace("lora_B.{}".format(opt_params["server_name"]), "lora_A.{}".format(opt_params["server_name"])), name

                    S_norm = torch.norm(S_truncate)
                    print("uba mode is " + opt_params["uba_mode"])
                    if opt_params["uba_mode"] == "ada":
                        print("B_norm", B_norm, "A_norm", A_norm, "S_norm", S_norm)
                        ratio = (A_norm + opt_params["uba_weight"] * opt_params["fedlora_uba"]**2*S_norm) / (B_norm + opt_params["uba_weight"] * S_norm)
                        ratio = ratio**0.5
                    else:
                        ratio = opt_params["fedlora_uba"]
                        
                    adapter_weights[lora_A_name].data = (U_truncate * S_truncate).T * ratio
                    adapter_weights[lora_B_name].data = Vh_truncate.T * S_truncate / ratio
            
        # assign new param to model
        for name, param in model.named_parameters():
            if name in adapter_weights:
                param.data = adapter_weights[name].data
    
    synchronize_lora(model, opt_params["server_name"], truncate_last=True)
    """
    for name, param in model.named_parameters():
        if name in adapter_weights or name in output_weights:
            adapter_weights[name] = param #store the server param
        elif 'client' in name:
            import re 
            server_adapter_name = re.sub(r'client_\d+', 'server', name)
            adapter_weight_full = adapter_weights[server_adapter_name].data.clone() #assign the same param to client models
            if len(param.data.shape) == 2:
                row, col = param.data.shape
                param.data = adapter_weight_full[:row, :col]
            elif len(param.data.shape) == 1:
                param.data = adapter_weight_full
            else:
                assert False
        
        #if 'lora_A' in name or 'lora_B' in name:
        #    import re
        #    server_adapter_name = re.sub(r'client_\d+', 'server', name)
        #    param.data = adapter_weights[server_adapter_name].data
        
    """
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
                        base_weights[base_name] +=  scaling * compute_adapter_weight(opt_params["model_name"], base_A_param, base_B_param) / client_num
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
                base_weights[base_name] -= scaling * compute_adapter_weight(opt_params["model_name"], base_A_param, base_B_param)
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
        adapter_name = "client_{}".format(client_id)
        #client_model = copy.deepcopy(model)
        model.set_adapter(adapter_name)
        client_model = model #alias
        # update client models
        #client_model = copy.deepcopy(model)

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
                    base_name = name.replace(adapter_name, opt_params["server_name"])
                    output_weights[base_name] += param.data / client_num
                else: #lora modules
                    #lora_params[name].append(param.data)
                    if 'lora_A' in name:
                        base_name = name.replace("lora_A.{}".format(adapter_name), "base_layer")
                        base_name_A, base_A_param = base_name, param.data
                    elif 'lora_B' in name:
                        base_name = name.replace("lora_B.{}".format(adapter_name), "base_layer")
                        assert base_name == base_name_A #ensure this module to be the sequel of A
                        base_B_param = param.data

                        #scaling = model_params["lora_alpha"] / model_params["lora_rank"]
                        # in the new implementation, lora_alpha is set to lora_rank in heteogeneous setting
                        if opt_params["hetero_rank"] == 1:
                            scaling = 1
                        elif opt_params["hetero_rank"] == -1: #homogeneous
                            scaling = model_params["lora_alpha"] / model_params["lora_rank"]
                        else:
                            raise NotImplementedError
                        original_base_weight_norm = torch.norm(base_weights[base_name])
                        base_weights[base_name] +=  scaling * compute_adapter_weight(opt_params["model_name"], base_A_param, base_B_param) / client_num
                        #print(base_name, torch.norm(base_weights[base_name]), original_base_weight_norm, torch.norm(base_weights[base_name] - untouch_base_weights[base_name]))
                        #print(base_weights[base_name])
                        #print(base_weights[base_name] - untouch_base_weights[base_name])
                    else: assert False
    
    model.set_adapter(opt_params["server_name"])
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
    #unload_model = model.unload()
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
    #from arch.lora import add_adapters_dataset
    #model , _, _ = add_adapters_dataset(opt_params["model_name"], unload_model, model_params["lora_rank"], model_params["lora_alpha"], lora_freeze_a=opt_params["lora_freeze_a"])

    #from arch.lora import add_adapters_hetero
    #model, _, _ = add_adapters_hetero(client_num, opt_params["model_name"], unload_model, model_params["lora_rank"], model_params["lora_alpha"], opt_params, lora_freeze_a=opt_params["lora_freeze_a"])
    

    print("reinitialize lora module")
    """
    for name, param in model.named_parameters():
        if 'lora' in name:
            print("==== LoRA params ====")
            print(name, torch.norm(param).item())
    
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora' in name:
            param.data = torch.randn_like(param.data)
    """
    from arch.lora import synchronize_lora
    synchronize_lora(model, opt_params["server_name"], truncate_last=True)
    """
    for name, param in model.named_parameters():
        if 'lora' in name:
            print("==== LoRA updated params ====")
            print(name, torch.norm(param).item())
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

        client_model_save = copy.deepcopy(client_model)

        # Download Sparsity
        if model_params["dl_density"] < 1:
            for n,p in client_model.named_parameters():
                if p.requires_grad:
                    if output_layer_name and output_layer_name in n:
                        pass
                    else:
                        p.data = p.data*server_mask[n]
        
        client_model_sparse_save = copy.deepcopy(client_model)

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
    return client_model_save, client_model_sparse_save


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

def _qr_transfer_from_B(B_weight, A_weight, gamma):
    """Recondition B while preserving the represented product B @ A."""
    if gamma == 0:
        raise ValueError("QR gauge transfer requires a nonzero lora_init_scale")
    Q_B, R_B = torch.linalg.qr(B_weight, mode="reduced")
    return gamma * Q_B, (R_B @ A_weight) / gamma


def _qr_transfer_from_A(B_weight, A_weight, gamma):
    """Recondition A.T while preserving the represented product B @ A."""
    if gamma == 0:
        raise ValueError("QR gauge transfer requires a nonzero lora_init_scale")
    Q_A, R_A = torch.linalg.qr(A_weight.T, mode="reduced")
    return (B_weight @ R_A.T) / gamma, gamma * Q_A.T


def _projector_average_subspace(current_basis, target_basis, beta):
    """Return the leading subspace of a weighted average of two projectors.

    The computation stays low rank: the leading left singular vectors of
    [sqrt(beta) * Q_current, sqrt(1-beta) * Q_target] are the leading
    eigenvectors of beta * Q_current Q_current.T +
    (1-beta) * Q_target Q_target.T.
    """
    if not 0 <= beta < 1:
        raise ValueError("probe tracking beta must be in [0, 1)")
    if current_basis.shape != target_basis.shape:
        raise ValueError(
            f"probe subspace shapes must match, got {current_basis.shape} and {target_basis.shape}"
        )

    current_basis = torch.linalg.qr(current_basis, mode="reduced")[0]
    target_basis = torch.linalg.qr(target_basis, mode="reduced")[0]
    combined = torch.cat(
        (
            math.sqrt(beta) * current_basis,
            math.sqrt(1.0 - beta) * target_basis,
        ),
        dim=1,
    )
    Q_combined, R_combined = torch.linalg.qr(combined, mode="reduced")
    U_core = torch.linalg.svd(R_combined, full_matrices=False)[0]
    rank = current_basis.shape[1]
    return Q_combined @ U_core[:, :rank]


def _align_basis_for_interpolation(reference_basis, candidate_basis):
    """Resolve basis-sign/rotation ambiguity before a Euclidean retraction."""
    U, _, Vh = torch.linalg.svd(
        candidate_basis.T @ reference_basis, full_matrices=False
    )
    return candidate_basis @ (U @ Vh)


def _interpolate_orthonormal_basis(reference_basis, candidate_basis, fraction):
    """Interpolate two subspaces and retract the result with reduced QR."""
    if not 0 <= fraction <= 1:
        raise ValueError("subspace interpolation fraction must be in [0, 1]")
    candidate_basis = _align_basis_for_interpolation(
        reference_basis, candidate_basis
    )
    blended = (1.0 - fraction) * reference_basis + fraction * candidate_basis
    return torch.linalg.qr(blended, mode="reduced")[0]


def _low_rank_product_spectral_norm(left, right):
    """Compute ||left @ right||_2 through a small core SVD."""
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            f"incompatible low-rank factors {left.shape} and {right.shape}"
        )
    Q_left, R_left = torch.linalg.qr(left, mode="reduced")
    Q_right, R_right = torch.linalg.qr(right.T, mode="reduced")
    return torch.linalg.svdvals(R_left @ R_right.T)[0]


def _probe_product_difference_spectral_norm(B_old, A_old, B_new, A_new):
    """Compute ||B_old A_old - B_new A_new||_2 without a dense product."""
    left = torch.cat((B_old, B_new), dim=1)
    right = torch.cat((A_old, -A_new), dim=0)
    return _low_rank_product_spectral_norm(left, right)


def _balanced_probe_tracking_step(
    B_old,
    A_old,
    target_B,
    target_A,
    gamma,
    beta,
    scaling,
    desired_update_norm,
    max_correction_ratio,
):
    """Track both Muon subspaces with balanced probes and a correction cap.

    B_old is (m, r), A_old is (r, n), target_B is (m, r), and target_A is
    (n, r). The returned probes satisfy B.T B = A A.T = gamma^2 I up to
    numerical error. If necessary, their joint rotation is shortened so
    that the base-weight compensation is bounded relative to the intended
    effective Muon update.
    """
    if gamma <= 0:
        raise ValueError("muonlora_v18 requires a positive lora_init_scale")
    if max_correction_ratio <= 0:
        raise ValueError(
            "muonlora_v18 requires a positive "
            "muonlora_max_correction_ratio"
        )

    Q_B_old = torch.linalg.qr(B_old, mode="reduced")[0]
    Q_A_old = torch.linalg.qr(A_old.T, mode="reduced")[0]
    Q_B_candidate = _projector_average_subspace(Q_B_old, target_B, beta)
    Q_A_candidate = _projector_average_subspace(Q_A_old, target_A, beta)

    max_correction_norm = max_correction_ratio * desired_update_norm
    fraction = 1.0
    B_new = gamma * Q_B_candidate
    A_new = gamma * Q_A_candidate.T
    correction_norm = scaling * _probe_product_difference_spectral_norm(
        B_old, A_old, B_new, A_new
    )

    # Back off jointly on both sides if the probe reparameterization would
    # require an excessively large low-precision base correction.
    for _ in range(8):
        if correction_norm <= max_correction_norm:
            break
        fraction *= 0.5
        Q_B_new = _interpolate_orthonormal_basis(
            Q_B_old, Q_B_candidate, fraction
        )
        Q_A_new = _interpolate_orthonormal_basis(
            Q_A_old, Q_A_candidate, fraction
        )
        B_new = gamma * Q_B_new
        A_new = gamma * Q_A_new.T
        correction_norm = scaling * _probe_product_difference_spectral_norm(
            B_old, A_old, B_new, A_new
        )

    return B_new, A_new, correction_norm, fraction


def _damped_pinv_from_svd(U, S, Vh, relative_damping=1e-3):
    """Return a smooth Tikhonov pseudoinverse from a matrix's reduced SVD."""
    if relative_damping <= 0:
        raise ValueError("relative_damping must be positive")
    damping = (relative_damping * S.max()).clamp_min(torch.finfo(S.dtype).eps)
    damped_reciprocal = S / (S.square() + damping.square())
    return (Vh.T * damped_reciprocal) @ U.T, damping


def _truncate_low_rank_product(left, right, max_rank, storage_dtype=torch.float32):
    """Compress left @ right without materializing the full matrix."""
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    Q_left, R_left = torch.linalg.qr(left, mode="reduced")
    Q_right, R_right = torch.linalg.qr(right.T, mode="reduced")
    U, S, Vh = torch.linalg.svd(R_left @ R_right.T, full_matrices=False)
    rank = min(max_rank, S.numel())
    sqrt_S = S[:rank].clamp_min(0).sqrt()
    compressed_left = (Q_left @ U[:, :rank]) * sqrt_S.unsqueeze(0)
    compressed_right = sqrt_S.unsqueeze(1) * (Vh[:rank] @ Q_right.T)
    return compressed_left.to(storage_dtype), compressed_right.to(storage_dtype)


def _transport_B_momentum_with_error_feedback(
    old_momentum, A_prev, A_cur, beta, error, max_rank
):
    """Transport B momentum through A's row space and retain a rank-capped residual."""
    prev_lift = torch.linalg.pinv(A_prev @ A_prev.T, rcond=1e-6) @ A_prev
    left_parts = [beta * old_momentum]
    right_parts = [prev_lift]
    if error is not None:
        error_left, error_right = error
        left_parts.append(beta * error_left.to(old_momentum))
        right_parts.append(error_right.to(old_momentum))
    lifted_left = torch.cat(left_parts, dim=1)
    lifted_right = torch.cat(right_parts, dim=0)

    aligned_momentum = lifted_left @ (lifted_right @ A_cur.T)
    cur_lift = torch.linalg.pinv(A_cur @ A_cur.T, rcond=1e-6) @ A_cur
    residual_left = torch.cat((lifted_left, -aligned_momentum), dim=1)
    residual_right = torch.cat((lifted_right, cur_lift), dim=0)
    new_error = _truncate_low_rank_product(
        residual_left, residual_right, max_rank=max_rank
    )
    return aligned_momentum, new_error


def _transport_A_momentum_with_error_feedback(
    old_momentum, B_prev, B_cur, beta, error, max_rank
):
    """Transport A momentum through B's column space and retain a rank-capped residual."""
    prev_lift = B_prev @ torch.linalg.pinv(B_prev.T @ B_prev, rcond=1e-6)
    left_parts = [beta * prev_lift]
    right_parts = [old_momentum]
    if error is not None:
        error_left, error_right = error
        left_parts.append(beta * error_left.to(old_momentum))
        right_parts.append(error_right.to(old_momentum))
    lifted_left = torch.cat(left_parts, dim=1)
    lifted_right = torch.cat(right_parts, dim=0)

    aligned_momentum = (B_cur.T @ lifted_left) @ lifted_right
    cur_lift = B_cur @ torch.linalg.pinv(B_cur.T @ B_cur, rcond=1e-6)
    residual_left = torch.cat((lifted_left, -cur_lift), dim=1)
    residual_right = torch.cat((lifted_right, aligned_momentum), dim=0)
    new_error = _truncate_low_rank_product(
        residual_left, residual_right, max_rank=max_rank
    )
    return aligned_momentum, new_error


def _low_rank_concat(parts):
    """Concatenate (left, right) factor pairs into a single pair without expanding."""
    return (
        torch.cat([left for left, _ in parts], dim=1),
        torch.cat([right for _, right in parts], dim=0),
    )


def _low_rank_fro_norm(left, right):
    """Frobenius norm of left @ right without forming the product."""
    R = torch.linalg.qr(left, mode="reduced")[1]
    return (R @ right).norm()


def _shared_ambient_ef_state(opt_params, key):
    """Fetch (or lazily create) one module's shared alignment-error state."""
    if "shared_ambient_ef" not in opt_params:
        opt_params["shared_ambient_ef"] = {}
    return opt_params["shared_ambient_ef"].setdefault(key, {})


def _ambient_momentum_norm(M_B, M_A, B):
    """||M_B (B^T M_B)^+ M_A||_F, the implied ambient momentum magnitude.

    Computed through an r x r QR so the (m, n) product is never formed.
    """
    X = M_B @ torch.linalg.pinv(B.T @ M_B, rcond=1e-6)      # (m, r)
    R = torch.linalg.qr(X, mode="reduced")[1]                # (r, r)
    return (R @ M_A).norm()


def _shared_alignment_error_step(
    B,
    A_w,
    M_B,
    M_A,
    drop,
    state,
    dense_buffer,
    max_rank,
    error_decay,
    debt_cap,
    storage_dtype=torch.float32,
):
    """Accumulate this round's momentum-alignment loss in one shared ambient buffer
    and hand back what each factor can absorb.

    ``drop`` is the ambient (m, n) mass that this round's momentum transport failed
    to carry, passed as a low-rank (left, right) pair. ``M_B`` (m, r) and ``M_A``
    (r, n) are the per-factor momenta the caller already maintains -- the ambient
    momentum is never instantiated, it stays implicit in those two and in the
    existing pseudo-gradient reconstruction M_B (B^T M_B)^+ M_A used downstream.

    One buffer serves both factors, and both absorb from it every round. What binds
    is the pseudo-gradient's reach: its column space lies in col(M_B) and its row
    space in row(M_A), so M_B carries the column side of the debt and M_A the row
    side. Using the same convention that defines the drop -- the ambient image of
    M_A is B M_A and of M_B is M_B A -- the two absorbed pieces are B^+ E and E A^+,
    and what neither can express is (I - P_B) E (I - Pi_A), the doubly-orthogonal
    block, which is reachable only once the frames rotate.

    Returns (delta_B, delta_A, stats), to be added to M_B and M_A respectively.
    """
    BtB_inv = torch.linalg.pinv(B.T @ B, rcond=1e-6)
    AAt_inv = torch.linalg.pinv(A_w @ A_w.T, rcond=1e-6)

    if dense_buffer:
        drop_full = drop[0] @ drop[1]
        error = state.get("error")
        E = drop_full if error is None else error_decay * error.to(drop_full) + drop_full

        delta_A = BtB_inv @ (B.T @ E)          # (r, n): ambient B delta_A = P_B E
        delta_B = (E @ A_w.T) @ AAt_inv        # (m, r): ambient delta_B A = E Pi_A
        overlap = BtB_inv @ (B.T @ E @ A_w.T) @ AAt_inv     # (r, r)
        # inclusion-exclusion leaves exactly (I - P_B) E (I - Pi_A)
        E = E - B @ delta_A - delta_B @ A_w + B @ overlap @ A_w

        reference = _ambient_momentum_norm(M_B, M_A, B)
        error_norm = E.norm()
        clipped = False
        if debt_cap > 0 and error_norm > debt_cap * reference and error_norm > 0:
            E = E * (debt_cap * reference / error_norm)
            error_norm = debt_cap * reference
            clipped = True
        state["error"] = E.to(storage_dtype)
    else:
        # Same recursion with the buffer kept rank-capped instead of dense.
        error = state.get("error")
        parts = [drop]
        if error is not None:
            parts.insert(0, (error_decay * error[0].to(B), error[1].to(B)))
        E_left, E_right = _low_rank_concat(parts)

        delta_A = BtB_inv @ ((B.T @ E_left) @ E_right)
        delta_B = (E_left @ (E_right @ A_w.T)) @ AAt_inv
        overlap = BtB_inv @ ((B.T @ E_left) @ (E_right @ A_w.T)) @ AAt_inv
        E_left, E_right = _truncate_low_rank_product(
            *_low_rank_concat([
                (E_left, E_right),
                (-B, delta_A),
                (-delta_B, A_w),
                (B @ overlap, A_w),
            ]),
            max_rank=max_rank,
        )

        reference = _ambient_momentum_norm(M_B, M_A, B)
        error_norm = _low_rank_fro_norm(E_left.to(B), E_right.to(B))
        clipped = False
        if debt_cap > 0 and error_norm > debt_cap * reference and error_norm > 0:
            E_left = E_left * (debt_cap * reference / error_norm).to(E_left)
            error_norm = debt_cap * reference
            clipped = True
        state["error"] = (E_left.to(storage_dtype), E_right.to(storage_dtype))

    stats = {
        "error_norm": float(error_norm),
        "reference_norm": float(reference),
        "debt_ratio": float(error_norm / reference) if float(reference) > 0 else 0.0,
        "clipped": clipped,
    }
    return delta_B, delta_A, stats


def _factor_alignment_ef_state(opt_params, key):
    """Fetch one factor-owned ambient error buffer."""
    if "factor_ambient_ef" not in opt_params:
        opt_params["factor_ambient_ef"] = {}
    return opt_params["factor_ambient_ef"].setdefault(key, {})


def _factor_alignment_error_step(
    B,
    A_w,
    target_momentum,
    drop,
    state,
    target_factor,
    error_decay,
):
    """Redeem an alignment loss only through the momentum that lost it.

    v20 keeps the debt dense and in FP64. An A-momentum debt is defined in the
    lift B @ M_A and is redeemed solely with delta_A = B^+ @ E. A
    B-momentum debt is symmetric, using delta_B = E @ A^+.
    """
    if target_factor not in ("A", "B"):
        raise ValueError(f"unknown factor target {target_factor!r}")

    drop_full = drop[0] @ drop[1]
    old_error = state.get("error")
    E = drop_full if old_error is None else error_decay * old_error + drop_full

    if target_factor == "A":
        delta = torch.linalg.pinv(B.T @ B, rcond=1e-6) @ (B.T @ E)
        E = E - B @ delta
        reference = _low_rank_fro_norm(B, target_momentum)
    else:
        delta = (E @ A_w.T) @ torch.linalg.pinv(A_w @ A_w.T, rcond=1e-6)
        E = E - delta @ A_w
        reference = _low_rank_fro_norm(target_momentum, A_w)

    # E is already float64: all drop factors and factor momenta are promoted
    # before this helper is called. Retaining it at that precision is required
    # for the dense error-feedback identity to persist across rounds.
    state["error"] = E
    error_norm = E.norm()
    stats = {
        "error_norm": float(error_norm),
        "reference_norm": float(reference),
        "debt_ratio": float(error_norm / reference) if float(reference) > 0 else 0.0,
        "clipped": False,
    }
    return delta, stats


def _advance_alternating_phase(opt_params, server_epoch, alternate_update):
    """Pick which LoRA factor receives this round's update."""
    if alternate_update == False:
        print("Keep update on A...")
        opt_params["update_B"] = False
        return
    assert opt_params["muonlora_switch_interval"] > 0
    if (server_epoch - 1) % opt_params["muonlora_switch_interval"] == 0:
        if "update_B" in opt_params:
            # switch update parameter
            opt_params["update_B"] = not opt_params["update_B"]
            print("Switching update to {}...".format("B" if opt_params["update_B"] else "A"))
        else:
            #lazy initialize update_B
            print("Init update on B...")
            opt_params["update_B"] = True


def _projected_muon_factor_split(B, A, U, V, merge_alpha, update_B,
                                 use_scaled_identity, gamma):
    """Split U @ V into a one-sided adapter move and a rank-r residual.

    A is the stored (r, n) weight. U @ V already includes -lr / s and
    optional shape scaling. Never materialize this dense product here.
    The shortcut assumes A A.T = B.T B = gamma**2 I (up to storage rounding);
    disable it to use the actual partner Gram matrix's pseudoinverse.
    """
    if use_scaled_identity and (not math.isfinite(gamma) or gamma <= 0):
        raise ValueError("projected Muon scaled-identity inverse needs lora_init_scale > 0")
    if update_B:
        coordinates = V @ A.T
        if use_scaled_identity:
            coordinates = coordinates / gamma**2
        else:
            coordinates = coordinates @ torch.linalg.pinv(A @ A.T, rcond=1e-6)
        B_new = B + merge_alpha * (U @ coordinates)
        # U V - (B_new - B) A = U (V - alpha * coordinates * A).
        return B_new, U, V - merge_alpha * (coordinates @ A)

    coordinates = B.T @ U
    if use_scaled_identity:
        coordinates = coordinates / gamma**2
    else:
        coordinates = torch.linalg.pinv(B.T @ B, rcond=1e-6) @ coordinates
    A_new = A + merge_alpha * (coordinates @ V)
    # U V - B (A_new - A) = (U - alpha * B * coordinates) V.
    return A_new, U - merge_alpha * (B @ coordinates), V


def get_muonlora_hparams(fedlora_avg_name):
    if fedlora_avg_name == 'muonlora_v1':
        use_model_grad, use_rtol_inv, use_norm_grad, apply_momentum, moment_on_factor = False, True, False, False, True
    elif fedlora_avg_name == 'muonlora_v2':
        use_model_grad, use_rtol_inv, use_norm_grad, apply_momentum, moment_on_factor = False, False, True, False, True
    elif fedlora_avg_name == 'muonlora_v3':
        use_model_grad, use_rtol_inv, use_norm_grad, apply_momentum, moment_on_factor = False, False, False, False, True
    elif fedlora_avg_name == 'muonlora_v4':
        use_model_grad, use_rtol_inv, use_norm_grad, apply_momentum, moment_on_factor = True, False, True, False, True
    elif fedlora_avg_name == 'muonlora_v5':
        use_model_grad, use_rtol_inv, use_norm_grad, apply_momentum, moment_on_factor = True, False, False, True, True
    elif fedlora_avg_name == 'muonlora_v6':
        use_model_grad, use_rtol_inv, use_norm_grad, apply_momentum, moment_on_factor = True, True, False, True, True
    elif fedlora_avg_name == 'muonlora_v7':
        use_model_grad, use_rtol_inv, use_norm_grad, apply_momentum, moment_on_factor = True, False, False, True, True
    elif fedlora_avg_name == 'muonlora_v8':
        use_model_grad, use_rtol_inv, use_norm_grad, apply_momentum, moment_on_factor = True, True, False, True, True
    elif fedlora_avg_name == 'muonlora_v9':
        """directly adding momentum to the lora adapters."""
        use_model_grad, use_rtol_inv, use_norm_grad, apply_momentum, moment_on_factor = True, False, True, True, True
    elif fedlora_avg_name in ['muonlora_v10', 'muonlora_v11', 'muonlora_v12', 'muonlora_v13', 'muonlora_v14', 'muonlora_v15', 'muonlora_v16', 'muonlora_v17', 'muonlora_v18', 'muonlora_v19', 'muonlora_v20', 'muonlora_v23']:
        """split muon update: fuse singular-vector-aligned part into server adapter, keep residual as muon update; alternates A/B sides across epochs."""
        use_model_grad, use_rtol_inv, use_norm_grad, apply_momentum, moment_on_factor = True, False, True, False, True
    else:
        raise NotImplementedError

    if fedlora_avg_name == 'muonlora_v10':
        partial_merge, orth_then_merge, alternate_update, aligned_momentum = True, True, True, False
    elif fedlora_avg_name == 'muonlora_v11':
        # do not do QR to A or B before the update
        partial_merge, orth_then_merge, alternate_update, aligned_momentum = True, False, True, False
    elif fedlora_avg_name == 'muonlora_v12':
        #alternate_update == False ==> only update on A
        #alternate_update == True + large switch_interval ==> only update on B
        partial_merge, orth_then_merge, alternate_update, aligned_momentum = True, False, False, False
    elif fedlora_avg_name == 'muonlora_v13':
        """principled momentum aggregation: project old momentum onto new factor's column/row space before accumulating."""
        partial_merge, orth_then_merge, alternate_update, aligned_momentum = True, False, True, True
    elif fedlora_avg_name == 'muonlora_v14':
        """v13 + re-orthonormalize the updated LoRA factor after each partial merge (Sec 3.7)."""
        partial_merge, orth_then_merge, alternate_update, aligned_momentum = True, False, True, True
    elif fedlora_avg_name == 'muonlora_v15':
        """v14-style reconditioning with a product-preserving QR gauge transfer."""
        partial_merge, orth_then_merge, alternate_update, aligned_momentum = True, False, True, True
    elif fedlora_avg_name == 'muonlora_v16':
        """v15 + rank-capped effective-weight-space momentum error feedback."""
        partial_merge, orth_then_merge, alternate_update, aligned_momentum = True, False, True, True
    elif fedlora_avg_name == 'muonlora_v17':
        """v16 with v14-style base-weight orthonormalization correction in place of
        the product-preserving QR gauge transfer."""
        partial_merge, orth_then_merge, alternate_update, aligned_momentum = True, False, True, True
    elif fedlora_avg_name == 'muonlora_v18':
        """v16 + simultaneous balanced probe tracking with exact base compensation."""
        partial_merge, orth_then_merge, alternate_update, aligned_momentum = True, False, True, True
    elif fedlora_avg_name == 'muonlora_v19':
        """v14 exactly, plus one shared ambient buffer that accumulates the momentum
        alignment loss so the other factor's phase can absorb it."""
        partial_merge, orth_then_merge, alternate_update, aligned_momentum = True, False, True, True
    elif fedlora_avg_name in ('muonlora_v20', 'muonlora_v23'):
        """v20: factor-owned EF; v23: same EF with projected Muon factor steps."""
        partial_merge, orth_then_merge, alternate_update, aligned_momentum = True, False, True, True
    else:
        partial_merge, orth_then_merge, alternate_update, aligned_momentum = False, False, False, False
    use_damped_inv = False #fedlora_avg_name in ['muonlora_v15', 'muonlora_v16']
    use_momentum_error_feedback = fedlora_avg_name in ['muonlora_v16', 'muonlora_v17', 'muonlora_v18']
    use_product_preserving_qr_gauge = fedlora_avg_name in ['muonlora_v15', 'muonlora_v16']
    # Toggle this off to make v18 use the legacy alternating merge path.
    use_balanced_probe_tracking = fedlora_avg_name == 'muonlora_v18'
    use_shared_ambient_ef = fedlora_avg_name == 'muonlora_v19'
    use_factor_ambient_ef = fedlora_avg_name in ('muonlora_v20', 'muonlora_v23')
    use_dense_ef_buffer = fedlora_avg_name in ['muonlora_v19', 'muonlora_v20', 'muonlora_v23']
    # v23-only design toggles, configured here rather than on the CLI.
    # Disable the projected step to recover v20's singular-vector factor move.
    use_projected_muon_factor_update = fedlora_avg_name == 'muonlora_v23'
    # Uniform-SV initialization and per-round QR give Gram = gamma**2 I.
    # False uses the measured Gram pseudoinverse instead. This shortcut applies
    # ONLY to the factor step, not to momentum transport or error feedback.
    projected_muon_scaled_identity = True
    return use_model_grad, use_rtol_inv, use_norm_grad, apply_momentum, moment_on_factor, partial_merge, \
            orth_then_merge, alternate_update, aligned_momentum, use_damped_inv, use_momentum_error_feedback, \
            use_balanced_probe_tracking, use_product_preserving_qr_gauge, use_shared_ambient_ef, \
            use_dense_ef_buffer, use_factor_ambient_ef, \
            use_projected_muon_factor_update, projected_muon_scaled_identity

def federated_muonlora(model, loss_name, criterion, lora_rank, train_graphs, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, opt_params, model_params, server_epoch):
    client_num, client_opt_name, client_epoch = opt_params["client_num"], opt_params["client_opt_name"], opt_params["client_epoch"]
    import copy
    from main import train

    from utilities import get_gpu_memory

    use_model_grad, use_rtol_inv, use_norm_grad, apply_momentum, moment_on_factor, partial_merge, \
        orth_then_merge, alternate_update, aligned_momentum, use_damped_inv, \
        use_momentum_error_feedback, use_balanced_probe_tracking, \
        use_product_preserving_qr_gauge, use_shared_ambient_ef, \
        use_dense_ef_buffer, use_factor_ambient_ef, \
        use_projected_muon_factor_update, projected_muon_scaled_identity = get_muonlora_hparams(
            fedlora_avg_name=opt_params["fedlora_avg"])
    # v14-style re-orthonormalization of the updated factor after each merge.
    v14_style_reorth = opt_params["fedlora_avg"] in (
        "muonlora_v14", "muonlora_v17", "muonlora_v19", "muonlora_v20", "muonlora_v23")
    if opt_params["fedlora_avg"] == "muonlora_v23":
        gamma = float(opt_params.get("lora_init_scale", -1))
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("muonlora_v23 requires lora_init_scale > 0 for scaled-orthonormal factors")
        print(f"[muonlora_v23] projected_factor_update={use_projected_muon_factor_update}, "
              f"projected_scaled_identity={projected_muon_scaled_identity}")
    if use_model_grad:
        opt_params["local_update_ON"] = False
    else:
        opt_params["local_update_ON"] = True

    print(f"[riemannion] use_model_grad={use_model_grad}, use_rtol_inv={use_rtol_inv}, use_norm_grad={use_norm_grad}, "
        f"apply_momentum={apply_momentum}, moment_on_factor={moment_on_factor}, partial_merge={partial_merge}, "
        f"orth_then_merge={orth_then_merge}, alternate_update={alternate_update}, "
        f"aligned_momentum={aligned_momentum}, use_damped_inv={use_damped_inv}, "
        f"use_momentum_error_feedback={use_momentum_error_feedback}, "
        f"use_shared_ambient_ef={use_shared_ambient_ef}, "
        f"use_factor_ambient_ef={use_factor_ambient_ef}, "
        f"use_dense_ef_buffer={use_dense_ef_buffer}")

    adapter_names = []
    adapter_weights = {}
    output_weights = {}
    original_params_data = {}
    output_layer_name = opt_params["output_layer_name"]

    model.set_adapter(opt_params["server_name"])
    for name, param in model.named_parameters():
        if param.requires_grad:
            if output_layer_name and output_layer_name in name:
                output_weights[name]= 0
            else:
                adapter_names.append(name)
                adapter_weights[name] = torch.zeros_like(param)
                original_params_data[name] = param.data.clone()
                
    client_opt_params = copy.deepcopy(opt_params)
    client_opt_params["train_stats"] = False

    if opt_params["client_partial"] < 1:
        client_num = int(opt_params["client_partial"] * client_num)
        client_selected = np.random.choice(opt_params["client_num"], client_num, replace=False)
    else:
        client_selected = np.arange(client_num)

    training_time_accumulated = 0
    for client_id in client_selected:
        # update client models
        adapter_name = "client_{}".format(client_id)
        if opt_params["local_update_ON"]:
            model.set_adapter(adapter_name)
        client_model = model #alias

        client_model.train()
        optimizer, lr_scheduler, _= load_optimizer(client_opt_name, client_model, client_lr, opt_params["client_momentum"], opt_params["client_weight_decay"], opt_params["lr_decay"], opt_params["epochs_lr_decay"], False, model_params, opt_params)
        #vector_to_parameters(old_params, client_model.parameters())
        
        print("="*10, " 1 ", "="*10)
        get_gpu_memory()
        for epoch in range(client_epoch):
            try:
                train_graphs.loader_iter += 1
                assert iter(train_loaders[0]) == train_loaders[0]
                import time
                start_time = time.time()
                _, model_grad = train(client_model, loss_name, criterion, device, train_loaders[0], optimizer, lr_scheduler, server_epoch, client_opt_params)
                end_time = time.time()
                training_time_accumulated += end_time - start_time
                print(f"Time taken for client {client_id}: {end_time - start_time} seconds")
            except StopIteration:
                # reinitialize iterator
                print("\nData Iterator is reloaded")
                train_graphs.loader_iter += 1
                train_loaders[0] = iter(train_loaders[1])
                _, model_grad = train(client_model, loss_name, criterion, device, train_loaders[0], optimizer, lr_scheduler, server_epoch, client_opt_params)

        print("="*10, " 2 ", "="*10)
        get_gpu_memory()
        
        for name, param in client_model.named_parameters():
            if param.requires_grad:
                #param_names.append(name)
                server_adapter_name = name.replace("{}".format(adapter_name), opt_params["server_name"])
                if output_layer_name and output_layer_name in name:
                    if use_model_grad:
                        output_weights[server_adapter_name] += model_grad[name]
                    else:
                        output_weights[server_adapter_name] += param.data #/ client_num
                    
                else:
                    if server_adapter_name in adapter_weights:
                        row, col = param.data.shape
                        #
                        if use_model_grad:
                            adapter_weights[server_adapter_name][:row, :col] += model_grad[name]
                        else:
                            adapter_weights[server_adapter_name][:row, :col] += param.data #/client_num
                        #print("client norm change: ", param.data.norm().item(), original_params_data[server_adapter_name].norm().item(),
                        #      (original_params_data[server_adapter_name]-param).norm().item())
                    else:
                        assert False

    print(f"Total training time: {training_time_accumulated} seconds")
    print("="*10, " 3 ", "="*10)
    get_gpu_memory()

    # average step is after the summation -- to provide more precisions
    for server_adapter_name in output_weights:
        output_weights[server_adapter_name] = output_weights[server_adapter_name] / client_num

    for server_adapter_name in adapter_weights:
        adapter_weights[server_adapter_name] = adapter_weights[server_adapter_name] / client_num

    if opt_params["local_update_ON"]:
        model.set_adapter(opt_params["server_name"])
    #truncate_err, truncate_err_ratio = compute_truncate_err(model, adapter_weights, client_num, opt_params["model_name"], opt_params["server_name"])
    server_optimizer.zero_grad()

    #train_graphs.truncate_err.append(truncate_err)
    #train_graphs.truncate_err_ratio.append(truncate_err_ratio)
    #print("Truncation Error: ", train_graphs.truncate_err[-1])
    #print("Truncation Error Ratio: ", train_graphs.truncate_err_ratio[-1])

    if opt_params["train_stats"]:
        grad_norm = 0

    params_grads = {}
    alignment_drops = {}
    factor_alignment_drops = {}
    #print("adapter diff norm")
    cur_adapter_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            cur_adapter_weights[name] = param.data

    for name, param in model.named_parameters():
        if param.requires_grad:
            if output_layer_name and output_layer_name in name:
                if use_model_grad:
                    param.grad = output_weights[name]
                else:
                    param.grad = param.data - output_weights[name]
            elif name in adapter_weights:
                if use_model_grad:
                    param.grad = adapter_weights[name]
                else:
                    param.grad = param.data - adapter_weights[name]
                if moment_on_factor:
                    if "momentum" not in opt_params:
                        opt_params["momentum"] = {}
                    if aligned_momentum and "prev_factor" not in opt_params:
                        opt_params["prev_factor"] = {}
                    if use_momentum_error_feedback and "momentum_alignment_error" not in opt_params:
                        opt_params["momentum_alignment_error"] = {}
                    if name in opt_params["momentum"]:
                        if aligned_momentum:
                            old_mom = opt_params["momentum"][name].to(torch.float64)
                            new_grad = param.grad.to(torch.float64)
                            if use_momentum_error_feedback:
                                if 'lora_B' in name:
                                    if not opt_params["update_B"]:
                                        error = opt_params["momentum_alignment_error"].get(name)
                                        prev_factor_name = name.replace("lora_B", "lora_A")
                                        A_prev = opt_params["prev_factor"][prev_factor_name].to(torch.float64)
                                        A_cur = cur_adapter_weights[prev_factor_name].to(torch.float64)
                                        aligned_mom, new_error = _transport_B_momentum_with_error_feedback(
                                            old_mom,
                                            A_prev,
                                            A_cur,
                                            opt_params["server_momentum"],
                                            error,
                                            opt_params["lora_rank"],
                                        )
                                        opt_params["momentum_alignment_error"][name] = new_error
                                        params_grads[name] = aligned_mom + new_grad
                                    else:
                                        # This factor is the one being updated, so no transport
                                        # runs. The retained error is a momentum-like quantity
                                        # (the transport scales it by beta each round), so keep
                                        # decaying it here or it would be re-injected at full
                                        # magnitude once this factor is frozen again.
                                        error = opt_params["momentum_alignment_error"].get(name)
                                        if error is not None:
                                            opt_params["momentum_alignment_error"][name] = (
                                                opt_params["server_momentum"] * error[0],
                                                error[1],
                                            )
                                        params_grads[name] = opt_params["server_momentum"] * old_mom + new_grad
                                else:
                                    assert 'lora_A' in name
                                    if opt_params["update_B"]:
                                        error = opt_params["momentum_alignment_error"].get(name)
                                        prev_factor_name = name.replace("lora_A", "lora_B")
                                        B_prev = opt_params["prev_factor"][prev_factor_name].to(torch.float64)
                                        B_cur = cur_adapter_weights[prev_factor_name].to(torch.float64)
                                        aligned_mom, new_error = _transport_A_momentum_with_error_feedback(
                                            old_mom,
                                            B_prev,
                                            B_cur,
                                            opt_params["server_momentum"],
                                            error,
                                            opt_params["lora_rank"],
                                        )
                                        opt_params["momentum_alignment_error"][name] = new_error
                                        params_grads[name] = aligned_mom + new_grad
                                    else:
                                        # This factor is the one being updated, so no transport
                                        # runs. The retained error is a momentum-like quantity
                                        # (the transport scales it by beta each round), so keep
                                        # decaying it here or it would be re-injected at full
                                        # magnitude once this factor is frozen again.
                                        error = opt_params["momentum_alignment_error"].get(name)
                                        if error is not None:
                                            opt_params["momentum_alignment_error"][name] = (
                                                opt_params["server_momentum"] * error[0],
                                                error[1],
                                            )
                                        params_grads[name] = opt_params["server_momentum"] * old_mom + new_grad
                            else:
                                if 'lora_B' in name:
                                    if not opt_params["update_B"] or use_product_preserving_qr_gauge:
                                        # B is (m, r): right-multiply old momentum by (r×r) change-of-basis
                                        prev_factor_name = name.replace("lora_B", "lora_A")
                                        prev_factor = opt_params["prev_factor"][prev_factor_name].to(torch.float64)
                                        core = prev_factor @ prev_factor.T
                                        C = torch.linalg.pinv(core, rcond=1e-6) @ prev_factor @ cur_adapter_weights[prev_factor_name].T.to(torch.float64)
                                        aligned_mom = old_mom @ C
                                        if use_shared_ambient_ef or use_factor_ambient_ef:
                                            # beta * (M_B A_prev - aligned A_cur): the B-momentum
                                            # mass this A-frame transport could not carry.
                                            beta_ = opt_params["server_momentum"]
                                            drop = (
                                                beta_ * torch.cat((old_mom, -aligned_mom), dim=1),
                                                torch.cat((prev_factor,
                                                           cur_adapter_weights[prev_factor_name].to(torch.float64)), dim=0),
                                            )
                                            if use_shared_ambient_ef:
                                                alignment_drops[name] = drop
                                            if use_factor_ambient_ef:
                                                factor_alignment_drops[name] = drop
                                    else:
                                        aligned_mom = old_mom
                                else:
                                    assert 'lora_A' in name
                                    if opt_params["update_B"] or use_product_preserving_qr_gauge:
                                        # A is (r, n): left-multiply old momentum by (r×r) change-of-basis
                                        prev_factor_name = name.replace("lora_A", "lora_B")
                                        prev_factor = opt_params["prev_factor"][prev_factor_name].to(torch.float64)
                                        core = prev_factor.T @ prev_factor
                                        C = cur_adapter_weights[prev_factor_name].to(torch.float64).T @ prev_factor @ torch.linalg.pinv(core, rcond=1e-6)
                                        aligned_mom = C @ old_mom
                                        if use_shared_ambient_ef or use_factor_ambient_ef:
                                            # beta * (B_prev M_A - B_cur aligned): the A-momentum
                                            # mass this B-frame transport could not carry.
                                            beta_ = opt_params["server_momentum"]
                                            drop = (
                                                beta_ * torch.cat(
                                                    (prev_factor,
                                                     -cur_adapter_weights[prev_factor_name].to(torch.float64)), dim=1),
                                                torch.cat((old_mom, aligned_mom), dim=0),
                                            )
                                            if use_shared_ambient_ef:
                                                alignment_drops[name.replace("lora_A", "lora_B")] = drop
                                            if use_factor_ambient_ef:
                                                factor_alignment_drops[name] = drop
                                    else:
                                        aligned_mom = old_mom
                                params_grads[name] = opt_params["server_momentum"] * aligned_mom + new_grad
                        else:
                            params_grads[name] = opt_params["server_momentum"] * opt_params["momentum"][name] + param.grad.to(torch.float64)
                    else:
                        params_grads[name] = param.grad.clone().to(torch.float64)
                    opt_params["momentum"][name] = params_grads[name].clone()
                    #opt_params["prev_factor"][name] = param.data
                else:
                    params_grads[name] = param.grad.clone().to(torch.float64)
                #print(name, param.grad.norm().item())
                """
                print("param original norm: ", param.data.norm().item(), param.data.dtype)
                print("param original norm 2: ", original_params_data[name].norm().item(), original_params_data[name].data.dtype)
                print("adapter_weights norm: ", adapter_weights[name].norm().item(), adapter_weights[name].data.dtype)
                print("adapter diff norm: ", param.grad.norm().item())
                print("adapter diff norm 2: ", (param.data - adapter_weights[name]).norm().item())
                """
            else:
                assert False

            #if opt_params["train_stats"]:
            #    grad_norm += torch.norm(param.grad).item()**2
    #if opt_params["train_stats"]:
    #    train_graphs.grad_norm.append(grad_norm ** 0.5)
    #    print("grad norm:", train_graphs.grad_norm[-1])
    
    if use_factor_ambient_ef:
        # v20 retains each dense FP64 debt with the same temporal decay as the
        # heavy-ball momentum that created it.
        ef_error_decay = opt_params["server_momentum"]

    if use_shared_ambient_ef:
        # Preserve the historical v19 defaults.
        ef_max_rank = 2 * opt_params["lora_rank"]
        ef_debt_cap = 10.0
        ef_error_decay = 1.0

    if use_factor_ambient_ef:
        for target_name, drop in factor_alignment_drops.items():
            is_A_target = "lora_A" in target_name
            name_A = target_name if is_A_target else target_name.replace("lora_B", "lora_A")
            name_B = target_name.replace("lora_A", "lora_B") if is_A_target else target_name
            target_factor = "A" if is_A_target else "B"
            delta, ef_stats = _factor_alignment_error_step(
                cur_adapter_weights[name_B].to(torch.float64),
                cur_adapter_weights[name_A].to(torch.float64),
                params_grads[target_name],
                drop,
                _factor_alignment_ef_state(opt_params, target_name),
                target_factor,
                ef_error_decay,
            )
            params_grads[target_name] = params_grads[target_name] + delta
            opt_params["momentum"][target_name] = params_grads[target_name].clone()
            print(f"factor alignment EF {target_name}: ||E||={ef_stats['error_norm']:.4e} "
                  f"||E||/||M_lift||={ef_stats['debt_ratio']:.3f} "
                  f"absorbed_{target_factor}={delta.norm().item():.4e}")

        # A factor-owned debt is part of heavy-ball momentum. If its source
        # factor is active this round, it is not transported, but it still
        # ages once; otherwise its temporal weight would depend on the switch
        # interval rather than server_momentum.
        processed = set(factor_alignment_drops)
        for name, state in opt_params.get("factor_ambient_ef", {}).items():
            if name in processed or "error" not in state:
                continue
            state["error"] = ef_error_decay * state["error"]

    if use_shared_ambient_ef:
        for name_B, drop in alignment_drops.items():
            name_A = name_B.replace("lora_B", "lora_A")
            delta_B, delta_A, ef_stats = _shared_alignment_error_step(
                cur_adapter_weights[name_B].to(torch.float64),   # B: (m, r)
                cur_adapter_weights[name_A].to(torch.float64),   # A: (r, n)
                params_grads[name_B],                            # M_B
                params_grads[name_A],                            # M_A
                drop,
                _shared_ambient_ef_state(opt_params, name_B),
                use_dense_ef_buffer,
                ef_max_rank,
                ef_error_decay,
                ef_debt_cap,
            )
            for nm, d in ((name_B, delta_B), (name_A, delta_A)):
                params_grads[nm] = params_grads[nm] + d
                opt_params["momentum"][nm] = params_grads[nm].clone()
            print(f"shared alignment EF {name_B}: ||E||={ef_stats['error_norm']:.4e} "
                  f"||E||/||M_ambient||={ef_stats['debt_ratio']:.3f} "
                  f"absorbed_B={delta_B.norm().item():.4e} "
                  f"absorbed_A={delta_A.norm().item():.4e}"
                  + (" [clipped]" if ef_stats["clipped"] else ""))

    print("="*10, " 4 ", "="*10)
    get_gpu_memory()


    # update the previous factors with the current factor (before real updates)
    if aligned_momentum:
        for name, param in model.named_parameters():
            if param.requires_grad:
                # store prev_factor in model dtype
                opt_params["prev_factor"][name] = param.data.clone()

    server_lr = 0
    for group in server_optimizer.param_groups:
        server_lr = group['lr']

    #server_optimizer.step()
    # compute the muon update directions
    muon_updates = {}
    updated_base_weights = {}
    server_param_updates = {}  # for muonlora_v10 and onwards: stores updated server adapter params after fusion
    balanced_probe_updates = {}
    """
    for name, param in model.named_parameters():
        if "muon_update" in name and 'lora_B' in name:
            grad_param_name_B = name.replace("muon_update", opt_params["server_name"])
            grad_param_name_A = grad_param_name_B.replace("lora_B", "lora_A")

            A_param = original_params_data[grad_param_name_A].T #n*r
            A_inv = torch.linalg.pinv(A_param) # r*n

            #print("pinv error:", torch.norm(A_inv @ A_param - torch.eye(A_inv.shape[0]).to(A_inv)).item())
            assert torch.norm(A_inv @ A_param - torch.eye(A_inv.shape[0]).to(A_inv)) < 1e-5

            Q_A, R_A = torch.linalg.qr(params_grads[grad_param_name_A].T, mode='reduced') #n*r, r*r
            Q_B, R_B = torch.linalg.qr(params_grads[grad_param_name_B], mode='reduced') #m*r, r*r

            C = R_B @ A_inv @ Q_A #r * r
            U, S, Vh = torch.linalg.svd(C, full_matrices=True)

            muon_update_name_A = name.replace("lora_B", "lora_A")
            muon_update_name_B = name
            muon_updates[muon_update_name_A] = -1 * Vh @ Q_A.T
            muon_updates[muon_update_name_B] = Q_B @ U @ torch.diag(S).to(U)

            recovered_grad = -1 * muon_updates[muon_update_name_B] @ muon_updates[muon_update_name_A]
            print("gradB norm:", params_grads[grad_param_name_B].norm().item())
            print("gradB error: ", (recovered_grad @ A_param - params_grads[grad_param_name_B]).norm().item())
            B_param = original_params_data[grad_param_name_B]
            print("gradA norm:", params_grads[grad_param_name_A].norm().item())
            print("gradA error: ", (recovered_grad.T @ B_param - params_grads[grad_param_name_A].T).norm().item())
    """
    if partial_merge and not use_balanced_probe_tracking:
        # Momentum transport above compares the pre-update snapshot from the
        # previous round with the current factor.  Advance only after that
        # comparison, so its gate denotes the factor that changed last round.
        _advance_alternating_phase(opt_params, server_epoch, alternate_update)

    print("="*10, " 5 ", "="*10)
    get_gpu_memory()


    for name, param in model.named_parameters():
        if "muon_update" in name and 'lora_B' in name:
            
            grad_param_name_B = name.replace("muon_update", opt_params["server_name"])
            grad_param_name_A = grad_param_name_B.replace("lora_B", "lora_A")
            muon_update_name_A = name.replace("lora_B", "lora_A")
            muon_update_name_B = name
            base_name = muon_update_name_B.replace("lora_B.muon_update", "base_layer")
            
            #grad_B, grad_A = params_grads[grad_param_name_B].to(torch.float64), params_grads[grad_param_name_A].T.to(torch.float64)
            # already fp64
            grad_B, grad_A = params_grads[grad_param_name_B], params_grads[grad_param_name_A].T

            A_param = original_params_data[grad_param_name_A].T.to(torch.float64) #n*r
            B_param = original_params_data[grad_param_name_B].to(torch.float64) #m*r
            
            proj_G = B_param.T @ grad_B #r*r
            U_G, S_G, Vh_G = torch.linalg.svd(proj_G, full_matrices=False)
            print("proj_G SVD: ", S_G)
            #proj_G_inv = torch.linalg.pinv(proj_G, atol=1e-6) #r*r
            """
            if opt_params["fedlora_avg"] in ['muonlora_v1']:
                proj_G_inv = torch.linalg.pinv(proj_G, rtol=1e-3) #r*r
            elif opt_params["fedlora_avg"] in ['muonlora_v2', 'muonlora_v3']:
                proj_G_inv = torch.linalg.pinv(proj_G) #r*r
            else:
                raise NotImplementedError(f"Choose how to inverse the matrix for {opt_params['fedlora_avg']}")
            """
            if use_damped_inv:
                proj_G_inv, inverse_damping = _damped_pinv_from_svd(U_G, S_G, Vh_G)
                print("proj_G inverse damping: ", inverse_damping.item())
            elif use_rtol_inv:
                proj_G_inv = torch.linalg.pinv(proj_G, rtol=1e-3)
            else:
                proj_G_inv = torch.linalg.pinv(proj_G)
            _, S_G_inv, _ = torch.linalg.svd(proj_G_inv, full_matrices=False)
            print("proj_G_inv SVD: ", S_G_inv)
            """
            recovered_grad = (grad_B @ torch.diag(S_G_inv) @ grad_A.T).to(torch.float32)
            print(name, "Gradient frob-norm: ", recovered_grad.norm().item())
            stable_rank = torch.linalg.norm(recovered_grad) / torch.linalg.matrix_norm(recovered_grad, ord=2)
            print(f"stable rank: {stable_rank}")
            """
            #print("pinv error:", torch.norm(A_inv @ A_param - torch.eye(A_inv.shape[0]).to(A_inv)).item())
            #assert torch.norm(proj_G_inv @ proj_G - torch.eye(proj_G_inv.shape[0]).to(proj_G_inv)) < 1e-5

            muon_update_name_A = name.replace("lora_B", "lora_A")
            muon_update_name_B = name

            #if opt_params["fedlora_avg"] in ['muonlora_v1', 'muonlora_v3']:
            #elif opt_params["fedlora_avg"] == 'muonlora_v2':
            if use_norm_grad: 
                """
                pseudo_grad = grad_B @ proj_G_inv @ grad_A.T
                U, _, Vh = torch.linalg.svd(pseudo_grad, full_matrices=False)
                muon_updates[muon_update_name_A] = -server_lr *  Vh.to(param.dtype)
                muon_updates[muon_update_name_B] = U.to(param.dtype)
                """
                M = grad_B
                N = (proj_G_inv @ grad_A.T).T

                Q_M, R_M = torch.linalg.qr(M)
                Q_N, R_N = torch.linalg.qr(N)

                R = R_M @ R_N.T
                U_R, S_R, Vh_R = torch.linalg.svd(R)

                
                muon_updates[muon_update_name_A] = (Q_N @ Vh_R.T).T #.to(param.dtype) #r*n keep it fp64
                muon_updates[muon_update_name_B] = (Q_M @ U_R) #.to(param.dtype) #m*r
                
                """
                print("Pseudo grad Norm: ", torch.norm(muon_updates[muon_update_name_B] @ muon_updates[muon_update_name_A]))
                if torch.norm(muon_updates[muon_update_name_B] @ muon_updates[muon_update_name_A]).item() > 0.01:
                    print("Muon update is too large! Warning!")
                """
                if partial_merge == True:
                    if use_balanced_probe_tracking:
                        # v17 treats the two server factors as balanced gradient
                        # probes. The effective Muon step is merged into the base;
                        # changing both probes is compensated exactly below.
                        scaling = opt_params["lora_alpha"] / opt_params["lora_rank"]
                        inv_s = 1.0 / scaling
                        target_B = muon_updates[muon_update_name_B]
                        target_A = muon_updates[muon_update_name_A].T

                        if opt_params["muonlora_scaled"]:
                            I_size = target_B.shape[0]
                            J_size = target_A.shape[0]
                            muon_scale = math.sqrt(I_size / J_size)
                        else:
                            muon_scale = 1.0

                        # Keep one factor orthonormal in the temporary adapter;
                        # merge_to_base's scaling then yields exactly
                        # -server_lr * muon_scale * target_B @ target_A.T.
                        muon_updates[muon_update_name_B] = (
                            -server_lr * inv_s * muon_scale * target_B
                        )
                        muon_updates[muon_update_name_A] = target_A.T

                        probe_beta = opt_params["muonlora_probe_beta"]
                        gamma = float(opt_params.get("lora_init_scale", 1.0))
                        desired_update_norm = server_lr * muon_scale
                        B_new, A_new, correction_norm, tracking_fraction = (
                            _balanced_probe_tracking_step(
                                B_param,
                                A_param.T,
                                target_B,
                                target_A,
                                gamma,
                                probe_beta,
                                scaling,
                                desired_update_norm,
                                opt_params["muonlora_max_correction_ratio"],
                            )
                        )

                        # Quantize now and use these exact stored values when the
                        # base compensation is formed, avoiding a probe/base
                        # mismatch caused by a later bf16/fp16 cast.
                        B_new = B_new.to(original_params_data[grad_param_name_B].dtype)
                        A_new = A_new.to(original_params_data[grad_param_name_A].dtype)
                        server_param_updates[grad_param_name_B] = B_new
                        server_param_updates[grad_param_name_A] = A_new
                        balanced_probe_updates[base_name] = (
                            B_param, A_param.T,
                            B_new.to(torch.float64), A_new.to(torch.float64),
                            muon_update_name_B, muon_update_name_A,
                        )
                        print(
                            "muonlora_v18 balanced probe tracking: "
                            f"beta={probe_beta:.6f} fraction={tracking_fraction:.6f} "
                            f"||compensation||2={correction_norm.item():.6e}"
                        )
                        # v18 has already constructed both adapter updates and
                        # the exact base compensation. Do not fall through to
                        # the legacy alternating A/B fusion, which relies on
                        # opt_params["update_B"].
                        continue
                    elif orth_then_merge:

                        if (server_epoch - 1) % opt_params["muonlora_switch_interval"]== 0:
                            # re-do SVD
                            #A_server = original_params_data[grad_param_name_A].to(torch.float64)  # (r, n)
                            #B_server = original_params_data[grad_param_name_B].to(torch.float64)  # (m, r)
                            #W_server = B_server @ A_server  # (m, n)
                            #U_svd, S_svd, Vh_svd = torch.linalg.svd(W_server, full_matrices=False)
                            # U_svd: (m, r)  S_svd: (r,)  Vh_svd: (r, n)
                            Q_B, R_B = torch.linalg.qr(B_param)
                            # A needs to be transposed so torch.linalg.qr operates on the columns
                            # Q_A will be (n, r), R_A will be (r, r)
                            Q_A, R_A = torch.linalg.qr(A_param)

                            # Mathematically: W = B @ A = (Q_B @ R_B) @ (R_A.T @ Q_A.T)
                            # So W = Q_B @ (R_B @ R_A.T) @ Q_A.T

                            # 2. Form the tiny core matrix to be SVD-ed
                            core_matrix = R_B @ R_A.T  # Shape: (r, r)

                            # 3. Perform SVD on the r x r core matrix instead of the m x n matrix
                            U_core, S_svd, Vh_core = torch.linalg.svd(core_matrix)  # All are (r, r) or (r,)

                            # 4. Project the singular vectors back to the original m and n dimensions
                            U_svd = Q_B @ U_core       # Shape: (m, r)
                            Vh_svd = Vh_core @ Q_A.T   # Shape: (r, n)

                            assert U_svd.shape == B_param.shape

                            U_svd_t = U_svd.to(param.dtype)
                            Vh_svd_t = Vh_svd.to(param.dtype)
                            S_diag = torch.diag(S_svd.to(param.dtype))

                            if opt_params["update_B"]:
                                original_params_data[grad_param_name_B] = U_svd_t @ S_diag
                                original_params_data[grad_param_name_A] = Vh_svd_t
                                server_param_updates[grad_param_name_A] = original_params_data[grad_param_name_A]
                            else:
                                original_params_data[grad_param_name_B] = U_svd_t
                                original_params_data[grad_param_name_A] = S_diag @ Vh_svd_t
                                server_param_updates[grad_param_name_B] = original_params_data[grad_param_name_B]

                        inv_s = opt_params["lora_rank"] / opt_params["lora_alpha"]
                        if opt_params["update_B"]:
                            # update is on B
                            muon_updates[muon_update_name_B] *= -server_lr * inv_s
                        else:
                            muon_updates[muon_update_name_A] *= -server_lr * inv_s
                    elif orth_then_merge == False:
                        # keep the original original_params_data
                        # Divide by s = lora_alpha/lora_rank so that:
                        #   merge_to_base adds s * (muon_B) @ (muon_A) = -lr * U @ (...),
                        # and the factor update ΔB = alpha * muon_B = -alpha*lr/s * U,
                        # matching the paper's formula W <- W - lr*UV^T with B <- B - alpha*lr/s * U.
                        inv_s = opt_params["lora_rank"] / opt_params["lora_alpha"]
                        if opt_params["update_B"]:
                            # update is on B
                            muon_updates[muon_update_name_B] *= -server_lr * inv_s
                        else:
                            muon_updates[muon_update_name_A] *= -server_lr * inv_s
                    # Split the update into a server-adapter-fused part and a residual muon update.
                    # Full update = V @ U  where V = muon_updates[B] (m×r), U = muon_updates[A] (r×n)
                    # SVD of current server adapter weight W = B_server @ A_server = U_svd diag(S) Vh_svd
                    # Alternating fusion:
                    #   A-side (even epoch): fuse V @ Vh_svd into server adapter (lossless via row-space preservation)
                    #     A_server_new = Vh_svd,  B_server_new = U_svd @ diag(S) + V
                    #     residual muon_updates[A] = U - Vh_svd,  muon_updates[B] = V  (unchanged)
                    #   B-side (odd epoch): fuse U_svd @ U into server adapter (symmetric)
                    #     B_server_new = U_svd,  A_server_new = diag(S) @ Vh_svd + U
                    #     residual muon_updates[B] = V - U_svd,  muon_updates[A] = U  (unchanged)
                    V = muon_updates[muon_update_name_A]  # (r, n)
                    U = muon_updates[muon_update_name_B]  # (m, r)

                    #if server_epoch % 2 == 0:
                    if opt_params["update_B"]:
                        # B-side fusion: fuse U into B, keep A=Vh_svd fixed
                        # B_server_new @ A_server_new = (U_svd @ S + U) @ Vh_svd = W + U @ Vh_svd
                        if opt_params["muonlora_scaled"]:
                            I_size, J_size = U.shape[0], V.shape[1]
                            scale = math.sqrt(I_size / J_size)
                            muon_updates[muon_update_name_B] *= scale
                            U = muon_updates[muon_update_name_B]
                        if use_projected_muon_factor_update:
                            (server_param_updates[grad_param_name_B],
                             muon_updates[muon_update_name_B],
                             muon_updates[muon_update_name_A]) = _projected_muon_factor_split(
                                B_param, A_param.T, U, V,
                                opt_params["muonlora_merge_alpha"], True,
                                projected_muon_scaled_identity,
                                float(opt_params["lora_init_scale"]))
                        else:
                            server_param_updates[grad_param_name_B] = B_param + opt_params["muonlora_merge_alpha"] * U
                            muon_updates[muon_update_name_A] = V - opt_params["muonlora_merge_alpha"] * A_param.T
                        # muon_updates[muon_update_name_B] = U (unchanged)
                        from utilities import principal_angle
                        print(f"param norm: {B_param.float().norm().item()} U norm: {U.float().norm().item()}")
                        #principal_angle(grad_param_name_B, original_params_data[grad_param_name_B].float(), U.float())
                        #principal_angle(grad_param_name_B + " after update", original_params_data[grad_param_name_B].float(), server_param_updates[grad_param_name_B].float())

                        # v14: re-orthonormalize the updated B factor (Sec 3.7).
                        # Let Q*R = B_new (QR decomp). Set B <- gamma*Q.
                        # Correction to W: (lora_alpha/r) * (B_new - gamma*Q) @ A_weight,
                        # stored as rank-r factors (B_corr, A_corr_weight) and merged later.
                        if v14_style_reorth:
                            B_new = server_param_updates[grad_param_name_B]  # (m, r), float64
                            Q_B, _ = torch.linalg.qr(B_new, mode='reduced')   # Q_B: (m, r)
                            gamma_orth = float(opt_params.get("lora_init_scale", 1.0))
                            B_corr = B_new - gamma_orth * Q_B       # (m, r) — left factor
                            A_corr = A_param.T                       # (r, n) — right factor (= lora_A weight)
                            if "orth_corrections" not in opt_params:
                                opt_params["orth_corrections"] = {}
                            opt_params["orth_corrections"][base_name] = (B_corr, A_corr)
                            server_param_updates[grad_param_name_B] = gamma_orth * Q_B
                            print(f"{opt_params['fedlora_avg']} B-side orth: ||B_new-gamma*Q||={B_corr.norm().item():.4f}")
                        elif use_product_preserving_qr_gauge:
                            B_new = server_param_updates[grad_param_name_B]  # (m, r), float64
                            gamma_orth = float(opt_params.get("lora_init_scale", 1.0))
                            B_retracted, A_transferred = _qr_transfer_from_B(
                                B_new, A_param.T, gamma_orth
                            )
                            server_param_updates[grad_param_name_B] = B_retracted
                            server_param_updates[grad_param_name_A] = A_transferred
                            print(f"{opt_params['fedlora_avg']} B-side product-preserving QR transfer applied")
                    else:
                        # A-side fusion: fuse V into A, keep B=U_svd fixed
                        # B_server_new @ A_server_new = U_svd @ (S @ Vh_svd + V) = W + U_svd @ V
                        if opt_params["muonlora_scaled"]:
                            I_size, J_size = U.shape[0], V.shape[1]
                            scale = math.sqrt(I_size / J_size)
                            muon_updates[muon_update_name_A] *= scale
                            V = muon_updates[muon_update_name_A]
                        if use_projected_muon_factor_update:
                            (server_param_updates[grad_param_name_A],
                             muon_updates[muon_update_name_B],
                             muon_updates[muon_update_name_A]) = _projected_muon_factor_split(
                                B_param, A_param.T, U, V,
                                opt_params["muonlora_merge_alpha"], False,
                                projected_muon_scaled_identity,
                                float(opt_params["lora_init_scale"]))
                        else:
                            server_param_updates[grad_param_name_A] = A_param.T + opt_params["muonlora_merge_alpha"] * V
                            muon_updates[muon_update_name_B] = U - opt_params["muonlora_merge_alpha"] * B_param
                        from utilities import principal_angle
                        print(f"param norm: {A_param.float().norm().item()} V norm: {V.float().norm().item()}")
                        #principal_angle(grad_param_name_A, original_params_data[grad_param_name_A].float().T, V.float().T)
                        #principal_angle(grad_param_name_A + " after update", original_params_data[grad_param_name_A].float().T, server_param_updates[grad_param_name_A].float().T)
                        # muon_updates[muon_update_name_A] = U  (unchanged)
                        #print(f"v10 B-side fusion: ||fused||={torch.norm(U_svd_t @ V_mu).item():.4f}, ||residual||={torch.norm(muon_updates[muon_update_name_B] @ muon_updates[muon_update_name_A]).item():.4f}")

                        # v14: re-orthonormalize the updated A factor (Sec 3.7).
                        # A_weight is (r, n); columns of A_math = A_weight.T are (n, r).
                        # Let Q*R = A_math (QR decomp). Set A_math <- gamma*Q, i.e. A_weight <- gamma*Q.T.
                        # Correction: (lora_alpha/r) * B @ (A_weight_new - gamma*Q.T),
                        # stored as rank-r factors (B_corr, A_corr_weight) and merged later.
                        if v14_style_reorth:
                            A_weight_new = server_param_updates[grad_param_name_A]  # (r, n), float64
                            Q_A, _ = torch.linalg.qr(A_weight_new.T, mode='reduced')  # Q_A: (n, r)
                            gamma_orth = float(opt_params.get("lora_init_scale", 1.0))
                            B_corr = B_param                             # (m, r) — left factor
                            A_corr = A_weight_new - gamma_orth * Q_A.T  # (r, n) — right factor
                            if "orth_corrections" not in opt_params:
                                opt_params["orth_corrections"] = {}
                            opt_params["orth_corrections"][base_name] = (B_corr, A_corr)
                            server_param_updates[grad_param_name_A] = gamma_orth * Q_A.T
                            print(f"{opt_params['fedlora_avg']} A-side orth: ||A_new-gamma*Q||={A_corr.norm().item():.4f}")
                        elif use_product_preserving_qr_gauge:
                            A_weight_new = server_param_updates[grad_param_name_A]  # (r, n), float64
                            gamma_orth = float(opt_params.get("lora_init_scale", 1.0))
                            B_transferred, A_retracted = _qr_transfer_from_A(
                                B_param, A_weight_new, gamma_orth
                            )
                            server_param_updates[grad_param_name_B] = B_transferred
                            server_param_updates[grad_param_name_A] = A_retracted
                            print(f"{opt_params['fedlora_avg']} A-side product-preserving QR transfer applied")
                else:
                    #scale either side is ok
                    inv_s = opt_params["lora_rank"] / opt_params["lora_alpha"]
                    muon_updates[muon_update_name_A] *= -server_lr * inv_s
                    if opt_params["muonlora_scaled"]:
                        I_size, J_size = muon_updates[muon_update_name_B].shape[0], muon_updates[muon_update_name_A].shape[1]
                        scale = math.sqrt(I_size / J_size)
                        muon_updates[muon_update_name_A] *= scale
                #print(muon_update_name_A, muon_updates[muon_update_name_A].shape, muon_updates[muon_update_name_B].shape)
            else:
                if apply_momentum:
                    muon_updates[muon_update_name_B], muon_updates[muon_update_name_A] = muon_momentum(grad_B @ proj_G_inv, grad_A, server_lr, opt_params, base_name, use_rtol_inv, param.dtype)
                else:
                    muon_updates[muon_update_name_A] = -server_lr * grad_A.T.to(param.dtype) #r*n
                    muon_updates[muon_update_name_B] = (grad_B @ proj_G_inv).to(param.dtype) #m*r
                    print("Pseudo grad Norm: ", torch.norm(muon_updates[muon_update_name_B] @ muon_updates[muon_update_name_A]))
                    if torch.norm(muon_updates[muon_update_name_B] @ muon_updates[muon_update_name_A]).item() > 0.01:
                        print("Muon update is too large! Warning!")
                
                if opt_params["muonlora_scaled"]:
                    I_size, J_size = muon_updates[muon_update_name_B].shape[0], muon_updates[muon_update_name_A].shape[1]
                    scale = math.sqrt(I_size / J_size)
                    muon_updates[muon_update_name_B] *= scale
                
            
            #recovered_grad = -1 * muon_updates[muon_update_name_B] @ muon_updates[muon_update_name_A]
            #print("gradB norm:", params_grads[grad_param_name_B].norm().item())
            #print("gradB error: ", (recovered_grad @ original_params_data[grad_param_name_A].T - params_grads[grad_param_name_B]).norm().item())
            
            #print("gradA norm:", params_grads[grad_param_name_A].norm().item())
            #print("gradA error: ", (recovered_grad.T @ original_params_data[grad_param_name_B] - params_grads[grad_param_name_A].T).norm().item())
            #muon_updates[muon_update_name_A] = (A_param.T -  server_lr * grad_A.T).to(param.dtype)
            #muon_updates[muon_update_name_B] = (B_param -  server_lr * grad_B).to(param.dtype)

            #muon_updates[muon_update_name_A] = step_server_adapters[grad_param_name_A]
            #muon_updates[muon_update_name_B] = step_server_adapters[grad_param_name_B]

            #base_name = name.replace("lora_B.muon_update", "base_layer")
            #updated_base_weights[base_name] = base_weights[base_name] + (muon_updates[muon_update_name_B] @ muon_updates[muon_update_name_A]).T
            #server_adapter_name_B = name.replace("muon_update", opt_params["server_name"])
            #server_adapter_name_A = server_adapter_name_B.replace("lora_B", "lora_A")

            #muon_updates[muon_update_name_A] = adapter_weights[server_adapter_name_A]
            #muon_updates[muon_update_name_B] = adapter_weights[server_adapter_name_B]

    #sys.exit()
    # muonlora_v10: apply fused updates to server adapter params before merging muon_update
    if partial_merge:
        #apply fused updates to server adapter params before merging muon_update
        assert server_param_updates != {}
        for name, param in model.named_parameters():
            if name in server_param_updates:
                param.data = server_param_updates[name].to(param.dtype)

    muon_update_num = 0
    #print("muon update")
    print("="*10, " 6 ", "="*10)
    get_gpu_memory()

    for name, param in model.named_parameters():
        if "muon_update" in name:
            param.data = muon_updates[name]
        #if opt_params["server_name"] in name:
        #   muon_name = name.replace(opt_params["server_name"], "muon_update")
        #    param.data = muon_updates[muon_name]
            muon_update_num += 1
            #print(name, param.data.norm().item(),
            #      (param.data - muon_updates[muon_name]).norm().item())

    assert muon_update_num == len(muon_updates)

    #### muonlora merge step
    #print("I am merging server_name adapter")
    #model.merge_adapter([opt_params["server_name"]])
    #print("I am merging muon_update adapter")
    #model.merge_adapter(["muon_update"])
    if use_balanced_probe_tracking:
        # Apply the desired Muon step and the exact adapter-reparameterization
        # compensation in one cast to the base dtype:
        #   dW_base = s * (U_step V_step + B_old A_old - B_new A_new).
        scaling = opt_params["lora_alpha"] / opt_params["lora_rank"]
        applied_probe_updates = 0
        for name, param in model.named_parameters():
            if name not in balanced_probe_updates:
                continue
            B_old, A_old, B_new, A_new, update_B_name, update_A_name = (
                balanced_probe_updates[name]
            )
            effective_base_update = scaling * (
                muon_updates[update_B_name] @ muon_updates[update_A_name]
                + B_old @ A_old
                - B_new @ A_new
            )
            if opt_params["model_name"] == "gpt2":
                effective_base_update = effective_base_update.T
            elif opt_params["model_name"] not in [
                "meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.1-8B",
                "meta-llama/Llama-3.2-1B",
                "meta-llama/Llama-3.2-3B",
            ]:
                raise NotImplementedError
            param.data += effective_base_update.to(param.dtype)
            applied_probe_updates += 1
        assert applied_probe_updates == len(balanced_probe_updates)
    else:
        merge_to_base(model,
                    adapter_name="muon_update",
                    lora_r=opt_params["lora_rank"],
                    lora_alpha=opt_params["lora_alpha"],
                    model_name=opt_params["model_name"])

    print("="*10, " 7 ", "="*10)
    get_gpu_memory()

    # v14: apply orthonormalization corrections via the orth_correction adapter
    if v14_style_reorth and opt_params.get("orth_corrections"):
        for name, param in model.named_parameters():
            if "orth_correction" not in name:
                continue
            if "lora_B" in name:
                key = name.replace("lora_B.orth_correction", "base_layer")
                if key in opt_params["orth_corrections"]:
                    B_corr, _ = opt_params["orth_corrections"][key]
                    param.data = B_corr.to(param.dtype)
            elif "lora_A" in name:
                key = name.replace("lora_A.orth_correction", "base_layer")
                if key in opt_params["orth_corrections"]:
                    _, A_corr = opt_params["orth_corrections"][key]
                    param.data = A_corr.to(param.dtype)
        merge_to_base(model,
                    adapter_name="orth_correction",
                    lora_r=opt_params["lora_rank"],
                    lora_alpha=opt_params["lora_alpha"],
                    model_name=opt_params["model_name"])
        opt_params["orth_corrections"] = {}

    #model.merge_adapter(["fr_save_neg_init"])
    """
    merge_to_base(model,
                adapter_name="fr_save_neg_init", 
                lora_r=opt_params["lora_rank"], 
                lora_alpha=opt_params["lora_alpha"], 
                model_name=opt_params["model_name"])
    """
    #reset server adapter to fr_save_init -- prepare for the next round, skip the output layer

    if opt_params["local_update_ON"]:
        from arch.lora import synchronize_lora_server
        synchronize_lora_server(model, "fr_save_init", opt_params["server_name"], truncate_last=True, skip_output_layer_name=output_layer_name)

        # reinitialize the client lora params with fr_save_init, must include the output layer
        synchronize_lora(model, opt_params["server_name"], truncate_last=True)

    if server_lr_scheduler is not None:
        server_lr_scheduler.step()

    for group in server_optimizer.param_groups:
        print("server lr", group['lr'])

    if opt_params["local_update_ON"]:
        model.set_adapter(opt_params["server_name"])

def Orth(matrix):
    Q, _ = torch.qr(matrix)
    return Q

def two_side_linear(A, B , C, D, use_rtol_inv):
    #solve for the linear system
    #G A = C
    #G.T B = D
    # G = C (C.T @ B)^{-1} D.T
    if use_rtol_inv:
        proj_G_inv = torch.linalg.pinv(B.T @ C, rtol=1e-3)
    else:
        proj_G_inv = torch.linalg.pinv(B.T @ C)

    M = C
    N = (proj_G_inv @ D.T).T

    Q_M, R_M = torch.linalg.qr(M)
    Q_N, R_N = torch.linalg.qr(N)

    R = R_M @ R_N.T
    U_R, S_R, Vh_R = torch.linalg.svd(R)

    
    #muon_updates[muon_update_name_A] = -server_lr * (Q_N @ Vh_R.T).T.to(param.dtype) #r*n
    #muon_updates[muon_update_name_B] = (Q_M @ U_R).to(param.dtype) #m*r


    return Q_M @ U_R, (Q_N @ Vh_R.T) #m*r, n*r


def muon_momentum(G_L, G_R, lr, opt_params, base_name, use_rtol_inv, dtype):
    # G = G_L @ G_R.T
    # P_t = M
    print("running muon_momentum")
    beta = opt_params["server_momentum"]
    assert beta >= 0 and beta < 1

    if "muon_states" not in opt_params:
        opt_params["muon_states"] = {}
    if base_name not in opt_params["muon_states"]:
        opt_params["muon_states"][base_name] = {}
    base_states = opt_params["muon_states"][base_name]
    if "U" not in base_states:
        # This is the first epoch
        M = G_L
        N = G_R

        Q_M, R_M = torch.linalg.qr(M)
        Q_N, R_N = torch.linalg.qr(N)

        R = R_M @ R_N.T
        U_R, S_R, Vh_R = torch.linalg.svd(R)
        update_R = (Q_N @ Vh_R.T).T #r*n
        update_L = (Q_M @ U_R) #m*r

        base_states["U"] = update_L
        base_states["V_t"] = update_R.T #n*r
        base_states["P"] = beta * G_L @ (G_R.T @ base_states["V_t"]) #m*r
        #opt_params["U"] = Orth(opt_params["P"])
        base_states["W"] = beta * G_R @ (G_L.T @ base_states["U"]) #n*r
        #opt_params["V_t"] = Orth(opt_params["W"]) #n*r
        #return -lr * update_L, update_R
        return -lr * base_states["U"].to(dtype), base_states["V_t"].T.to(dtype)
    elif opt_params["fedlora_avg"] in ["muonlora_v5", "muonlora_v6"]:
        print(base_name)
        base_states["P"] += (1-beta) * G_L @ (G_R.T @ base_states["V_t"])
        new_U = Orth(base_states["P"]) #m*r
        
        #W = M.T @ U
        base_states["W"] = beta * base_states["W"] @ (base_states["U"].T @ new_U) + (1-beta) * G_R @ (G_L.T @ new_U)  #n*r
        print("U.TU: ", base_states["P"].norm().item(), (base_states["U"].T @ new_U).norm().item())
        base_states["U"] = new_U

        #update P and W
        #new_V = Orth(base_states["W"]) #n*r
        print("W col norm:", torch.norm(base_states["W"], dim=0))
        new_V = base_states["W"] / (torch.norm(base_states["W"], dim=0) + 1e-8)
        print("new_V col norm:", torch.norm(new_V, dim=0))
        base_states["P"] = beta * base_states["P"] @ (base_states["V_t"].T @ new_V)
        print("V.TV: ", base_states["V_t"].norm().item(), new_V.norm().item(), (base_states["V_t"].T @ new_V).norm().item())
        print("U.T G_L: ", (new_U.T @ G_L).norm().item(), G_L.norm().item())
        print("G_R.T V: ", (G_R.T @ new_V).norm().item(), G_R.norm().item())
        #print("U.T G V: ", (new_U.T @ G_L @ G_R.T @ new_V).norm.item())
        base_states["V_t"] = new_V
        return -lr * base_states["U"].to(dtype), base_states["V_t"].T.to(dtype)
    elif opt_params["fedlora_avg"] in ["muonlora_v7", "muonlora_v8"]:
        print(base_name)
        base_states["P"] += G_L @ (G_R.T @ base_states["V_t"])
        new_U = Orth(base_states["P"]) #m*r
        
        #W = M.T @ U
        base_states["W"] = beta * base_states["W"] @ (base_states["U"].T @ new_U) + G_R @ (G_L.T @ new_U)  #n*r
        print("U.TU: ", base_states["P"].norm().item(), (base_states["U"].T @ new_U).norm().item())
        if base_states["P"].norm().item() > 100:
            print("WARNING: accumulated P is too large.")
        base_states["U"] = new_U

        update_L, update_R = two_side_linear(base_states["V_t"], new_U, base_states["P"], base_states["W"], use_rtol_inv)

        #update P and W
        new_V = Orth(base_states["W"]) #n*r
        #print("W col norm:", torch.norm(base_states["W"], dim=0))
        #new_V = base_states["W"] / (torch.norm(base_states["W"], dim=0) + 1e-8)
        #print("new_V col norm:", torch.norm(new_V, dim=0))
        base_states["P"] = beta * base_states["P"] @ (base_states["V_t"].T @ new_V)
        print("V.TV: ", base_states["V_t"].norm().item(), new_V.norm().item(), (base_states["V_t"].T @ new_V).norm().item())
        print("U.T G_L: ", (new_U.T @ G_L).norm().item(), G_L.norm().item())
        print("G_R.T V: ", (G_R.T @ new_V).norm().item(), G_R.norm().item())
        #print("U.T G V: ", (new_U.T @ G_L @ G_R.T @ new_V).norm.item())
        base_states["V_t"] = new_V
        return -lr * update_L.to(dtype), update_R.T.to(dtype)
    """
    else:
        base_states["P"] += G_L @ (G_R.T @ base_states["V_t"])
        new_U = Orth(base_states["P"]) #m*r
        
        #W = M.T @ U
        base_states["W"] = beta * base_states["W"] @ (base_states["U"].T @ new_U) + G_R @ (G_L.T @ new_U)  #n*r
        print("U.TU: ", (base_states["U"].T @ new_U).norm().item())
        base_states["U"] = new_U

        #update P and W
        new_V = Orth(base_states["W"]) #n*r
        base_states["P"] = beta * base_states["P"] @ (base_states["V_t"].T @ new_V)
        print("V.TV: ", (base_states["V_t"].T @ new_V).norm().item())
        print("U.T G_L: ", (new_U.T @ G_L).norm().item(), G_L.norm().item())
        print("G_R V: ", (G_R.T @ new_V).norm().item(), G_R.norm().item())
        #print("U.T G V: ", (new_U.T @ G_L @ G_R.T @ new_V).norm.item())
        base_states["V_t"] = new_V
        return -lr * base_states["U"].to(dtype), base_states["V_t"].T.to(dtype)
    """
    """
    else:
        #opt_params["P"] = opt_params["P"] @ ((opt_params["V_t-1"]).T @ (opt_params["V_t"])) #m*r
        #opt_params["P"] -= beta * opt_params["U"] @ opt_params["W"] @ opt_params["V_t"]
        #print(base_states["P"].shape, G_L.shape, G_R.T.shape, base_states["V_t"].shape)
        base_states["P"] += G_L @ (G_R.T @ base_states["V_t"])
        new_U = Orth(base_states["P"]) #m*r
        
        #W = M.T @ U
        base_states["W"] = beta * base_states["W"] @ (base_states["U"].T @ new_U) + G_R @ (G_L.T @ new_U)  #n*r
        base_states["U"] = new_U

        #update P and W
        new_V = Orth(base_states["W"]) #n*r
        base_states["P"] = base_states["P"] @ (base_states["V_t"].T @ new_V) - (1-beta) * base_states["U"] @ (base_states["W"].T @ new_V) #m*r
        base_states["V_t"] = new_V

        print(base_name, base_states["P"].norm(), base_states["W"].norm(), base_states["U"].norm(), base_states["V_t"].norm())
        return -lr * base_states["U"].to(dtype), base_states["V_t"].T.to(dtype)
    """

def get_fr_hparams(fedlora_avg_name):
    if fedlora_avg_name == "fr":
        use_model_grad = False
    elif fedlora_avg_name == "fr_v2":
        use_model_grad = True
    else:
        raise NotImplementedError
    return use_model_grad

def federated_frlora(model, loss_name, criterion, lora_rank, train_graphs, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, opt_params, model_params, server_epoch):
    client_num, client_opt_name, client_epoch = opt_params["client_num"], opt_params["client_opt_name"], opt_params["client_epoch"]
    import copy
    import math
    from utilities import vector_to_grads, vector_to_grads_sq
    from main import train

    use_model_grad = get_fr_hparams(fedlora_avg_name=opt_params["fedlora_avg"])
    if use_model_grad:
        opt_params["local_update_ON"] = False
    else:
        opt_params["local_update_ON"] = True

    adapter_names = []
    adapter_weights = {}
    output_weights = {}
    output_layer_name = opt_params["output_layer_name"]
    base_original_param = {}

    model.set_adapter(opt_params["server_name"])
    for name, param in model.named_parameters():
        # select lora_A and lora_B
        if 'base_layer' in name and 'weight' in name:
            base_original_param[name] = param.data.clone()
        if param.requires_grad:
            if output_layer_name and output_layer_name in name:
                output_weights[name]= 0
            else:
                adapter_names.append(name)
                adapter_weights[name] = torch.zeros_like(param)

    client_opt_params = copy.deepcopy(opt_params)
    client_opt_params["train_stats"] = False

    if opt_params["client_partial"] < 1:
        client_num = int(opt_params["client_partial"] * client_num)
        client_selected = np.random.choice(opt_params["client_num"], client_num, replace=False)
    else:
        client_selected = np.arange(client_num)

    for client_id in client_selected:
        # update client models
        adapter_name = "client_{}".format(client_id)
        if opt_params["local_update_ON"]:
            model.set_adapter(adapter_name)
        client_model = model #alias

        client_model.train()
        optimizer, lr_scheduler, _= load_optimizer(client_opt_name, client_model, client_lr, opt_params["client_momentum"], opt_params["client_weight_decay"], opt_params["lr_decay"], opt_params["epochs_lr_decay"], False, model_params, opt_params)
        #vector_to_parameters(old_params, client_model.parameters())
        for epoch in range(client_epoch):
            train_graphs.loader_iter += 1
            try:
                assert iter(train_loaders[0]) == train_loaders[0]
                _, model_grad = train(client_model, loss_name, criterion, device, train_loaders[0], optimizer, lr_scheduler, server_epoch, client_opt_params)
            except StopIteration:
                # reinitialize iterator
                print("\nData Iterator is reloaded")
                train_loaders[0] = iter(train_loaders[1])
                _, model_grad = train(client_model, loss_name, criterion, device, train_loaders[0], optimizer, lr_scheduler, server_epoch, client_opt_params)

        for name, param in client_model.named_parameters():
            if param.requires_grad:
                #param_names.append(name)
                server_adapter_name = name.replace("{}".format(adapter_name), opt_params["server_name"])
                if output_layer_name and output_layer_name in name:
                    if use_model_grad:
                        output_weights[server_adapter_name] += model_grad[name]
                    else:
                        output_weights[server_adapter_name] += param.data #/ client_num
                else:
                    if server_adapter_name in adapter_weights:
                        row, col = param.data.shape
                        if use_model_grad:
                            adapter_weights[server_adapter_name][:row, :col] += model_grad[name]
                        else:
                            adapter_weights[server_adapter_name][:row, :col] += param.data #/client_num
                    else:
                        assert False
    
    for server_adapter_name in output_weights:
        output_weights[server_adapter_name] = output_weights[server_adapter_name] / client_num

    for server_adapter_name in adapter_weights:
        adapter_weights[server_adapter_name] = adapter_weights[server_adapter_name] / client_num

    if opt_params["local_update_ON"]:
        model.set_adapter(opt_params["server_name"])
    #truncate_err, truncate_err_ratio = compute_truncate_err(model, adapter_weights, client_num, opt_params["model_name"], opt_params["server_name"])
    server_optimizer.zero_grad()

    #train_graphs.truncate_err.append(truncate_err)
    #train_graphs.truncate_err_ratio.append(truncate_err_ratio)
    #print("Truncation Error: ", train_graphs.truncate_err[-1])
    #print("Truncation Error Ratio: ", train_graphs.truncate_err_ratio[-1])

    if opt_params["train_stats"]:
        grad_norm = 0

    #print("Gradient Norm: ")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if output_layer_name and output_layer_name in name:
                if use_model_grad:
                    param.grad = output_weights[name]
                else:
                    param.grad = param.data - output_weights[name]
            elif name in adapter_weights:
                if use_model_grad:
                    param.grad = adapter_weights[name]
                else:
                    param.grad = param.data - adapter_weights[name]
            else:
                assert False
            #print(name, param.grad.dtype, param.grad.norm())
            if opt_params["train_stats"]:
                grad_norm += torch.norm(param.grad).item()**2
    if opt_params["train_stats"]:
        train_graphs.grad_norm.append(grad_norm ** 0.5)
        print("grad norm:", train_graphs.grad_norm[-1])
    
    server_optimizer.step()

    
    #### frlora merge step
    """
    model.add_weighted_adapter([opt_params["server_name"], "fr_save_init"], [1.0, -1.0], \
                    adapter_name="fr_merge", combination_type= 'cat')

    print("after add_weighted_adapter")
    for name, param in model.named_parameters():
        if "base_layer" in name:
            print(name, param.norm().item())
            break

        if "fr_merge" in name:
            print(name, param.norm().item())

    model.merge_adapter(["fr_merge"])
    
    base_layer_param = 0
    for name, param in model.named_parameters():
        if "base_layer" in name:
            #print(name, param.norm().item())
            base_layer_name = name
            base_layer_param = param.data.clone()
            print("base_layer_param norm: ", base_layer_param.norm())
            break 

    for name, param in model.named_parameters():
        name_lookup_A = base_layer_name.replace("base_layer", f"lora_A.{opt_params['server_name']}")
        name_lookup_B = base_layer_name.replace("base_layer", f"lora_B.{opt_params['server_name']}")
        if name == name_lookup_A:
            #print(name, param.norm().item())
            adapter_A = param.data.clone()
        if name == name_lookup_B:
            #print(name, param.norm().item())
            adapter_B = param.data.clone()

    print("merged weight norm should be 1: ")
    base_layer_param = base_layer_param + (adapter_B @ adapter_A).T
    print(torch.norm(base_layer_param).item())
    """
    #model.merge_adapter([opt_params["server_name"]])
    merge_to_base(model,
                adapter_name=opt_params["server_name"], 
                lora_r=opt_params["lora_rank"], 
                lora_alpha=opt_params["lora_alpha"], 
                model_name=opt_params["model_name"])
    """
    for name, param in model.named_parameters():
        if "base_layer" in name:
            #print(name, param.norm().item())
            base_layer_param = param.data.clone()
            print("base_layer_param norm after merge server adapter: ", base_layer_param.norm())
            break 
    """
    #model.merge_adapter(["fr_save_neg_init"])
    merge_to_base(model,
                adapter_name="fr_save_neg_init", 
                lora_r=opt_params["lora_rank"], 
                lora_alpha=opt_params["lora_alpha"], 
                model_name=opt_params["model_name"])
    """
    print("comparing merged adapters and computed adapters")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in adapter_weights:
                print(name, (param.data - adapter_weights[name]).norm().item())
        elif name in output_weights:
            print(name, (param.data - output_weights[name]).norm().item())
    """

    print("after merging")
    for name, param in model.named_parameters():
        if "base" in name and 'weight' in name:
            pseudo_grad_norm = (param.data - base_original_param[name]).norm().item()
            print(name, type(pseudo_grad_norm), pseudo_grad_norm)
            if pseudo_grad_norm > 0.01:
                print("Base layer update is too large! Warning!")

    #model.merge_adapter(["fr_save_init"])

    #reset server adapter to fr_save_init -- prepare for the next round, skip the output layer
    from arch.lora import synchronize_lora_server
    synchronize_lora_server(model, "fr_save_init", opt_params["server_name"], truncate_last=True, skip_output_layer_name=output_layer_name)
    #synchronize_lora(model, opt_params["server_name"], truncate_last=True)

    
    # reinitialize the client lora params with fr_save_init, must include the output layer
    #synchronize_lora(model, "fr_save_init", truncate_last=True)
    synchronize_lora(model, opt_params["server_name"], truncate_last=True)

    """
    print("after merge server adapter")
    for name, param in model.named_parameters():
        print(name, param.norm().item())
    """
    if server_lr_scheduler is not None:
        server_lr_scheduler.step()

    for group in server_optimizer.param_groups:
        print("server lr", group['lr'])

    # very important: merge_adapter will change the current active adapter
    model.set_adapter(opt_params["server_name"])
    

def add_noise(mat, noise, clip_quantile, sketch_size=-1):
    q95 = torch.quantile(mat.abs().reshape(-1), clip_quantile)
    print("clip threshold: ", q95)
    clip_threshold = q95
    #clip_threshold = torch.min(0.2 * torch.ones_like(clip_threshold), clip_threshold)
    clip_threshold = torch.min(0.1 * torch.ones_like(clip_threshold), clip_threshold)
    clip_threshold = torch.max(0.01 * torch.ones_like(clip_threshold), clip_threshold)
    mat = torch.clamp(mat, -clip_threshold, clip_threshold)
    mat_clone = mat.clone()
    if sketch_size == -1:
        noise = noise * clip_threshold
        mat = mat + torch.randn_like(mat) * noise
    else:
        from hadamard_transform import hadamard_transform, pad_to_power_of_2 
        import math
        p = max(mat.size())
        p_pad = pow(2, math.ceil(math.log(p)/math.log(2)))

        D = (((torch.randn(p_pad, dtype=torch.float32) > 0).float() - 0.5) * 2).to(mat)
        sample_rows = torch.stack([torch.arange(sketch_size), torch.randperm(p_pad)[:sketch_size]]).to(mat)
        sub_sample_row = torch.sparse_coo_tensor(sample_rows, torch.ones(sketch_size).to(mat), [sketch_size, p_pad]) * ((p_pad/sketch_size)**0.5)

        assert mat.ndim == 2
        #sketch
        if mat.shape[0] > mat.shape[1]:
            mat = mat.T
        params_pad = pad_to_power_of_2(mat.detach()) #r * m -> r * p_pad
        hadamard_params_pad = hadamard_transform(D*params_pad)
        sketched_mat = hadamard_params_pad @ sub_sample_row.T #r * sketch
        #desketch
        desk_mat = sketched_mat @ sub_sample_row # r * p_pad
        desk_mat = hadamard_transform(desk_mat) * D # r * p_pad
        desk_mat = desk_mat[:, :p]
        
        if mat_clone.shape[0] > mat_clone.shape[1]: #compare the orginal shape
            mat = desk_mat.T
        else:
            mat = desk_mat
        print("sketching error: ", torch.norm(mat - mat_clone).item(), torch.norm(mat_clone).item())
        #additive noise
        noise = noise * clip_threshold
        mat = mat + torch.randn_like(mat) * noise
    return mat

def private_lora_avg(model, loss_name, criterion, lora_rank, train_graphs, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, opt_params, model_params, server_epoch):
    print("in private lora avg")
    client_num, client_opt_name, client_epoch = opt_params["client_num"], opt_params["client_opt_name"], opt_params["client_epoch"]
    import copy
    import math
    from utilities import vector_to_grads, vector_to_grads_sq
    from main import train

    adapter_names = []
    adapter_weights = {}
    output_weights = {}
    output_layer_name = opt_params["output_layer_name"]

    model.set_adapter(opt_params["server_name"])
    if opt_params["lora_freeze_a"]:
        for n, p in model.named_parameters():
            if "lora_A" in n:
                p.requires_grad=False
    for name, param in model.named_parameters():
        # select lora_A and lora_B
        if param.requires_grad:
            if output_layer_name and output_layer_name in name:
                output_weights[name]= 0
            else:
                adapter_names.append(name)
                adapter_weights[name] = torch.zeros_like(param)

    client_opt_params = copy.deepcopy(opt_params)
    client_opt_params["train_stats"] = False

    if opt_params["client_partial"] < 1:
        client_num = int(opt_params["client_partial"] * client_num)
        client_selected = np.random.choice(opt_params["client_num"], client_num, replace=False)
    else:
        client_selected = np.arange(client_num)

    print("after local train on client: ", client_selected)
    for client_id in client_selected:
        # update client models
        adapter_name = "client_{}".format(client_id)
        
        client_model = model #alias
        #client_model = copy.deepcopy(model)
        client_model.set_adapter(adapter_name)
        if opt_params["lora_freeze_a"]:
            for n, p in client_model.named_parameters():
                if "lora_A" in n:
                    p.requires_grad=False

        client_model.train()
        optimizer, lr_scheduler, _= load_optimizer(client_opt_name, client_model, client_lr, opt_params["client_momentum"], opt_params["client_weight_decay"], opt_params["lr_decay"], opt_params["epochs_lr_decay"], False, model_params, opt_params)
        for epoch in range(client_epoch):
            train(client_model, loss_name, criterion, device, train_loaders[client_id], optimizer, lr_scheduler, server_epoch, client_opt_params)
            
        for name, param in client_model.named_parameters():
            if param.requires_grad:
                #param_names.append(name)
                server_adapter_name = name.replace("{}".format(adapter_name), opt_params["server_name"])
                if output_layer_name and output_layer_name in name:
                    output_weights[server_adapter_name] += param.data / client_num
                else:
                    if server_adapter_name in adapter_weights:
                        row, col = param.data.shape
                        noise_param = add_noise(param.data, opt_params["privacy_noise"], opt_params["privacy_clip"], opt_params["sketch_size"])
                        print(name, param.data.norm().item(), noise_param.norm().item())
                        adapter_weights[server_adapter_name][:row, :col] += noise_param/client_num
                        #print(server_adapter_name, noise_param.norm().item())
                    else:
                        assert False
    model.set_adapter(opt_params["server_name"])
    if opt_params["lora_freeze_a"]:
        for n, p in model.named_parameters():
            if "lora_A" in n:
                p.requires_grad=False

    print("after privacy perturb")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if output_layer_name and output_layer_name in name:
                param.data = output_weights[name]
            elif name in adapter_weights:
                param.data = adapter_weights[name]
                print(name, param.data.norm())
            else:
                assert False
   
    synchronize_lora(model, opt_params["server_name"], truncate_last=True)
    """
    for name, param in model.named_parameters():
        if name in adapter_weights or name in output_weights:
            adapter_weights[name] = param #store the server param
        elif 'client' in name:
            import re 
            server_adapter_name = re.sub(r'client_\d+', 'server', name)
            adapter_weight_full = adapter_weights[server_adapter_name].data.clone() #assign the same param to client models
            if len(param.data.shape) == 2:
                row, col = param.data.shape
                param.data = adapter_weight_full[:row, :col]
            elif len(param.data.shape) == 1:
                param.data = adapter_weight_full
            else:
                assert False
        
        #if 'lora_A' in name or 'lora_B' in name:
        #    import re
        #    server_adapter_name = re.sub(r'client_\d+', 'server', name)
        #    param.data = adapter_weights[server_adapter_name].data
        
    """
    if server_lr_scheduler is not None:
        server_lr_scheduler.step()

    for group in server_optimizer.param_groups:
        print("server lr", group['lr'])



def privacy_lora_svd(model, loss_name, criterion, lora_rank, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, opt_params, model_params, server_epoch):
    from main import train
    if opt_params["fedlora_avg"] != "sb":
        model.set_adapter(opt_params["server_name"])
        if opt_params["lora_freeze_a"]:
            for n, p in model.named_parameters():
                if "lora_A" in n:
                    p.requires_grad=False
   
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
                adapter_weights[name] = param
    
    client_num, client_opt_name, client_epoch = opt_params["client_num"], opt_params["client_opt_name"], opt_params["client_epoch"]
    
    base_names = []
    base_weights = {}
    base_adapter_weights = {}
    base_adapter_names = {}
    from utilities import get_gpu_memory
    #get_gpu_memory()
    server_optimizer.zero_grad()
    
    for i in range(0, len(adapter_names), 2):
        lora_A_name, lora_B_name = adapter_names[i], adapter_names[i+1]
        base_weight_name = lora_A_name.replace("lora_A.{}".format(opt_params["server_name"]), "base_layer")
        base_adapter_names[base_weight_name] = [lora_A_name, lora_B_name]
    
    lora_params = {}

    if opt_params["client_partial"] < 1:
        client_num = int(opt_params["client_partial"] * client_num)
        client_selected = np.random.choice(opt_params["client_num"], client_num, replace=False)
    else:
        client_selected = np.arange(client_num)

    print("after local train on client: ", client_selected)
    for client_id in client_selected:
        adapter_name = "client_{}".format(client_id)
        model.set_adapter(adapter_name)
        client_model = model #alias
        #get_gpu_memory()
        client_model.train()
        optimizer, lr_scheduler, _= load_optimizer(client_opt_name, client_model, client_lr, opt_params["client_momentum"], opt_params["client_weight_decay"], opt_params["lr_decay"], opt_params["epochs_lr_decay"], False, model_params, opt_params)
        #client_opt_params = copy.deepcopy(opt_params)
        from utilities import get_model_size
        #print(get_model_size(client_model))
        opt_params["train_stats"] = False
        for epoch in range(client_epoch):
            train(client_model, loss_name, criterion, device, train_loaders[client_id], optimizer, lr_scheduler, server_epoch, opt_params)
        opt_params["train_stats"] = True
        
        for name, param in client_model.named_parameters():
            #print(name, param.shape)
            if param.requires_grad:
                server_adapter_name = name.replace("{}".format(adapter_name), opt_params["server_name"])
                if output_layer_name and output_layer_name in name:
                    output_weights[server_adapter_name] += param.data / client_num
                else:
                    lora_params[server_adapter_name] = add_noise(param.data, opt_params["privacy_noise"], opt_params["privacy_clip"], opt_params["sketch_size"]) / (client_num**0.5)

        #get_gpu_memory()
        for name in opt_params["server_params"]:
            if output_layer_name and output_layer_name in name:
                pass
            else:
                lora_A_name = name.replace("base_layer", "lora_A.{}".format(opt_params["server_name"]))
                lora_B_name = name.replace("base_layer", "lora_B.{}".format(opt_params["server_name"]))
                lora_A_param, lora_B_param = lora_params[lora_A_name], lora_params[lora_B_name]
                #lora_A_param, lora_B_param = torch.cat(lora_params[lora_A_name]+[lora_A_param], dim=0), torch.cat(lora_params[lora_B_name]+[-lora_B_param], dim=1)
                #opt_params["server_params"][name].grad -= (lora_B_param.to(torch.float16) @ lora_A_param.to(torch.float16)).T
                if name in base_weights:
                    base_weights[name] += (lora_B_param @ lora_A_param).T
                else:
                    base_weights[name] = (lora_B_param @ lora_A_param).T

    model.set_adapter(opt_params["server_name"])

    if server_lr_scheduler is not None:
        server_lr_scheduler.step()

    for group in server_optimizer.param_groups:
        print("server lr", group['lr'])

    truncate_err = 0
 
    for name, param in model.named_parameters():
        if name not in opt_params["server_params"]:
            # examine if this is the lora module base name
            continue
        if output_layer_name and output_layer_name in name:
            if param.requires_grad:
                param.data = output_weights[name].clone() #opt_params["server_params"][name].clone()
        else:
            error_feedback = 0
            double_matrix = base_weights[name] #opt_params["server_params"][name].data
            import scipy.sparse.linalg as sp
            cpu_matrix = (double_matrix + error_feedback).contiguous().cpu().numpy()
            U_truncate, S_truncate, Vh_truncate = sp.svds(cpu_matrix, k=lora_rank)
            U_truncate= torch.from_numpy(U_truncate.copy()).to(device)
            S_truncate= torch.sqrt(torch.from_numpy(S_truncate.copy()).to(device))
            print(S_truncate)
            Vh_truncate= torch.from_numpy(Vh_truncate.copy()).to(device)
            lora_A_name, lora_B_name = base_adapter_names[name]
            ratio = opt_params["fedlora_uba"]
            adapter_weights[lora_A_name].data = (U_truncate * S_truncate).T * ratio
            adapter_weights[lora_B_name].data = Vh_truncate.T * S_truncate / ratio

    from arch.lora import synchronize_lora
    synchronize_lora(model, opt_params["server_name"], truncate_last=False)

    from arch.lora import get_lora_norm, get_weight_norm
    get_lora_norm(adapter_weights)
