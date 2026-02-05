import torch
from optimizer.load_optimizer import load_optimizer
import copy
import torch.nn.functional as F
from arch.lora import synchronize_lora
import numpy as np

def compute_adapter_weight(model_name, lora_A_param, lora_B_param):
    if model_name in ["google/vit-base-patch16-224-in21k", "roberta-base"]:
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
    truncate_err, truncate_err_ratio = compute_truncate_err(model, adapter_weights, client_num, opt_params["model_name"], opt_params["server_name"])
    server_optimizer.zero_grad()

    train_graphs.truncate_err.append(truncate_err)
    train_graphs.truncate_err_ratio.append(truncate_err_ratio)
    print("Truncation Error: ", train_graphs.truncate_err[-1])
    print("Truncation Error Ratio: ", train_graphs.truncate_err_ratio[-1])

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


def federated_frlora(model, loss_name, criterion, lora_rank, train_graphs, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, opt_params, model_params, server_epoch):
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
    for client_id in range(client_num):
        # update client models
        adapter_name = "client_{}".format(client_id)
        model.set_adapter(adapter_name)
        client_model = model #alias

        client_model.train()
        optimizer, lr_scheduler, _= load_optimizer(client_opt_name, client_model, client_lr, opt_params["client_momentum"], opt_params["client_weight_decay"], opt_params["lr_decay"], opt_params["epochs_lr_decay"], False, model_params, opt_params)
        #vector_to_parameters(old_params, client_model.parameters())
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
                        adapter_weights[server_adapter_name][:row, :col] += param.data/client_num
                    else:
                        assert False
    
    
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
    """
    model.merge_adapter([opt_params["server_name"]])
    model.merge_adapter(["fr_save_neg_init"])

    #synchronize_lora(model, opt_params["server_name"], truncate_last=True)
    # reinitialize the client lora params
    synchronize_lora(model, "fr_save_init", truncate_last=True)

    #reset server adapter to fr_save_init -- prepare for the next round
    from arch.lora import synchronize_lora_server
    synchronize_lora_server(model, "fr_save_init", opt_params["server_name"], truncate_last=True)

    if server_lr_scheduler is not None:
        server_lr_scheduler.step()

    for group in server_optimizer.param_groups:
        print("server lr", group['lr'])
    

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