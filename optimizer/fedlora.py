import torch
from optimizer.load_optimizer import load_optimizer
import copy

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
        print("param norms: ", norm_A.item(), norm_B.item())
        train_graphs.lora_A_norm.append(norm_A.item())
        train_graphs.lora_B_norm.append(norm_B.item())
    
    client_num, client_opt_name, client_epoch = opt_params["client_num"], opt_params["client_opt_name"], opt_params["client_epoch"]
    
    base_names = []
    base_weights = {}
    base_adapter_weights = {}
    base_adapter_names = {}
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

    lora_params = {}
    aggregated_weights = {}
    for client_id in range(client_num):
        from main import train
        # update client models
        client_model = copy.deepcopy(model)
        client_model.train()
        optimizer, lr_scheduler, _= load_optimizer(client_opt_name, client_model, client_lr, opt_params["client_momentum"], opt_params["client_weight_decay"], opt_params["lr_decay"], opt_params["epochs_lr_decay"], False, model_params, opt_params)
        client_opt_params = copy.deepcopy(opt_params)
        client_opt_params["train_stats"] = False
        for epoch in range(client_epoch):
            train(client_model, loss_name, criterion, device, train_loaders[client_id], optimizer, lr_scheduler, server_epoch, client_opt_params)      
        for name, param in client_model.named_parameters():
            #print(name, param.shape)
            if param.requires_grad:
                #param_names.append(name)
                if output_layer_name in name:
                    output_weights[name] += param.data / client_num
                elif name in lora_params:
                    lora_params[name].append(param.data)
                else:
                    lora_params[name] = [param.data]

    adapter_weights_avg = {}
    for i in range(0, len(adapter_names), 2):
        # A: r * input; B: output * r
        lora_A_name, lora_B_name = adapter_names[i], adapter_names[i+1]
        lora_A_param, lora_B_param = torch.cat(lora_params[lora_A_name], dim=0), torch.cat(lora_params[lora_B_name], dim=1)
        lora_A_avg, lora_B_avg = torch.mean(torch.stack(lora_params[lora_A_name]), dim=0), torch.mean(torch.stack(lora_params[lora_B_name]), dim=0)
        adapter_weights_avg[lora_A_name], adapter_weights_avg[lora_B_name] = lora_A_avg, lora_B_avg

        lora_matrix = lora_B_param @ lora_A_param / client_num        
        base_weight_name = lora_A_name.replace("lora_A.default", "base_layer")
        aggregated_weights[base_weight_name] = lora_matrix

        U, S, Vh = torch.linalg.svd(lora_matrix, full_matrices=False)
        U_truncate, S_truncate, Vh_truncate = U[:, :lora_rank], S[:lora_rank], Vh[:lora_rank, :]
        
    if opt_params["train_stats"]:
        from utilities import project_to_orth_space
        train_graphs.fedlora_A_align.append(project_to_orth_space(lora_A_param.T, Vh_truncate.T)[-1].item())
        train_graphs.fedlora_B_align.append(project_to_orth_space(lora_B_param, U_truncate)[-1].item())
        print(train_graphs.fedlora_A_align[::5])
        print(train_graphs.fedlora_B_align[::5])
        print("======")
        from utilities import cosine_similarity_batch
        train_graphs.fedlora_A_cosine.append(cosine_similarity_batch(lora_A_param.T, torch.tile(lora_A_avg.T, (1,client_num)), ret_abs=True).item())
        train_graphs.fedlora_B_cosine.append(cosine_similarity_batch(lora_B_param, torch.tile(lora_B_avg, (1,client_num)), ret_abs=True).item())
        print(train_graphs.fedlora_A_cosine[::5])
        print(train_graphs.fedlora_B_cosine[::5])

        norm_A_diff, norm_B_diff = 0, 0 
        norm_A, norm_B = 0, 0 
        for name in adapter_weights:
            if 'lora_A' in name:
                norm_A += torch.norm(adapter_weights[name]) ** 2
                norm_A_diff += torch.norm(adapter_weights_avg[name] - adapter_weights[name]) ** 2
            elif 'lora_B' in name:
                norm_B += torch.norm(adapter_weights[name]) ** 2
                norm_B_diff += torch.norm(adapter_weights_avg[name] - adapter_weights[name]) ** 2
        print("param norms: ", norm_A.item(), norm_B.item(), norm_A_diff.item(), norm_B_diff.item())

    server_optimizer.zero_grad()
    #for name, param in model.named_parameters():
    grad_norm = 0
    for name in opt_params["server_params"]:
        if name in aggregated_weights.keys():
            #param.requires_grad = True # going to update dense weight
            if opt_params["fedlora_avg"] == "svd_v2":
                lora_A_name = name.replace("base_layer", "lora_A.default")
                A = adapter_weights[lora_A_name]
                spectral_norm_A = torch.linalg.matrix_norm(A, ord=2) ** 2
                print("spectral norm A:", spectral_norm_A)
                opt_params["server_params"][name].grad = (base_adapter_weights[name] - aggregated_weights[name]).T / spectral_norm_A
            else:
                opt_params["server_params"][name].grad = (base_adapter_weights[name] - aggregated_weights[name]).T
            grad_norm += torch.linalg.norm(opt_params["server_params"][name].grad) ** 2
        elif output_layer_name in name:
            opt_params["server_params"][name].grad = base_weights[name].data - output_weights[name]
    
    if opt_params["train_stats"]:
        train_graphs.grad_norm.append(grad_norm.item())
        print("grad norm:", train_graphs.grad_norm[-1])
    server_optimizer.step()

    for name, param in model.named_parameters():
        if name in aggregated_weights.keys():
            #param.requires_grad = False # turn off updates in dense weights
            if opt_params["fedlora_avg"] in ["svd", "svd_v2"]:
                # SVD
                U, S, Vh = torch.linalg.svd(opt_params["server_params"][name].data, full_matrices=False)
                #print(S[:lora_rank+5])
                U_truncate, S_truncate, Vh_truncate = U[:, :lora_rank], torch.sqrt(S[:lora_rank]), Vh[:lora_rank, :]
                #print(name)
                #print(U.shape, Vh.shape)
                #print(S)
                lora_A_name, lora_B_name = base_adapter_names[name]
                #print(lora_A_name, lora_B_name)
                #print("====")
                #print(adapter_weights[lora_A_name].data.shape, adapter_weights[lora_B_name].data.shape)
                adapter_weights[lora_A_name].data = (U_truncate * S_truncate).T
                adapter_weights[lora_B_name].data = Vh_truncate.T * S_truncate
                #print(opt_params["server_params"][name].shape, U_truncate.shape, Vh_truncate.shape)
                #print(lora_A_name, adapter_weights[lora_A_name].data.shape)
                #print(lora_B_name, adapter_weights[lora_B_name].data.shape)
                #print(adapter_weights[lora_A_name].data.shape, adapter_weights[lora_B_name].data.shape)
            elif opt_params["fedlora_avg"] == "sketch":
                # sketching
                m, _ = param.shape
                #Q = torch.rand(m, lora_rank).to(param) / (lora_rank**0.5)
                Q = torch.normal(0, 1/(lora_rank**0.5), (m, lora_rank)).to(param)
                
                lora_A_name, lora_B_name = base_adapter_names[name]
                adapter_weights[lora_A_name].data = Q.T
                adapter_weights[lora_B_name].data = param.detach().T @ Q
                #print(param.shape, lora_A_name, adapter_weights[lora_A_name].data.shape, adapter_weights[lora_B_name].data.shape)
                #adapter_weights[lora_B_name].data = param.detach() @ Q
            elif opt_params["fedlora_avg"] == "sketch_v2":
                """from paper https://arxiv.org/pdf/1609.00048"""
                A = opt_params["server_params"][name].data # m*n
                row_A, col_A = A.shape[0], A.shape[1]
                col_k, row_l = lora_rank, lora_rank
                Omega, Phi = torch.randn(col_A, col_k).to(A), torch.randn(row_l, row_A).to(A)
                Omega, _ = torch.linalg.qr(Omega)
                Phi = torch.linalg.qr(Phi.T)[0].T
                Y, W = A @ Omega, Phi @ A
                Q, _ = torch.linalg.qr(Y) # m*k
                U, T = torch.linalg.qr(Phi @ Q)
                X = torch.linalg.solve_triangular(T, U.T @ W, upper=True)
                lora_A_name, lora_B_name = base_adapter_names[name]
                adapter_weights[lora_A_name].data = Q.T
                adapter_weights[lora_B_name].data = X.T

        elif output_layer_name in name:
            param.data = opt_params["server_params"][name]