import os
import sys

import torch
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import gc
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models

from tqdm import tqdm
from collections import OrderedDict
#os.environ["SCIPY_USE_PROPACK"] =  "1"
from scipy.sparse.linalg import svds

from IPython import embed

from graphs import graphs
from path_manage import get_running_directory, get_directory, continue_training
from optimizer.sam import disable_running_stats, enable_running_stats
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from optimizer.load_optimizer import load_optimizer


from utilities import map_update, dict_to_, graph_update
# setting path
#current_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir = os.path.dirname(current_dir)
#sys.path.append(parent_dir)
#print(parent_dir)
#from utilities import get_hessian_eigenvalues_weight_decay, get_hessian_eigenvalues

def srht_sketch(P, H, D):
    return 

def sparse_skethc(m,n,s):
    assert s < m
    non_zero_index = torch.randint(low=0, high=m, size=(s, n))
    return 

def federated_train_1(model, loss_name, criterion, device, num_classes, train_loaders, server_optimizer, exp_avg, exp_avg_sq, opt_params, server_epoch):
    client_num, client_opt_name, client_lr, client_epoch = opt_params["client_num"], opt_params["client_opt_name"], opt_params["client_lr"], opt_params["client_epoch"]
    momentum, momentum_v = opt_params["server_momentum"], 0.999
    vector_m, vector_v = 0, 0
    import copy
    import math
    from utilities import vector_to_grads
    
    #from utilities import state_dict_to_vector, vector_to_state_dict
    # initialize client models, optimizers
    old_params = parameters_to_vector(model.parameters())
    device = old_params.get_device()
    p = len(old_params)
    #sketch_size = int(opt_params["sketch_size"] * p)
    sketch_size = opt_params["sketch_size"]
    exp_avg = exp_avg if exp_avg is not None else torch.zeros(p).to(device)
    exp_avg_sq = exp_avg_sq if exp_avg_sq is not None else torch.zeros(p).to(device)

    #sketch_matrix_m, sketch_matrix_v = torch.randn(sketch_size, p).to(device) / (sketch_size**0.5), torch.randn(sketch_size, p).to(device) / (sketch_size**0.5)
    from hadamard_transform import hadamard_transform, pad_to_power_of_2 
    p_pad = pow(2, math.ceil(math.log(p)/math.log(2)))
    #for i in range(200):
    if sketch_size != -1:
        D = (((torch.randn(p_pad, dtype=torch.float32) > 0).float() - 0.5) * 2).to(device)
        sample_rows = torch.stack([torch.arange(sketch_size), torch.randperm(p_pad)[:sketch_size]]).to(device)
        sub_sample_row = torch.sparse_coo_tensor(sample_rows, torch.ones(sketch_size).to(device), [sketch_size, p_pad]) * ((p_pad/sketch_size)**0.5)

        D_sq = (((torch.randn(p_pad, dtype=torch.float32) > 0).float() - 0.5) * 2).to(device)
        sample_rows_sq = torch.stack([torch.arange(sketch_size), torch.randperm(p_pad)[:sketch_size]]).to(device)
        sub_sample_row_sq = torch.sparse_coo_tensor(sample_rows_sq, torch.ones(sketch_size).to(device), [sketch_size, p_pad]) * ((p_pad/sketch_size)**0.5)

    
    #running_stats = {}
    for client_id in range(client_num):
        # update client models
        client_model = copy.deepcopy(model)

        client_model.train()
        optimizer, lr_scheduler, _= load_optimizer(client_opt_name, client_model, client_lr, opt_params["client_momentum"], weight_decay, lr_decay, epochs_lr_decay, model_params, **opt_params)
        #vector_to_parameters(old_params, client_model.parameters())
        for epoch in range(client_epoch):
            train(client_model, loss_name, criterion, device, num_classes, train_loaders[client_id], optimizer, lr_scheduler, server_epoch, opt_params)
        """
        for name in client_model.state_dict():
            if 'running_mean' in name or 'running_var' in name:
                #print(client_model.state_dict()[name])
                if name not in running_stats:
                    running_stats[name] = client_model.state_dict()[name] / client_num
                else:
                    running_stats[name] += client_model.state_dict()[name] / client_num
        """
        """
        sketch_updates = models[client_id].state_dict().copy()
        if sketch_size:
            for name, param in models[]
                sketch_updates[name] = rand_dict[name] @ param
        """
        new_params = parameters_to_vector(client_model.parameters())

        if sketch_size == -1:
            vector_m += (old_params - new_params).detach()
            vector_v += ((old_params - new_params) ** 2).detach()
        else:
            new_params_pad = pad_to_power_of_2((old_params - new_params).detach())

            hadamard_params_pad = hadamard_transform(D*new_params_pad)
            vector_m += sub_sample_row @ hadamard_params_pad

            hadamard_params_pad = hadamard_transform(D_sq*(new_params_pad ** 2))
            vector_v += sub_sample_row_sq @ hadamard_params_pad
        #vector_m += sketch_matrix_m @ (old_params - new_params).detach()
        #vector_v += sketch_matrix_v @ ((old_params - new_params) ** 2).detach()
        
        #vector_m += (old_params - new_params).detach()
        #vector_v += ((old_params - new_params) ** 2).detach()
        #vector_m += sub_sample_row @ hadamard_params_pad
        """
        x_recovered = 0
        for i in range(1, 200):
            D = (((torch.randn(p_pad, dtype=torch.float32) > 0).float() - 0.5) * 2).to(device)
            sample_rows = torch.stack([torch.arange(sketch_size), torch.randperm(p_pad)[:sketch_size]]).to(device)
            sub_sample_row = torch.sparse_coo_tensor(sample_rows, torch.ones(sketch_size).to(device), [sketch_size, p_pad]) * ((p_pad/sketch_size)**0.5)
            
            hadamard_params_pad = hadamard_transform(D*new_params_pad)
            vector_m = sub_sample_row @ hadamard_params_pad

            desketched_vec = sub_sample_row.T @ vector_m
            desketched_vec = hadamard_transform(desketched_vec) * D

            x_recovered += desketched_vec
            print(i, torch.norm(x_recovered[:p]/i - new_params_pad[:p]))
        """
    
    # server update
    
    #vector_m, vector_v = vector_m / client_num, vector_v / client_num
    #vector_m = sketch_matrix_m.T @ vector_m
    #vector_v = sketch_matrix_v.T @ vector_v
    if opt_params["server_opt_name"] in ['adam', 'sgdm']:
        vector_m = vector_m / client_num
        vector_v = vector_v / client_num
        if sketch_size != -1:
            vector_m = sub_sample_row.T @ vector_m
            vector_m = hadamard_transform(vector_m) * D
            print("sketch error:", torch.norm(vector_m - new_params_pad), torch.norm(new_params_pad))
            vector_m = vector_m[:p]

            vector_v = sub_sample_row_sq.T @ vector_v
            vector_v = hadamard_transform(vector_v) * D_sq
            print("sketch error:", torch.norm(vector_v - new_params_pad**2), torch.norm(new_params_pad**2))
            vector_v = vector_v[:p]

        server_optimizer.zero_grad()
        vector_to_grads(vector_m, model.parameters())
        server_optimizer.step()

        exp_avg = momentum * exp_avg + (1-momentum) * vector_m
        exp_avg_sq = momentum_v * exp_avg_sq + (1-momentum_v) * vector_v
        print("vector_m:", torch.min(vector_m).item(), torch.max(vector_m).item())
        real_update = exp_avg / (1-momentum**server_epoch)
        print("exp_avg: ", torch.min(real_update).item(), torch.max(real_update).item())
        #real_update =  (exp_avg / (1-momentum**server_epoch)) / torch.sqrt(F.relu(exp_avg_sq / (1-momentum_v**server_epoch)) + 0.005)
        #print("update:", torch.min(exp_avg / (1-momentum**server_epoch)).item())
        if opt_params["server_opt_name"] == 'adam':
            new_params = old_params - lr * (exp_avg / (1-momentum**server_epoch)) / torch.sqrt(F.relu(exp_avg_sq / (1-momentum_v**server_epoch)) + 0.005)
        elif opt_params["server_opt_name"] == 'sgdm':
            new_params = old_params - lr * exp_avg / (1-momentum**server_epoch)
    elif 'gd' == opt_params["server_opt_name"]:
        vector_m = vector_m / client_num
        if sketch_size != -1:
            vector_m = sub_sample_row.T @ vector_m
            vector_m = hadamard_transform(vector_m) * D
            print("sketch error:", torch.norm(vector_m - new_params_pad), torch.norm(new_params_pad))
            vector_m = vector_m[:p]
        new_params = old_params - lr * vector_m
    vector_to_parameters(new_params.detach(), model.parameters())
    return exp_avg, exp_avg_sq

def federated_train(model, loss_name, criterion, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, opt_params, server_epoch):
    client_num, client_opt_name, client_epoch = opt_params["client_num"], opt_params["client_opt_name"], opt_params["client_epoch"]
    #client_num, client_opt_name, client_lr, client_epoch = opt_params["client_num"], opt_params["client_opt_name"], opt_params["client_lr"], opt_params["client_epoch"]
    momentum, momentum_v = opt_params["server_momentum"], 0.999
    vector_m, vector_v = 0, 0
    vector_m_true = 0
    vector_m_norm = []
    import copy
    import math
    from utilities import vector_to_grads, vector_to_grads_sq

    #from utilities import state_dict_to_vector, vector_to_state_dict
    # initialize client models, optimizers
    old_params = parameters_to_vector(model.parameters())
    device = old_params.get_device()
    p = len(old_params)
    #sketch_size = int(opt_params["sketch_size"] * p)
    sketch_size = opt_params["sketch_size"]

    #sketch_matrix_m, sketch_matrix_v = torch.randn(sketch_size, p).to(device) / (sketch_size**0.5), torch.randn(sketch_size, p).to(device) / (sketch_size**0.5)
    from hadamard_transform import hadamard_transform, pad_to_power_of_2 
    p_pad = pow(2, math.ceil(math.log(p)/math.log(2)))
    #for i in range(200):
    if sketch_size != -1:
        D = (((torch.randn(p_pad, dtype=torch.float32) > 0).float() - 0.5) * 2).to(device)
        sample_rows = torch.stack([torch.arange(sketch_size), torch.randperm(p_pad)[:sketch_size]]).to(device)
        sub_sample_row = torch.sparse_coo_tensor(sample_rows, torch.ones(sketch_size).to(device), [sketch_size, p_pad]) * ((p_pad/sketch_size)**0.5)

        D_sq = (((torch.randn(p_pad, dtype=torch.float32) > 0).float() - 0.5) * 2).to(device)
        sample_rows_sq = torch.stack([torch.arange(sketch_size), torch.randperm(p_pad)[:sketch_size]]).to(device)
        sub_sample_row_sq = torch.sparse_coo_tensor(sample_rows_sq, torch.ones(sketch_size).to(device), [sketch_size, p_pad]) * ((p_pad/sketch_size)**0.5)

    
    #running_stats = {}
    for client_id in range(client_num):
        # update client models
        client_model = copy.deepcopy(model)

        client_model.train()
        optimizer, lr_scheduler, _= load_optimizer(client_opt_name, client_model, client_lr, opt_params["client_momentum"], opt_params["client_weight_decay"], lr_decay, epochs_lr_decay, False, model_params, **opt_params)
        #vector_to_parameters(old_params, client_model.parameters())
        for epoch in range(client_epoch):
            train(client_model, loss_name, criterion, device, train_loaders[client_id], optimizer, lr_scheduler, server_epoch, opt_params)
            
        new_params = parameters_to_vector(client_model.parameters())
        if opt_params["server_opt_name"] == "clip_sgd":
            vector_m_norm.append(torch.norm(old_params - new_params).item())

        if sketch_size == -1:
            #param_norm = torch.norm(old_params - new_params).detach()
            vector_m += (old_params - new_params).detach() #* min(1, opt_params["clip_tau"] / param_norm.item())
            #if opt_params["clip_tau"] / param_norm.item() < 1: print("clip")
            vector_v += ((old_params - new_params) ** 2).detach()
            #vector_m_norm.append(torch.norm(old_params - new_params).item())
        else:
            vector_m_true += (old_params - new_params).detach()
            new_params_pad = pad_to_power_of_2((old_params - new_params).detach())

            hadamard_params_pad = hadamard_transform(D*new_params_pad)
            vector_m += sub_sample_row @ hadamard_params_pad

            #hadamard_params_pad = hadamard_transform(D_sq*(new_params_pad ** 2))
            #vector_v += sub_sample_row_sq @ hadamard_params_pad
            hadamard_params_pad = hadamard_transform(D*(new_params_pad ** 2))
            vector_v += sub_sample_row @ hadamard_params_pad

    vector_m = vector_m / client_num
    vector_v = vector_v / client_num
    if sketch_size != -1:
        vector_m = sub_sample_row.T @ vector_m
        vector_m = hadamard_transform(vector_m) * D
        print("sketch error:", torch.norm(vector_m - new_params_pad), torch.norm(new_params_pad))
        vector_m = vector_m[:p]

        #vector_v = sub_sample_row_sq.T @ vector_v
        #vector_v = hadamard_transform(vector_v) * D_sq
        vector_v = sub_sample_row.T @ vector_v
        vector_v = hadamard_transform(vector_v) * D
        print("sketch error:", torch.norm(vector_v - new_params_pad**2), torch.norm(new_params_pad**2))
        vector_v = vector_v[:p]

    server_optimizer.zero_grad()
    if opt_params["server_opt_name"] == "clip_sgd":
        vector_to_grads(vector_m * min(1, opt_params["clip_tau"] / np.mean(vector_m_norm).item()), model.parameters())
        print(np.mean(vector_m_norm).item())
        if opt_params["clip_tau"] / np.mean(vector_m_norm).item() < 1:
            print("clipped")
        else:
            print("no clip")
        """
        if opt_params["clip_tau"] < np.max(vector_m_norm).item():
            print("clipped")
            vector_to_grads(vector_m * min(1, opt_params["clip_tau"] / np.max(vector_m_norm).item()), model.parameters())
        else:
            print("no clip")
        """
        #vector_to_grads(vector_m, model.parameters())
    else:
        vector_to_grads(vector_m, model.parameters())
        vector_to_grads_sq(vector_v, model.parameters())
    server_optimizer.step()

    if server_lr_scheduler is not None:
        server_lr_scheduler.step()

    #for group in server_optimizer.param_groups:
    #    print(group['lr'])

    train_graphs.pseudo_grad_norm.append(vector_m_norm)



def train(model, loss_name, criterion, device, train_loader, optimizer, lr_scheduler, epoch, opt_params):
    #old_params = parameters_to_vector(model.parameters())
    model.train()
    
    pbar = tqdm(total=len(train_loader), position=0, leave=True)

    # initialize training statistics
    accuracy = 0
    loss = torch.FloatTensor([0])
    track_train_stats = {}

    for batch_idx, input in enumerate(train_loader, start=1):
        if not opt_params["hf_model"]:
            data, target = input
            if data.shape[0] != batch_size:
                continue

            data, target = data.to(device), target.to(device)
            if opt_params["mixup"] == "cut":
                data, target, rand_target, lambda_= cutmix((data, target))

            if opt_name != "gd":
                optimizer.zero_grad()
            out = model(data)

            if opt_params["mixup"] == "none":
                loss = criterion(out, target)
            elif opt_params["mixup"] == "cut":
                loss = criterion(out, target) * lambda_ + criterion(out, rand_target)*(1.-lambda_)
            else:
                assert False
        else:
            #print(input)
            dict_to_(input, device)
            target = input["labels"].to(device)
            if opt_name != "gd":
                optimizer.zero_grad()
            output = model(**input)
            loss, out = output.loss, output.logits

        if opt_params["forward_backward"]:
            if opt_name == "adahessian":
                loss.backward(create_graph=True)
            else:
                loss.backward()

            if compute_acc:
                if out.dim() > 1:
                    accuracy = torch.mean((torch.argmax(out,dim=1)==target).float()).item()
                else:
                    accuracy = torch.mean((out*target > 0).float()).item()

        
        if opt_name in ["sam", "sam_on"]:
            if opt_params["train_stats"]:
                from analysis.grad_norm import get_grad_norm
                grad_norm = get_grad_norm(model, ascent=True)
                map_update(track_train_stats, grad_norm, reduction = "append")
            train_stats = optimizer.first_step(zero_grad=True)
            print(train_stats['ascent_step_cos'])
            if not (epoch == 1 and batch_idx==1):
                map_update(track_train_stats, train_stats, reduction="sum")
            # second forward-backward step
            disable_running_stats(model)
            if not opt_params["hf_model"]:
                out = model(data)
                #loss = criterion(out, F.one_hot(target, num_classes=num_classes).float()) * num_classes
                loss = criterion(out, target).float()
                #loss.backward()
                #train_stats = optimizer.second_step(zero_grad=True)
            else:
                output = model(**input)
                loss, out = output.loss, output.logits
            loss.backward()
            if opt_params["train_stats"]:
                from analysis.grad_norm import get_grad_norm
                grad_norm = get_grad_norm(model)
                map_update(track_train_stats, grad_norm, reduction = "append")
            train_stats = optimizer.second_step(zero_grad=True)

            map_update(track_train_stats, train_stats, reduction="sum")
            #cos_descent_ascent += optimizer.second_step(zero_grad=True)
            enable_running_stats(model)
        elif opt_name == "replay_sam":
            disable_running_stats(model)
            if not (epoch == 1 and batch_idx==1):
                train_stats = optimizer.first_step(zero_grad=True)
                map_update(track_train_stats, train_stats, reduction="sum")
            # second forward-backward step
            out = model(data)
            loss = criterion(out, target).float()
            loss.backward()
            optimizer.second_step(zero_grad=True)
            enable_running_stats(model)
        elif opt_name.startswith("look_sam"):
            disable_running_stats(model)
            if not (batch_idx-1) % 10:
            #if True:
                optimizer.first_step(zero_grad=True)
                out = model(data)
                loss = criterion(out, target).float()
                loss.backward()
                if "v2" in opt_name:
                    optimizer.second_step_v2(zero_grad=True)
                else:
                    optimizer.second_step(zero_grad=True)
                #sys.exit()
            else:
                if "v2" in opt_name:
                    optimizer.normal_step_v2(zero_grad=True)
                else:
                    optimizer.normal_step(zero_grad=True)
            enable_running_stats(model)
        elif opt_name == "alternate_sam":
            disable_running_stats(model)
            if opt_params["forward_backward"]: # in odd step
                optimizer.odd_step(zero_grad=True)
                opt_params["forward_backward"] = False
            else:
                optimizer.even_first_step(zero_grad=True)
                out = model(data)
                loss = criterion(out, target).float()
                loss.backward()
                optimizer.even_second_step(zero_grad=True)
                opt_params["forward_backward"] = True
            enable_running_stats(model)
        elif opt_name == "alternate_sam_v2":
            disable_running_stats(model)
            if opt_params["forward_backward"]: # in odd step
                if opt_params["opt_first_step"]:
                    optimizer.odd_first_step(zero_grad=True)
                    opt_params["opt_first_step"] = False
                else:
                    optimizer.odd_second_step(zero_grad=True)
                    opt_params["forward_backward"] = False
                    opt_params["opt_first_step"] = True
            else:
                optimizer.even_first_step(zero_grad=True)
                out = model(data)
                loss = criterion(out, target).float()
                loss.backward()
                optimizer.even_second_step(zero_grad=True)
                opt_params["forward_backward"] = True
            enable_running_stats(model)
        elif opt_name == "alternate_sam_v3":
            disable_running_stats(model)
            if epoch == 1 and batch_idx == 1:
                optimizer.initial_first_step(zero_grad=True)
                out = model(data)
                loss = criterion(out, target).float()
                loss.backward()
                optimizer.even_second_step(zero_grad=True)
            else:
                if opt_params["forward_backward"]: # in odd step
                    optimizer.odd_step(zero_grad=True)
                    opt_params["forward_backward"] = False
                else:
                    optimizer.even_first_step(zero_grad=True)
                    out = model(data)
                    loss = criterion(out, target).float()
                    loss.backward()
                    optimizer.even_second_step(zero_grad=True)
                    opt_params["forward_backward"] = True
            enable_running_stats(model)
        elif opt_name == "goldstein":
            gold_iters = 0
            while gold_iters < 1:
                gold_iters += 1
                optimizer.first_step(zero_grad=True)
                disable_running_stats(model)
                out = model(data)
                #loss = criterion(out, F.one_hot(target, num_classes=num_classes).float()) * num_classes
                loss = criterion(out, target).float()
                loss.backward()
                optimizer.second_step(zero_grad=False)
            optimizer.third_step(zero_grad=True)
            enable_running_stats(model)
        elif opt_name == "norm-sgd":
            if loss_name == 'MSELoss':
                optimizer.step(loss=loss)
            elif loss_name in ['CrossEntropyLoss', 'BCELoss']:
                optimizer.step(accuracy=accuracy)
            else:
                raise NotImplementedError
        elif opt_name == "gd":
            pass
        else:
            if opt_params["train_stats"]:
                from analysis.grad_norm import get_grad_norm
                grad_norm = get_grad_norm(model)
                map_update(track_train_stats, grad_norm, reduction = "append")
            optimizer.step()
                
        pbar.update(1)
        
        pbar.set_description(
            'Train    Epoch: {} [{}/{} ({:.0f}%)]   '
            'Batch Loss: {:.6f}  '
            'Batch Accuracy: {:.6f}'.format(
                epoch,
                batch_idx,
                len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item(),
                accuracy))
        
        #if debug and batch_idx > 20:
        if debug and batch_idx > 10:
           break

    if opt_name == "gd":
        optimizer.step()
        optimizer.zero_grad()
    
    pbar.close()

    #if opt_params["scheduler_name"] != 'none':
    if lr_scheduler is not None:
        lr_scheduler.step()

    #deal with training track statistics

    if opt_params["train_stats"]:
        graph_update(train_graphs, track_train_stats, normalizer=len(train_loader))
    """
    if "cos_descent_ascent" in track_train_stats:
        train_graphs.cos_descent_ascent.append(track_train_stats["cos_descent_ascent"] / len(train_loader))
    if "progress_dir" in track_train_stats:
        train_graphs.progress_dir.append(track_train_stats["progress_dir"] / len(train_loader))
    if "ascent_semi_cos" in track_train_stats:
        train_graphs.ascent_semi_cos.append(track_train_stats["ascent_semi_cos"] / len(train_loader))
    if "descent_norm" in track_train_stats:
        train_graphs.descent_norm.append(track_train_stats["descent_norm"] / len(train_loader))
    if "ascent_step_diff" in track_train_stats:
        train_graphs.ascent_step_diff.append(track_train_stats["ascent_step_diff"] / len(train_loader))
    if "descent_step_diff" in track_train_stats:
        train_graphs.descent_step_diff.append(track_train_stats["descent_step_diff"] / len(train_loader))
    if "grad_norm" in track_train_stats:
        print(len(track_train_stats["grad_norm"]))
        train_graphs.grad_norm.append(track_train_stats["grad_norm"])
        #return torch.norm(old_params - parameters_to_vector(model.parameters())).item()
    """

def analysis(graphs, analysis_list, model, model_name, criterion_summed, device, num_classes, compute_acc, train_loader, test_loader, analysis_loader, analysis_test_loader, opt_params, analysis_params):    
    if 'loss' in analysis_list:
        from analysis.loss import compute_loss
        compute_loss(graphs, model, loss_name, criterion, criterion_summed, device, num_classes, train_loader, test_loader, opt_params, compute_acc, compute_model_output='output' in analysis_list)

    if 'eigs' in analysis_list:
        from analysis.eigs import compute_eigenvalues
        compute_eigenvalues(graphs, model, criterion_summed, weight_decay, analysis_loader, analysis_test_loader, num_classes, device, use_hf_model=opt_params["hf_model"])

    if 'gn_eigs' in analysis_list:
        from analysis.eigs import compute_gn_eigenvalues
        compute_gn_eigenvalues(graphs, loss_name, model, analysis_loader, num_classes, device)
    
    if 'nc' in analysis_list:
        from analysis.nc import get_nc_statistics
        get_nc_statistics(graphs, model, features, classifier, loss_name, criterion_summed, weight_decay, num_classes, analysis_loader, analysis_test_loader, device, debug=False)

    if 'weight_norm' in analysis_list:
        from analysis.weight_norm import get_min_weight_norm, get_min_weight_norm_with_g, get_grad_loss_ratio
        get_min_weight_norm(graphs, model, C=num_classes, model_name=model_name)
        get_min_weight_norm_with_g(graphs, model, C=num_classes, model_name=model_name)
        get_grad_loss_ratio(graphs, model, loss_name, analysis_loader, criterion, criterion_summed, num_classes, device)

    if 'adv_eigs' in analysis_list:
        from analysis.adv_eigs import compute_adv_eigenvalues
        compute_adv_eigenvalues(graphs, model, criterion_summed, analysis_params["adv_eta"], weight_decay, analysis_loader, num_classes, device)

    if 'align' in analysis_list:
        from analysis.alignment import compute_weight_signal_alignment
        compute_weight_signal_alignment(graphs, model, analysis_params["signal"], analysis_params["signal_patch_index"], train_loader)

    if 'linear' in analysis_list:
        from analysis.alignment import get_linear_coefs
        get_linear_coefs(graphs, model)

    if 'activation' in analysis_list:
        assert model_name in ["conv_with_last", "conv_fixed_last", "scalarized_conv"]
        from analysis.activation import get_activation_pattern
        get_activation_pattern(graphs, model, device, train_loader)

    if 'diagonal' in analysis_list:
        assert model_name in ["scalarized_conv"]
        from analysis.diagonal import get_diagonal_coef, get_diagonal_invariate
        get_diagonal_coef(graphs, model, device, train_loader)
        get_diagonal_invariate(graphs, model, device, train_loader)

    if 'attention_map' in analysis_list:
        from analysis.attention_map import get_attention_map
        get_attention_map(graphs, model, device, vit_patch_size)
    """
    for i in range(2):
        print(model.module_list[i].weight)
    print(model.output_layer.weight)
    """

class features:
    pass

def hook(self, input, output):
    features.value = input[0].clone()

    
if __name__ == "__main__":
    DATASETS = ["spurious", "cifar", "cifar100", "mnist", "emnist", "mnist_cifar", "spurious-2d", "multi-view", "secondary_feature", "multi-view-orthogonal", "orthogonal", "scalarized", "weight_norm_teacher", "glue"]
    MODELS = ["2-mlp-sim-bn", "2-mlp-sim-ln", "conv_fixed_last", "conv_with_last", "weight_norm_torch", "scalarized_conv", "weight_norm", "weight_norm_v2", "weight_norm_width_scale", "resnet18", "resnet_fixup", "resnet_gn", "WideResNet", "WideResNet_WN_woG", "ViT", "emnistcnn", "google-bert/bert-base-cased", "google/vit-base-patch16-224-in21k", "dino_vit_small"]
    INIT_MODES = ["O(1)", "O(1/sqrt{m})"]
    LOSSES = ['MSELoss', 'CrossEntropyLoss', 'BCELoss']
    OPTIMIZERS = ['gd', 'goldstein','sam', 'sam_on', 'sgd', 'norm-sgd','adam', 'adamw', 'federated','replay_sam', 'alternate_sam', 'alternate_sam_v2', 'alternate_sam_v3', 'look_sam', 'look_sam_v2', 'adahessian', 'sketch_adam']
    BASE_OPTIMIZERS = ['sgd','adam']

    parser = argparse.ArgumentParser(description="Train Configuration.")
    parser.add_argument("--debug", type=bool, default=False, help="only run first 20 batches per epoch if set to True")
    parser.add_argument("--dataset",  type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("--model",  type=str, choices=MODELS, help="which model to train")

    #model
    parser.add_argument("--width", type=int, default=512, help="network width for weight norm or number of filters in convnets")
    parser.add_argument("--depth", type=int, default=7, help="network depth")
    parser.add_argument("--width_factor", type=int, default=8, help="width factor for WideResNet")
    parser.add_argument("--init_mode",  type=str, default="O(1)", choices=INIT_MODES, help="Initialization Mode")
    parser.add_argument("--basis_var", type=float, default=5, help="variance for initialization")
    parser.add_argument("--wn_scale", type=float, default=10, help="scaling coef for weight_norm model")

    #parser.add_argument("--vit_patch_size", type=int, default=8, help="patch size for ViT")
    parser.add_argument("--vit_patch_size", type=int, default=8, help="patch size for ViT")

    parser.add_argument("--loss",  type=str, choices=LOSSES, help="Training Loss")
    parser.add_argument("--opt",  type=str, choices=OPTIMIZERS, help="Training Optimizer")
    parser.add_argument("--scheduler",  type=str, choices=['none', 'cosine', 'multistep'], default='none', help="LR Scheduler")
    parser.add_argument('--analysis', nargs='+', type=str, help="quantities that will be analyzed")
    parser.add_argument("--lr", type=float, help="the learning rate")
    parser.add_argument("--lr_decay", type=float, default=1, help="the learning rate decat. default: no decay")
    parser.add_argument("--lr_min", type=float, default=1e-5, help="lr min for cosine scheduler")
    parser.add_argument("--momentum", type=float, default=0.0, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=0, help ="weight decay")
    parser.add_argument("--epoch", type=int, help="total training epoches")
    parser.add_argument("--batch_size", type=int, help="batch size in training, also the number of samples in analysis dataset")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="label_smoothing param")
    parser.add_argument("--log_interval", type=int, default=200, help="do analysis every $ of epochs")
    parser.add_argument("--train_stats", type=bool, default=False, help="track stats along training process. Behavior depends on the specific training algorithm.")

    
    # data
    parser.add_argument("--mixup", type=str, choices=["cut","none","mix"], default="none", help="mixup strategy for data augmentation")
    parser.add_argument("--sp_train_size", type=int, default=-1, help="training size for spurious dataset")
    parser.add_argument("--sp_feat_dim", type=int, default=20, help="dimension for spurious data")
    parser.add_argument("--sp_patch_dim", type=int, default=20, help="patch dimension for 2d spurious data")
    parser.add_argument("--augment", type=int, default=0, help="augment pattern")

    # optimizer hyperparameters
    parser.add_argument("--base_opt", type=str, default="sgd", choices=BASE_OPTIMIZERS, help="base optimizer for sam/norm-sgd optimizer")
    parser.add_argument("--sam_rho", type=float, default=0.2, help="rho for SAM")
    parser.add_argument("--sam_adaptive", type=bool, default=False, help="use adaptive SAM")
    parser.add_argument("--look_alpha", type=float, default=0.1, help="alpha for LookSAM/AlternateSAM")
    parser.add_argument("--gold_delta", type=float, default=1, help="delta for goldstein")
    parser.add_argument("--norm_sgd_lr", type=float, default=1e-3, help="learning rate for normalized sgd when overfit")

    # analysis hyperparameters
    parser.add_argument("--adv_eta", type=float, default=0.01, help="eta for adversarial perturbation")

    parser.add_argument("--multiple_run", type=bool, default=False, help="independent run without overwriting or loading")
    parser.add_argument("--run_from_scratch", type=bool, default=False, help="do not load from previous results")
    parser.add_argument("--store_model_checkpoint", type=bool, default=False, help="store the checkpoint models every analysis step")
    parser.add_argument("--no_train", action='store_true', help="train model")
    parser.add_argument("--do_eval", action='store_true', help="evaluate model")
    parser.add_argument("--model_average", nargs='+', type=int, default=[0,1], help="index of runs to be averaged")

    #federated learning hyperparameters
    parser.add_argument("--server_opt_name", type=str, default="adam", choices=OPTIMIZERS + ["clip_sgd"], help="optimizer of server")
    parser.add_argument("--client_num", type=int, default=1, help="number of clients")
    parser.add_argument("--client_opt_name", type=str, default="sgd", choices=["sgd", "adam"], help="optimizer of clients")
    parser.add_argument("--client_lr", type=float, default=0.01, help="lr of clients")
    parser.add_argument("--client_momentum", type=float, default=0.0, help="momentum of clients")
    parser.add_argument("--client_weight_decay", type=float, default=0.0, help="momentum of clients")
    parser.add_argument("--client_epoch", type=int, default=200, help="total epochs of client training")
    parser.add_argument("--sketch_size", type=int, default=1000, help="sketch size in communication")
    parser.add_argument("--non_iid_alpha", type=float, default=0.0, help="percentage of majority class in one client")
    parser.add_argument("--clip_tau", type=float, default=1.0, help="clip tau in clipping method")

    #llm hyperparameters
    parser.add_argument("--task_name", type=str, default="mrpc", help="task name")
    parser.add_argument("--max_seq_length", type=int, default=128, help="max_seq_length")

    args = parser.parse_args()


    debug               = args.debug # Only runs 10 batches per epoch for debugging
    no_train            = args.no_train
    do_eval             = args.do_eval
    multi_run           = args.multiple_run
    run_from_scratch    = args.run_from_scratch
    store_model_checkpoint = args.store_model_checkpoint
    model_params = {}
    opt_params = {}
    analysis_params = {}

    # dataset parameters
    dataset_name        = args.dataset #"spurious" #"cifar"\
    sp_train_size       = args.sp_train_size
    sp_feat_dim         = args.sp_feat_dim
    sp_patch_dim        = args.sp_patch_dim

    # model parameters
    model_name          = args.model #"2-mlp-sim-bn"#"weight_norm_torch" #"weight_norm" #"resnet18"
    width               = args.width #2048#512 #, 1024
    depth               = args.depth
    wn_init_mode        = args.init_mode  # "O(1)" # "O(1/sqrt{m})"
    wn_basis_var        = args.basis_var
    wn_scale            = args.wn_scale
    width_factor        = args.width_factor

    vit_patch_size      = args.vit_patch_size
    #vit_head_num        = args.vit_head_num
    #vit_depth           = args.vit_depth

    # Optimization Criterion
    # loss_name = 'CrossEntropyLoss'
    loss_name           = args.loss
    opt_name            = args.opt
    analysis_list       = args.analysis if args.analysis else [] # ['loss', 'eigs'] #['loss','eigs','nc',''weight_norm']
    analysis_interval   = args.log_interval
    tiny_analysis  = 'gn_eigs' in analysis_list # avoid endless running time in computing gauss newton matrix

    # Optimization hyperparameters
    lr_decay            = args.lr_decay #1# 0.1
    epochs              = args.epoch
    epochs_lr_decay     = [epochs//3, epochs*2//3]

    batch_size          = args.batch_size #512 # 128

    model_average       = args.model_average
    
    # for training process
    opt_params["mixup"]               = args.mixup
    opt_params["epoch"]               = args.epoch
    opt_params["scheduler_name"]      = args.scheduler
    opt_params["lr_min"]              = args.lr_min

    #hyperparameters for sam
    opt_params["base_opt"]            = args.base_opt
    opt_params["sam_rho"]             = args.sam_rho #0.1
    opt_params["sam_adaptive"]        = args.sam_adaptive #False
    opt_params["look_alpha"]          = args.look_alpha
    opt_params["norm_sgd_lr"]         = args.norm_sgd_lr
    opt_params["gold_delta"]          = args.gold_delta
    opt_params["train_stats"]         = args.train_stats
    opt_params["forward_backward"]    = opt_name not in ["replay_sam"]
    opt_params["opt_first_step"]      = True

    # analysis hyperparameters
    analysis_params["adv_eta"]        = args.adv_eta

    #federated learning parameters
    opt_params["server_opt_name"]  = args.server_opt_name
    opt_params["client_opt_name"]  = args.client_opt_name
    opt_params["client_lr"]        = args.client_lr
    opt_params["client_num"]       = args.client_num
    opt_params["client_epoch"]     = args.client_epoch
    opt_params["sketch_size"]      = args.sketch_size
    opt_params["server_momentum"]  = args.momentum
    opt_params["client_momentum"]  = args.client_momentum
    opt_params["client_weight_decay"]  = args.client_weight_decay
    opt_params["non_iid"]          = args.non_iid_alpha
    opt_params["clip_tau"]         = args.clip_tau
    
    exp_avg, exp_avg_sq            = None, None

    opt_params["hf_model"]         = args.dataset in ["glue"] or model_name in ["google/vit-base-patch16-224-in21k"]

    if debug:
        torch.autograd.set_detect_anomaly(True)
    if not multi_run:
        torch.manual_seed(32)

    lr                  = args.lr
    momentum            = args.momentum #0 # 0.9
    weight_decay        = args.weight_decay #0 # 5e-4 * 10
    label_smoothing     = args.label_smoothing

    if opt_params["scheduler_name"] != "none" and not run_from_scratch:
        raise NameError('Must run from scratch if using learning rate decay.')

    if dataset_name == "cifar":
        from data.cifar import load_cifar
        #if opt_name == 'gd':
        #    train_loader, test_loader, analysis_loader, analysis_test_loader, input_ch, num_pixels, C, transform_to_one_hot = load_cifar(loss_name, batch_size, sp_train_size)
        if opt_name == "federated":
            if not opt_params["hf_model"]:
                from data.cifar import load_cifar_federated
                train_loader, client_loaders, test_loader, analysis_loader, analysis_test_loader, input_ch, num_pixels, C, transform_to_one_hot, data_params = load_cifar_federated(loss_name, batch_size, client_num=opt_params["client_num"], alpha=opt_params["non_iid"])
            else:
                from data.cifar import load_cifar_vit_federated
                train_loader, client_loaders, test_loader, analysis_loader, analysis_test_loader, id2label, label2id, C, transform_to_one_hot, data_params = load_cifar_vit_federated(model_name =model_name, batch_size= batch_size, client_num=opt_params["client_num"], alpha=0.0)
            if opt_params["non_iid"] != 0:
                model_params = model_params | {"non_iid": opt_params["non_iid"]}
        elif opt_params["hf_model"]:
            from data.cifar import load_cifar_vit
            train_loader, test_loader, analysis_loader, analysis_test_loader, id2label, label2id, C, transform_to_one_hot, data_params = load_cifar_vit(model_name, batch_size)
        else:
            train_loader, test_loader, analysis_loader, analysis_test_loader, input_ch, num_pixels, C, transform_to_one_hot, data_params = load_cifar(loss_name, batch_size, augment=args.augment, tiny_analysis=tiny_analysis)
            if args.augment != 0:
                model_params = model_params | {"augment": args.augment}
    elif dataset_name == "cifar100":
        if opt_name == "federated":
            if opt_params["non_iid"] == 0:
                from data.cifar import load_cifar100_federated
                train_loader, client_loaders, test_loader, analysis_loader, analysis_test_loader, input_ch, num_pixels, C, transform_to_one_hot, data_params = load_cifar100_federated(loss_name, batch_size, client_num=opt_params["client_num"])
            else:
                from data.cifar import load_cifar100_federated_non_iid
                train_loader, client_loaders, test_loader, analysis_loader, analysis_test_loader, input_ch, num_pixels, C, transform_to_one_hot, data_params = load_cifar100_federated_non_iid(loss_name, batch_size, client_num=opt_params["client_num"], alpha=opt_params["non_iid"])
                model_params = model_params | {"non_iid": opt_params["non_iid"]}
        else:
            from data.cifar import load_cifar100
            train_loader, test_loader, analysis_loader, analysis_test_loader, input_ch, num_pixels, C, transform_to_one_hot, data_params = load_cifar100(loss_name, batch_size)
    elif dataset_name == "mnist":
        from data.mnist import load_mnist
        train_loader, test_loader, analysis_loader, input_ch, C, transform_to_one_hot = load_mnist(loss_name, batch_size)
    elif dataset_name == "emnist":
        if opt_name == "federated":
            from data.emnist import load_emnist_federated
            train_loader, client_loaders, test_loader, analysis_loader, analysis_test_loader, input_ch, num_pixels, C, transform_to_one_hot, data_params = load_emnist_federated(loss_name, batch_size, client_num=opt_params["client_num"])
        else:
            raise NotImplementedError
    elif dataset_name == "mnist_cifar":
        from data.mnist_cifar import load_mnist_cifar
        train_loader, test_loader, analysis_loader, analysis_test_loader, input_ch, num_pixels, C, transform_to_one_hot, data_params = load_mnist_cifar(mnist_classes=(0,1), cifar_classes=(1,9), batch_size=batch_size, randomize_mnist=False, randomize_cifar=False)
        _, test_loader, _, analysis_test_loader, _, _, _, _, _ = load_mnist_cifar(mnist_classes=(0,1), cifar_classes=(1,9), batch_size=batch_size, randomize_mnist=True, randomize_cifar=False)
        #_, _, analysis_loader_rand_cifar, _, _, _, _, _, _ = load_mnist_cifar(mnist_classes=(0,1), cifar_classes=(1,9), batch_size=batch_size, randomize_mnist=False, randomize_cifar=True)
    elif dataset_name == "spurious":
        from data.spurious import load_spurious_data
        train_loader, test_loader, analysis_loader, analysis_test_loader, num_pixels, C, transform_to_one_hot = load_spurious_data(loss_name, sp_feat_dim, sp_train_size, batch_size)
        model_params = model_params | {"feat_dim": sp_feat_dim, "train_size": sp_train_size}
    elif dataset_name == "spurious-2d":
        from data.spurious import load_signal_noise_data_2d
        train_loader, test_loader, analysis_loader, analysis_test_loader, num_pixels, C, transform_to_one_hot, data_params = load_signal_noise_data_2d(loss_name, sp_patch_dim, sp_feat_dim, sp_train_size, batch_size)
        model_params = model_params | {"patch_dim": sp_patch_dim, "feat_dim": sp_feat_dim, "train_size": sp_train_size}
        analysis_params = analysis_params | data_params
    elif dataset_name == "multi-view":
        from data.spurious import load_multi_view_data
        train_loader, test_loader, analysis_loader, analysis_test_loader, num_pixels, C, transform_to_one_hot, data_params = load_multi_view_data(loss_name, sp_patch_dim, sp_feat_dim, sp_train_size, batch_size)
        model_params = model_params | {"patch_dim": sp_patch_dim, "feat_dim": sp_feat_dim, "train_size": sp_train_size}
        analysis_params = analysis_params | data_params
    elif dataset_name == "multi-view-orthogonal":
        from data.spurious import load_multi_view_orthogonal_data
        train_loader, test_loader, analysis_loader, analysis_test_loader, num_pixels, C, transform_to_one_hot, data_params = load_multi_view_orthogonal_data(loss_name, sp_patch_dim, sp_feat_dim, sp_train_size, batch_size)
        model_params = model_params | {"patch_dim": sp_patch_dim, "feat_dim": sp_feat_dim, "train_size": sp_train_size}
        analysis_params = analysis_params | data_params
    elif dataset_name == "secondary_feature":
        from data.spurious import load_secondary_feature_data
        train_loader, test_loader, analysis_loader, analysis_test_loader, num_pixels, C, transform_to_one_hot, data_params = load_secondary_feature_data(loss_name, sp_patch_dim, sp_feat_dim, sp_train_size, batch_size)
        model_params = model_params | {"patch_dim": sp_patch_dim, "feat_dim": sp_feat_dim, "train_size": sp_train_size}
        analysis_params = analysis_params | data_params
    elif dataset_name == "orthogonal":
        from data.spurious import load_orthogonal_data
        train_loader, test_loader, analysis_loader, analysis_test_loader, num_pixels, C, transform_to_one_hot, data_params = load_orthogonal_data(loss_name, sp_patch_dim, sp_feat_dim, sp_train_size, batch_size)
        model_params = model_params | {"patch_dim": sp_patch_dim, "feat_dim": sp_feat_dim, "train_size": sp_train_size}
        analysis_params = analysis_params | data_params
    elif dataset_name == "scalarized":
        from data.spurious import load_scalaraized_data
        train_loader, test_loader, analysis_loader, analysis_test_loader, num_pixels, C, transform_to_one_hot, data_params = load_scalaraized_data(loss_name, sp_patch_dim, sp_feat_dim, sp_train_size, batch_size)
        model_params = model_params | {"patch_dim": sp_patch_dim, "train_size": sp_train_size}
        analysis_params = analysis_params | data_params
    elif dataset_name == "weight_norm_teacher":
        from data.synthetic import load_weight_norm_teacher
        train_loader, test_loader, analysis_loader, analysis_test_loader, num_pixels, C, transform_to_one_hot, data_params = load_weight_norm_teacher(sp_feat_dim, sp_train_size, batch_size)
        model_params = model_params | {"feat_dim": sp_feat_dim, "train_size": sp_train_size}
        analysis_params = analysis_params | data_params
    elif dataset_name == "glue":
        model_params = {"task_name": args.task_name, "seq_length": args.max_seq_length}
        if args.sp_train_size != -1:
            model_params = model_params | {"train_size": args.sp_train_size}
        if opt_name == "federated":
            from data.glue import load_glue_federated
            model, train_loader, client_loaders, test_loader, analysis_loader, analysis_test_loader, C, transform_to_one_hot, data_params = load_glue_federated(model_name, batch_size, opt_params["client_num"], model_params)
        else:
            from data.glue import load_glue
            model, train_loader, test_loader, analysis_loader, analysis_test_loader, C, transform_to_one_hot, data_params = load_glue(model_name, batch_size, model_params)
    
    compute_acc = data_params["compute_acc"]
    if args.mixup == "cut":
        from data.data_process import CutMix
        cutmix = CutMix(int(np.sqrt(num_pixels/input_ch)), beta=1)
        model_params = model_params | {"mixup" : "cut"}

    if model_name == "resnet18":
        model = models.resnet18(pretrained=False, num_classes=C)
        model_params = {} | model_params
        if dataset_name == "mnist":
            model.conv1 = nn.Conv2d(input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias=False) # Small dataset filter size used by He et al. (2015)
            model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        if opt_name == "federated":
            for name, module in model.named_modules():
                if isinstance(module, nn.modules.batchnorm.BatchNorm2d):
                    module.track_running_stats = False
    elif model_name == "resnet_fixup":
        from arch.resnet_fixup import fixup_resnet
        model = fixup_resnet(depth=depth, num_classes=C, input_ch=input_ch)
        model_params = {"depth": depth} | model_params
    elif model_name == "resnet_gn":
        from arch.resnet_gn import resnet_gn
        model = resnet_gn(depth=depth, num_classes=C)
        model_params = {"depth": depth} | model_params
    elif model_name == "WideResNet":
        from arch.wide_resnet import WideResNet
        model = WideResNet(depth=16, width_factor=width_factor, dropout=0.0, in_channels=input_ch, labels=C)
        model_params = {"width": width_factor} | model_params
    elif model_name == "ViT":
        #from torchvision.models.vision_transformer import VisionTransformer
        vit_params = {768: {"depth": 12, "heads": 12}, 1024: {"depth": 24, "heads": 16}, 1280: {"depth": 32, "heads": 16}}
        from vit_pytorch import ViT
        """
        model = VisionTransformer(image_size=int(np.sqrt(num_pixels/input_ch)), 
                                  patch_size= 4, 
                                  num_layers = 4, 
                                  num_heads= 8, # embed_dim must be divisible by num_heads
                                  hidden_dim = width,
                                  mlp_dim = width,
                                  num_classes = C) """
        model = ViT(image_size=int(np.sqrt(num_pixels/input_ch)), patch_size=8, num_classes=C, dim=width, depth=vit_params[width]["depth"], heads=vit_params[width]["heads"], mlp_dim=4*width) #depth=6, heads=8
        #from arch.vit import ViT
        #model = ViT(in_c=input_ch, num_classes=C, img_size=int(np.sqrt(num_pixels/input_ch)),patch=8,dropout=0,num_layers=7,hidden=width,mlp_hidden=width,head=12,is_cls_token=True)
        model_params = {"width": width,  "depth":vit_params[width]["depth"], "heads":vit_params[width]["heads"]} | model_params
    elif model_name == "emnistcnn":
        from arch.conv import EMNISTCNN
        model = EMNISTCNN(40, 160, 200, 0.4)
    elif model_name == "weight_norm":
        from arch.weight_norm import weight_norm_net
        model = weight_norm_net(num_pixels, [width, width], wn_init_mode, wn_basis_var, wn_scale, C)
        model_params = {"width": width, "init": wn_init_mode, "var": wn_basis_var, "scale": wn_scale} | model_params
    elif model_name == "weight_norm_v2":
        from arch.weight_norm import weight_norm_net_v2
        model = weight_norm_net_v2(num_pixels, [width, width], wn_init_mode, wn_basis_var, wn_scale, C)
        model_params = {"width": width, "init": wn_init_mode, "var": wn_basis_var, "scale": wn_scale} | model_params
    elif model_name == "weight_norm_width_scale":
        from arch.weight_norm import weight_norm_net_old
        model = weight_norm_net_old(num_pixels, [width, width], wn_init_mode, wn_basis_var, wn_scale, C)
        model_params = {"width": width, "init": wn_init_mode, "var": wn_basis_var, "scale": wn_scale} | model_params
    elif model_name == "weight_norm_torch":
        from arch.weight_norm import weight_norm_torch
        model = weight_norm_torch(num_pixels, [width, width], C)
        model_params = {"width": width} | model_params
    elif model_name == "WideResNet_WN_woG":
        from arch.weight_norm import WideResNet_WN_woG
        model = WideResNet_WN_woG(depth=16, width_factor=width_factor, dropout=0.0, in_channels=input_ch, labels=C)
        model_params = {"width_factor": width_factor} | model_params
    elif model_name == "2-mlp-sim-bn":
        from arch.mlp_sim_bn import mlp_sim_bn
        model = mlp_sim_bn(num_pixels, C)
        model_params = {} | model_params
    elif model_name == "2-mlp-sim-ln":
        from arch.mlp_sim_ln import mlp_sim_ln
        model = mlp_sim_ln(num_pixels, C)
        model_params = {} | model_params
    elif model_name == "conv_fixed_last":
        from arch.conv import conv_fixed_last_layer
        assert C == 1
        model = conv_fixed_last_layer(num_pixels, width)
        model_params = {"nfilters": width} | model_params
    elif model_name == "conv_with_last":
        from arch.conv import conv_with_last_layer
        assert C == 1
        model = conv_with_last_layer(num_pixels, width)
        model_params = {"nfilters": width} | model_params
    elif model_name == "scalarized_conv":
        from arch.conv import scalarized_conv
        assert C == 1
        model = scalarized_conv(width, sp_patch_dim)
        model_params = {"nfilters": width} | model_params
    elif model_name == "google/vit-base-patch16-224-in21k":
        #reference "https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_the_%F0%9F%A4%97_Trainer.ipynb#scrollTo=fZpqx7giniv8"
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                        id2label=id2label,
                                                        label2id=label2id)
    elif model_name == "dino_vit_small":
        from arch.dino_vit import vit_small
        model = vit_small(patch_size=vit_patch_size, num_classes=0).to(device)
        if vit_patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif vit_patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        model.load_state_dict(state_dict, strict=True)

        model_params = {"patch_size": vit_patch_size}

    print("number of parameters:", len(parameters_to_vector(model.parameters())))
    # analysis parameters
    """
    epoch_list          = [1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11,
                        12,  13,  14,  16,  17,  19,  20,  22,  24,  27,   29,
                        32,  35,  38,  42,  45,  50,  54,  59,  65,  71,   77,
                        85,  92,  101, 110, 121, 132, 144, 158, 172, 188,  206,
                        225, 245, 268, 293, 320, 350]
    """
    #epoch_list = np.arange(1, epochs+1, 100).tolist()
    # register hook that saves last-layer input into features
    if "nc" in analysis_list:
        classifier = model.fc
        classifier.register_forward_hook(hook)
    """
    class LabelSmoothingCrossEntropyLoss(torch.nn.Module):
        def __init__(self, classes, smoothing=0.0, dim=-1):
            super(LabelSmoothingCrossEntropyLoss, self).__init__()
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing
            self.cls = classes
            self.dim = dim

        def forward(self, pred, target):
            pred = pred.log_softmax(dim=self.dim)
            with torch.no_grad():
                true_dist = torch.zeros_like(pred)
                true_dist.fill_(self.smoothing / (self.cls - 1))
                true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))    

    class LabelSmoothingCrossEntropyLoss_summed(torch.nn.Module):
        def __init__(self, classes, smoothing=0.0, dim=-1):
            super(LabelSmoothingCrossEntropyLoss_summed, self).__init__()
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing
            self.cls = classes
            self.dim = dim

        def forward(self, pred, target):
            pred = pred.log_softmax(dim=self.dim)
            with torch.no_grad():
                true_dist = torch.zeros_like(pred)
                true_dist.fill_(self.smoothing / (self.cls - 1))
                true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            return torch.sum(torch.sum(-true_dist * pred, dim=self.dim))    
    """
    if loss_name == 'CrossEntropyLoss':
        
        assert transform_to_one_hot # assert target is index vector
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        criterion_summed = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='sum')
        
        #criterion = LabelSmoothingCrossEntropyLoss(classes=C, smoothing=label_smoothing)
        #criterion_summed = LabelSmoothingCrossEntropyLoss_summed(classes=C, smoothing=label_smoothing)
        if label_smoothing != 0:
            model_params = model_params | {"smooth": label_smoothing}
    elif loss_name == 'MSELoss':
        if transform_to_one_hot:
            def mse_sum_with_one_hot(out, target):
                return nn.MSELoss(reduction='sum')(out, F.one_hot(target, num_classes=C).float())

            def mse_with_one_hot(out, target):
                return nn.MSELoss()(out, F.one_hot(target, num_classes=C).float()) * C

            criterion = mse_with_one_hot
            criterion_summed = mse_sum_with_one_hot
        else:
            criterion = nn.MSELoss()
            criterion_summed = nn.MSELoss(reduction='sum')
    elif loss_name == "BCELoss":
        assert not transform_to_one_hot
        def BCE(out, target):
            return torch.mean(torch.log(1+torch.exp(-out * target)))
        
        def BCE_sum(out, target):
            return torch.sum(torch.log(1+torch.exp(-out * target)))
        
        criterion = BCE
        criterion_summed = BCE_sum

    if not no_train:
        train_graphs = graphs()
        optimizer, lr_scheduler, model_params= load_optimizer(opt_name, model, lr, momentum, weight_decay, lr_decay, epochs_lr_decay, True, model_params, **opt_params)
        if opt_name == "federated":
            from optimizer.load_optimizer import load_fake_scheduler
            client_lr_scheduler = load_fake_scheduler(opt_params["client_lr"], **opt_params)

        load_from_epoch = 0
        if not run_from_scratch:
            load_from_epoch = continue_training(lr, dataset_name, loss_name, opt_name, model_name, momentum, weight_decay, batch_size, epochs, multi_run, **model_params)
        epoch_list = np.arange(load_from_epoch+1, epochs+1, analysis_interval).tolist()
        if load_from_epoch != 0:
            from utilities import optimizer_to
            print("loading from trained epoch {}".format(load_from_epoch))
            load_from_dir = get_directory(lr, dataset_name, loss_name, opt_name, model_name, momentum, weight_decay, batch_size, load_from_epoch, multi_run, **model_params)
            model.load_state_dict(torch.load(os.path.join(load_from_dir, "model.ckpt")))
            optimizer.load_state_dict(torch.load(os.path.join(load_from_dir, "optimizer.ckpt")))
            optimizer_to(optimizer, device)
            with open(f'{load_from_dir}/train_graphs.pk', 'rb') as f:
                train_graphs = pickle.load(f)
        model = model.to(device)

        directory = get_directory(lr, dataset_name, loss_name, opt_name, model_name, momentum, weight_decay, batch_size, epochs, multi_run, **model_params)
        os.makedirs(directory, exist_ok=True)
        print(directory)

        import pickle

        cur_epochs = []
        for epoch in range(load_from_epoch+1, epochs + 1):
            if opt_name == "federated":
                #exp_avg, exp_avg_sq = federated_train(model, loss_name, criterion, device, C, client_loaders, exp_avg, exp_avg_sq, opt_params, epoch)
                client_lr = client_lr_scheduler.get_last_lr()[0]
                federated_train(model, loss_name, criterion, device, client_loaders, optimizer, None, client_lr, opt_params, epoch)
                client_lr_scheduler.step()
                print(client_lr)
            else:
                train(model, loss_name, criterion, device, train_loader, optimizer, lr_scheduler, epoch, opt_params)
                #lr_scheduler.step()
            
            if epoch in epoch_list:
                #print("Epoch: ", epoch)
                train_graphs.log_epochs.append(epoch)
                #analysis(train_graphs, model, criterion_summed, device, C, analysis_loader, test_loader)
                analysis(train_graphs, analysis_list, model, model_name, criterion_summed, device, C, compute_acc,train_loader, test_loader, analysis_loader, analysis_test_loader, opt_params, analysis_params)
                
                pickle.dump(train_graphs, open(f"{directory}/train_graphs.pk", "wb"))
                torch.save(model.state_dict(), f"{directory}/model.ckpt")
                torch.save(optimizer.state_dict(), f"{directory}/optimizer.ckpt")
                if store_model_checkpoint:
                    os.makedirs(f"{directory}/checkpoint_{epoch}")
                    torch.save(model.state_dict(), f"{directory}/checkpoint_{epoch}/model.ckpt") 
                    

    if do_eval:
        eval_graphs = graphs()
        if 'model_average' in analysis_list:
            running_directory = get_running_directory(lr, dataset_name, loss_name, opt_name, model_name, momentum, weight_decay, batch_size, epochs, **model_params)
            epoch_list = np.arange(1, epochs+1, analysis_interval).tolist()
            for epoch in epoch_list:
                eval_graphs.log_epochs.append(epoch)

                sdA = torch.load(os.path.join(running_directory, f"run_{model_average[0]}", f"checkpoint_{epoch}", "model.ckpt"))
                sdB = torch.load(os.path.join(running_directory, f"run_{model_average[1]}", f"checkpoint_{epoch}", "model.ckpt"))

                for key in sdA:
                    sdB[key] = (sdA[key] + sdB[key]) / 2

                model.load_state_dict(sdB)
                model = model.to(device)

                analysis(eval_graphs, analysis_list, model, model_name, criterion_summed, device, C, compute_acc, train_loader, test_loader, analysis_loader, analysis_test_loader, opt_params, analysis_params)
                
            os.makedirs(f"{running_directory}/avg_{model_average[0]}{model_average[1]}", exist_ok=True)
            pickle.dump(eval_graphs, open(f"{running_directory}/avg_{model_average[0]}{model_average[1]}/eval_graphs.pk", "wb"))
        elif 'attention_map' in analysis_list:
            analysis(eval_graphs, analysis_list, model, model_name, criterion_summed, device, C, compute_acc, train_loader, test_loader, analysis_loader, analysis_test_loader, opt_params, analysis_params)

            directory = get_directory(lr, dataset_name, loss_name, opt_name, model_name, momentum, weight_decay, batch_size, epochs, multi_run, **model_params)
            print(directory)
            os.makedirs(directory, exist_ok=True)
            pickle.dump(eval_graphs, open(f"{directory}/eval_graphs.pk", "wb"))
        
