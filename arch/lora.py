import sys
import torch
import torch.optim as optim
import numpy as np

def add_adapters_dataset(model_name, model, lora_rank, lora_alpha, lora_freeze_a=False, adapter_name="default"):
    if model_name == "google/vit-base-patch16-224-in21k":
        model, Lora_config = add_adapters(model, lora_rank, lora_alpha, "classifier", ["query", "value"], adapter_name=adapter_name)
        output_layer_name = 'classifier' 
    elif model_name == 'flair':
        add_adapters(model, lora_rank, lora_alpha, "classifier", ["query", "value"])
        # add_adapters(model, lora_rank, lora_alpha, 'classifier', ['convolution'])
    elif model_name == 'gpt2':
        model, Lora_config = add_adapters(model, lora_rank, lora_alpha, "score", ["c_attn", "c_proj", "c_fc"], freeze_a=lora_freeze_a, adapter_name=adapter_name) #"score"
        output_layer_name = 'score'
    elif model_name == 'reddit':
        add_adapters(model, lora_rank, lora_alpha, None, ["c_attn", "c_proj", "c_fc"])
    elif model_name == "roberta-base":
        model, Lora_config = add_adapters(model, lora_rank, lora_alpha, "classifier", ["query", "value"], adapter_name=adapter_name)
        output_layer_name = 'classifier'
    elif model_name in ["akjindal53244/Arithmo-Mistral-7B", "mistralai/Mistral-7B-v0.1"]:
        # from https://huggingface.co/upaya07/Arithmo2-Mistral-7B-adapter/blob/main/adapter_config.json
        #model, Lora_config = add_adapters(model, lora_rank, lora_alpha, None, ["o_proj", "q_proj", "v_proj", "down_proj", "up_proj", "k_proj", "gate_proj"], task_type="CAUSAL_LM")
        model, Lora_config = add_adapters(model, lora_rank, lora_alpha, None, ["q_proj", "v_proj"], task_type="CAUSAL_LM", adapter_name=adapter_name)
        output_layer_name = None   
    return model, output_layer_name, Lora_config

def synchronize_lora(model, server_name, truncate_last):
    #synchronize
    adapter_weights = {}
    truncate_err = 0
    for name, param in model.named_parameters():
        if server_name in name:
            adapter_weights[name] = param #store the server param
            if len(param.data.shape) == 2:
                row, col = param.data.shape
                print(name)
                r_axis = 1 if row < col else 0
                print(torch.norm(param.data, dim=r_axis))
        elif 'client' in name:
            import re
            server_adapter_name = re.sub(r'client_\d+', server_name, name)
            adapter_weight_full = adapter_weights[server_adapter_name].data.clone() #assign the same param to client models
            if len(param.data.shape) == 2:
                row, col = param.data.shape
                if truncate_last:
                    param.data = adapter_weight_full[:row, :col]
                else:
                    param.data = adapter_weight_full[-row:, -col:]
                truncate_err += torch.norm(adapter_weight_full)**2 - torch.norm(param.data)**2
            elif len(param.data.shape) == 1:
                param.data = adapter_weight_full
            else:
                assert False
    print("Truncation Error: ", truncate_err)

def examine_lora(model, name1, name2):
    for name, param in model.named_parameters():
        if name == name1:
            param1 = param
        elif name == name2:
            param2 = param
    print(param1.data_ptr())
    print(param2.data_ptr())
    print(param1 is param2)

    for name, param in model.named_parameters():
        print(name, torch.norm(param).item(), torch.norm(param.grad).item() if param.grad is not None else None)
        if name == "vit.encoder.layer.0.attention.attention.query.lora_A.client_0.weight":
            #param += torch.rand_like(param)
            param.add_(torch.rand_like(param))
            print(torch.norm(param).item())
        elif name == "vit.encoder.layer.0.attention.attention.query.lora_A.client_1.weight":
            print(torch.norm(param).item())
        

def add_adapters_homo(client_num, model_name, model, lora_rank, lora_alpha, opt_params, lora_freeze_a=False):
    client_rank = lora_rank
    lora_alpha = lora_rank
    model, output_layer_name, Lora_config = add_adapters_dataset(model_name, model, client_rank, lora_alpha, lora_freeze_a=lora_freeze_a, adapter_name=opt_params["server_name"])
    for i in range(client_num):
        client_id = "client_{}".format(i)
        model, output_layer_name, Lora_config = add_adapters_dataset(model_name, model, client_rank, lora_alpha, lora_freeze_a=lora_freeze_a, adapter_name=client_id)
    
    if opt_params["fedlora_avg"] == "svd":
        truncate_last=False
    elif opt_params["fedlora_avg"] == "avg":
        truncate_last=True
    else:
        assert False
    synchronize_lora(model, server_name=opt_params["server_name"], truncate_last=truncate_last)
    model.set_adapter(opt_params["server_name"])
    #examine_lora(model, name1="classifier.modules_to_save.server.weight", name2="classifier.modules_to_save.client_0.weight")
    return model, output_layer_name, Lora_config

def add_adapters_hetero(client_num, model_name, model, lora_rank, lora_alpha, opt_params, lora_freeze_a=False):
    model, output_layer_name, Lora_config = add_adapters_dataset(model_name, model, lora_rank, lora_rank, lora_freeze_a=lora_freeze_a, adapter_name=opt_params["server_name"])
    client_ranks = []
    np.random.seed(42)
    for i in range(client_num):
        client_rank = np.random.randint(4, lora_rank+1)
        #client_rank = lora_rank
        client_ranks.append(client_rank)
        client_id = "client_{}".format(i)
        model, output_layer_name, Lora_config = add_adapters_dataset(model_name, model, client_rank, client_rank, lora_freeze_a=lora_freeze_a, adapter_name=client_id)

    if opt_params["fedlora_avg"] == "svd":
        truncate_last=False
    elif opt_params["fedlora_avg"] == "avg":
        truncate_last=True
    else:
        assert False
    synchronize_lora(model, server_name=opt_params["server_name"], truncate_last=truncate_last)
    model.set_adapter(opt_params["server_name"])
    print("Client Ranks: ", client_ranks)
    return model, output_layer_name, Lora_config


def add_ft(model, output_layer_name, target_modules):
    for n, p in model.named_parameters():
        if output_layer_name and output_layer_name in n:
            p.requires_grad = True
        else:
            require_grad = False
            for module_name in target_modules:
                if module_name in n and not require_grad:
                    require_grad = True
            p.requires_grad = require_grad
                

def add_adapters(model, lora_rank, lora_alpha, output_layer_name, target_modules, freeze_a=False, task_type=None, adapter_name="default"):
    from peft import LoraConfig, get_peft_model

    if lora_rank > 0:
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=[output_layer_name] if output_layer_name is not None else [],
            task_type=task_type
        )
        #model = get_peft_model(model, config)
        model.add_adapter(config, adapter_name=adapter_name)
        if freeze_a:
            for n, p in model.named_parameters():
                if "lora_A" in n:
                    p.requires_grad=False
    elif lora_rank == 0: # linear fine-tune
        for n,p in model.named_parameters():
            if output_layer_name and output_layer_name in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
    elif lora_rank == -1: #full fine-tune
        add_ft(model, output_layer_name, target_modules)
    return model, config

def lora_name_to_base(lora_A_name, server_name, use_parallel=False):
    base_weight_name = lora_A_name.replace("lora_A.{}".format(server_name), "base_layer")
    if use_parallel:
        base_weight_name = "module."+base_weight_name
    return base_weight_name

def base_to_lora_name(base_name, server_name):
    lora_A_name = base_name.replace("base_layer", "lora_A.{}".format(server_name))
    lora_B_name = base_name.replace("base_layer", "lora_B.{}".format(server_name))
    return lora_A_name, lora_B_name

def get_lora_norm(adapter_weights):
    norm_A, norm_B = 0, 0 
    #norm_A_diff, norm_B_diff=0, 0
    for name in adapter_weights:
        if 'lora_A' in name:
            norm_A += torch.norm(adapter_weights[name]) ** 2
            #norm_A_diff += torch.norm(adapter_weights_avg[name] - adapter_weights[name]) ** 2
        elif 'lora_B' in name:
            norm_B += torch.norm(adapter_weights[name]) ** 2
            #norm_B_diff += torch.norm(adapter_weights_avg[name] - adapter_weights[name]) ** 2
    #print("param norms: ", norm_A.item(), norm_B.item(), norm_A_diff.item(), norm_B_diff.item())
    print("param norms: ", norm_A.item(), norm_B.item())
    return norm_A.item(), norm_B.item()

def get_base_layer_norm(model):
    norm_base_layer = 0
    for name, param in model.named_parameters():
        if 'base_layer' in name:
            norm_base_layer += torch.norm(param.data) ** 2
    print("base layer norms: ", norm_base_layer.item())
    return norm_base_layer.item()

def load_server_optimizer(model, lr, momentum, weight_decay, model_params, **kwargs):
    from torch.nn.parameter import Parameter
    parameters = {}
    adapter_weights = {}
    adapter_names = []
    base_names = []

    output_layer_name = kwargs["output_layer_name"]
    for name, param in model.named_parameters():
        # select lora_A and lora_B
        if param.requires_grad and "lora" in name:
            adapter_names.append(name)
            adapter_weights[name] = param
    #output_layer_name = adapter_names[-1]      
    get_lora_norm(adapter_weights)

    for i in range(0, len(adapter_names), 2):
        lora_A_name, lora_B_name = adapter_names[i], adapter_names[i+1]
        base_weight_name = lora_name_to_base(lora_A_name, kwargs["server_name"], kwargs["use_parallel"])
        base_names.append(base_weight_name)
        #parameters[base_weight_name] = Parameter((adapter_weights[lora_B_name] @ adapter_weights[lora_A_name]).T.to(torch.float16).to(kwargs["device"]))

        if kwargs["multi_gpu"]:
            parameters[base_weight_name] = Parameter((adapter_weights[lora_B_name] @ adapter_weights[lora_A_name]).T.to(torch.device("cuda:1")))
        else:  
            parameters[base_weight_name] = Parameter((adapter_weights[lora_B_name] @ adapter_weights[lora_A_name]).T.to(kwargs["device"]))                                    
        #print(lora_A_name, lora_B_name, adapter_weights[lora_B_name].shape, adapter_weights[lora_A_name].shape, (adapter_weights[lora_B_name] @ adapter_weights[lora_A_name]).shape)

    for name, param in model.named_parameters():
        #if name in base_names:
        #    parameters[name] = Parameter(torch.zeros_like(param.float()).to(kwargs["device"]))
        if output_layer_name and output_layer_name in name:
            if param.requires_grad:
                parameters[name] = Parameter(param.to(kwargs["device"]))

    if kwargs["server_opt_name"] == "sgd" or kwargs["server_opt_name"] == "gd":
        from torch.optim import SGD
        optimizer = SGD(parameters.values(),
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    elif kwargs["server_opt_name"] == "adam":
        from torch.optim import Adam
        optimizer = Adam(parameters.values(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    elif kwargs["server_opt_name"] == "adamw":
        from torch.optim import AdamW
        optimizer = AdamW(parameters.values(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)

    if kwargs["scheduler_name"] == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs["epoch"], eta_min=kwargs["lr_min"])
        model_params = model_params | {"scheduler": "cosine", "lr_min": kwargs["lr_min"]} 
    elif kwargs["scheduler_name"] == "multistep":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=kwargs["epochs_lr_decay"], gamma=kwargs["lr_decay"])
        model_params = model_params | {"scheduler": "cosine", "lr_decay": kwargs["lr_decay"]}
    else:
        lr_scheduler = None

    get_weight_norm(parameters)
    return optimizer, lr_scheduler, parameters



def get_weight_norm(weights):
    weight_norm = 0
    for name in weights:
        weight_norm += torch.norm(weights[name])**2
    print("weight norm: ", weight_norm.item())
    return weight_norm.item()
"""
def lora_reassign_weights(model, state_dict, r, lora_alpha, fan_in_fan_out=False, merge=True):
    is_merged = getattr(model, "is_merged", False)
    assert is_merged != merge, f'{is_merged} != {merge}: if is_merged, then must be unmerge; if not is_merged, then must merge'
    named_params = [(n, p) for n, p in model.named_parameters()]
    scaling = lora_alpha / r
    print(f'Lora configs: alpha={lora_alpha}, r={r}, scaling={scaling}')
    state_dict = {k.replace("base_model.model.", ""): v for k, v in state_dict.items()}
    replaced = set()
    merged_names = {
        # these are projector weights that got combined into single matrix in vllm
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }
    non_merged_names = ['o_proj', 'down_proj']
    for name, param in named_params:
        param.requires_grad = False
        if "_proj.weight" not in name:
            continue
        for wn, wn_series in merged_names.items():
            if name.endswith(f"{wn}.weight"):
                for stride_id, att_weight_name in enumerate(wn_series):
                    lora_a = name.replace(f"{wn}.weight", f"{att_weight_name}.lora_A.weight")
                    lora_b = name.replace(f"{wn}.weight", f"{att_weight_name}.lora_B.weight")
                    shard_size = param.shape[0] // len(wn_series)
                    if lora_a in state_dict:
                        assert lora_b in state_dict, f'{lora_b} not in state_dict'
                        assert state_dict[lora_b].shape[1] == r, f'{r=} != {state_dict[lora_b].shape}'
                        matrix = transpose(state_dict[lora_b] @ state_dict[lora_a], fan_in_fan_out) * scaling
                        assert param.data[shard_size * stride_id:shard_size * (stride_id + 1)].shape == matrix.shape
                        if merge:
                            param.data[shard_size * stride_id:shard_size * (stride_id + 1)] += matrix
                        else:
                            param.data[shard_size * stride_id:shard_size * (stride_id + 1)] -= matrix
                        replaced.add(lora_a)
                        replaced.add(lora_b)
        for wn in non_merged_names:
            if name.endswith(f"{wn}.weight"):
                lora_a = name.replace(f"{wn}.weight", f"{wn}.lora_A.weight")
                lora_b = name.replace(f"{wn}.weight", f"{wn}.lora_B.weight")
                if lora_a in state_dict:
                    assert lora_b in state_dict
                    matrix = transpose(state_dict[lora_b] @ state_dict[lora_a], fan_in_fan_out) * scaling
                    assert param.data.shape == matrix.shape, f'invalid shape: {name} {param.data.shape} != {matrix.shape}'
                    if merge:
                        param.data += matrix
                    else:
                        param.data -= matrix
                    replaced.add(lora_a)
                    replaced.add(lora_b)
    no_replaced = [k for k in state_dict.keys() if k not in replaced]
    assert len(no_replaced) == 0, f'some lora states not loaded, check again!: {no_replaced}'
    model.is_merged = merge


def lora_merge_unmerge_state_dict(llm, state_dict, peft_config, merge=True):
    # merge lora states to weights
    for worker in llm.llm_engine.workers:
        lora_reassign_weights(worker.model, state_dict, 
            r=peft_config.r, 
            lora_alpha=peft_config.lora_alpha, 
            fan_in_fan_out=peft_config.fan_in_fan_out, 
            merge=merge
        )
"""

def compute_base_proj(model, server_name):
    base_layers = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'base_layer.weight' in name:
                base_layers[name] = {}
                base_layers[name]["base_grad"] = param.grad
            elif 'lora_A' in name:
                base_name = name.replace("lora_A.{}".format(server_name), "base_layer")
                base_layers[base_name]["A"] = param.data
            elif 'lora_B' in name:
                base_name = name.replace("lora_B.{}".format(server_name), "base_layer")
                base_layers[base_name]["B"] = param.data
    
        ratio_A, ratio_B = [], []
        for name in base_layers:
            U_B, _ = torch.linalg.qr(base_layers[name]["B"])
            proj_B = (base_layers[name]["base_grad"]) @ U_B @ U_B.T
            ratio_B.append((torch.norm(proj_B) / torch.norm(base_layers[name]["base_grad"])).item())

            U_A, _ = torch.linalg.qr(base_layers[name]["A"].T)
            proj_A = U_A @ (U_A.T @ base_layers[name]["base_grad"])
            ratio_A.append((torch.norm(proj_A) / torch.norm(base_layers[name]["base_grad"])).item())
    return {"ratio_A": ratio_A, "ratio_B": ratio_B}



    