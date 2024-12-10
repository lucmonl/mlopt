import sys
import torch
import torch.optim as optim

def add_adapters_dataset(model_name, model, lora_rank, lora_alpha, lora_freeze_a=False):
    if model_name == "google/vit-base-patch16-224-in21k":
        add_adapters(model, lora_rank, lora_alpha, "classifier", ["query", "value"])
        return 'classifier'
    elif model_name == 'flair':
        add_adapters(model, lora_rank, lora_alpha, "classifier", ["query", "value"])
        # add_adapters(model, lora_rank, lora_alpha, 'classifier', ['convolution'])
    elif model_name == 'gpt2':
        add_adapters(model, lora_rank, lora_alpha, "score", ["c_attn", "c_proj", "c_fc"], freeze_a=lora_freeze_a)
        return "score"
    elif model_name == 'reddit':
        add_adapters(model, lora_rank, lora_alpha, None, ["c_attn", "c_proj", "c_fc"])
    elif model_name == "roberta-base":
        add_adapters(model, lora_rank, lora_alpha, "classifier", ["query", "value"])
        return 'classifier'
    elif model_name == "akjindal53244/Arithmo-Mistral-7B":
        # from https://huggingface.co/upaya07/Arithmo2-Mistral-7B-adapter/blob/main/adapter_config.json
        add_adapters(model, lora_rank, lora_alpha, None, ["o_proj", "q_proj", "v_proj", "down_proj", "up_proj", "k_proj", "gate_proj"], task_type="CAUSAL_LM")
        return None

def add_ft(model, output_layer_name, target_modules):
    for n, p in model.named_parameters():
        if output_layer_name in n:
            p.requires_grad = True
        else:
            require_grad = False
            for module_name in target_modules:
                if module_name in n and not require_grad:
                    require_grad = True
            p.requires_grad = require_grad
                

def add_adapters(model, lora_rank, lora_alpha, output_layer_name, target_modules, freeze_a=False, task_type=None):
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
        model = get_peft_model(model, config)
        if freeze_a:
            for n, p in model.named_parameters():
                if "lora_A" in n:
                    p.requires_grad=False
    elif lora_rank == 0: # linear fine-tune
        for n,p in model.named_parameters():
            if output_layer_name in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
    elif lora_rank == -1: #full fine-tune
        add_ft(model, output_layer_name, target_modules)
    for name, param in model.named_parameters():
        print(name, param.shape, param.requires_grad)

def load_server_optimizer(model, lr, momentum, weight_decay, model_params, **kwargs):
    from torch.nn.parameter import Parameter
    parameters = {}
    adapter_names = []
    base_names = []

    output_layer_name = kwargs["output_layer_name"]
    for name, param in model.named_parameters():
        # select lora_A and lora_B
        if param.requires_grad:
            adapter_names.append(name)
    #output_layer_name = adapter_names[-1]

    for i in range(0, len(adapter_names), 2):
        lora_A_name = adapter_names[i]
        base_weight_name = lora_A_name.replace("lora_A.default", "base_layer")
        base_names.append(base_weight_name)

    for name, param in model.named_parameters():
        if name in base_names:
            parameters[name] = Parameter(torch.zeros_like(param).to(kwargs["device"]))
        elif output_layer_name in name:
            parameters[name] = param

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
    
    return optimizer, lr_scheduler, parameters