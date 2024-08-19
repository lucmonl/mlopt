import sys

def add_adapters_dataset(dataset, model, lora_rank, lora_alpha):
    if dataset == 'cifar10':
        add_adapters(model, lora_rank, lora_alpha, "classifier", ["query", "value"])
    elif dataset == 'flair':
        add_adapters(model, lora_rank, lora_alpha, "classifier", ["query", "value"])
        # add_adapters(model, lora_rank, lora_alpha, 'classifier', ['convolution'])
    elif dataset == '20newsgroups':
        add_adapters(model, lora_rank, lora_alpha, "score", ["c_attn", "c_proj", "c_fc"])
        return "score"
    elif dataset == 'reddit':
        add_adapters(model, lora_rank, lora_alpha, None, ["c_attn", "c_proj", "c_fc"])

def add_adapters(model, lora_rank, lora_alpha, output_layer_name, target_modules):
    from peft import LoraConfig, get_peft_model

    if lora_rank > 0:
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=[output_layer_name] if output_layer_name is not None else [],
        )
        model = get_peft_model(model, config)
    elif lora_rank == 0: # linear fine-tune
        for n,p in model.named_parameters():
            if output_layer_name in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
    #if lora_rank == -1: full fine-tune
    #for name, param in model.named_parameters():
    #    print(name, param.shape, param.requires_grad)
    #sys.exit()
    
                