
from tqdm.auto import tqdm
import math
import peft
import torch
import types
from peft.utils import _get_submodules
import torch.nn.functional as F
from torch.nn import init


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

def get_svd_grad(input_matrix, rank, n_iter=10):
    """Use PyTorch's SVD which can utilize GPU acceleration"""
    
    # Handle meta tensors
    if hasattr(input_matrix, 'is_meta') and input_matrix.is_meta:
        input_matrix = input_matrix.to('cpu')
        input_matrix = input_matrix.clone().detach()
    
    # Convert to torch tensor if not already
    if not torch.is_tensor(input_matrix):
        input_matrix = torch.from_numpy(input_matrix)
    
    # Move to GPU 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_matrix = input_matrix.to(device)
    
    # Temporarily convert to float32 for SVD
    input_matrix_float = input_matrix.to(torch.float32)
    
    # Create random tensor in same dtype
    torch.manual_seed(42)  # for reproducibility
    size = input_matrix.size(1)
    R = torch.randn(size, rank, device=device, dtype=torch.float32)
    
    # Compute SVD with controlled dtypes
    U, s, V = torch.svd_lowrank(input_matrix_float, q=rank, niter=n_iter)
    
    # Convert results back to bfloat16
    U = U.to(torch.bfloat16)
    s = s.to(torch.bfloat16)
    V = V.to(torch.bfloat16)
    diag_s = torch.diag(s)
    
    # remove intermediate tensors from memory
    del input_matrix, input_matrix_float, R
    
    return U, diag_s, V.T

def replace_module_weights(target_module, new_weight):
    device = target_module.weight.device
    target_module.weight = torch.nn.Parameter(new_weight)

    # dispatch to correct device
    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)

def get_replacement_module(weight, module_name, type, lora_rank, init_scaling=1):
    if type == 'svd':
        Ur, sr, Vrt = get_svd_grad(weight.detach(), lora_rank)
        final_enc = torch.tensor(Ur @ sr, dtype=weight.dtype, device=weight.device)
        final_dec = torch.tensor(Vrt, dtype=weight.dtype, device=weight.device)
    else:
        raise NotImplementedError(f"{type} is currently not supported.")
    return final_enc, final_dec


def forward_latent(self, x: torch.Tensor):
    previous_dtype = x.dtype

    if self.active_adapter[0] not in self.lora_A.keys():
        return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    if self.disable_adapters:
        if self.r[self.active_adapter[0]] > 0 and self.merged:
            self.unmerge()
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    elif self.r[self.active_adapter[0]] > 0 and not self.merged:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        x = x.to(self.lora_A[self.active_adapter[0]].weight.dtype)

        # adding latent_mapping in the forward loop
        result += (
            self.lora_B[self.active_adapter[0]](
                self.default_lora_latent_mapping(
                    self.lora_A[self.active_adapter[0]](self.lora_dropout[self.active_adapter[0]](x))
                )
            )
            * self.scaling[self.active_adapter[0]]
        )
    else:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    result = result.to(previous_dtype)

    return result

def get_delta_weight(self, adapter) -> torch.Tensor:
    """
    Compute the delta weight for the given adapter.

    Args:
        adapter (str):
            The name of the adapter for which the delta weight should be computed.
    """
    device = self.lora_B[adapter].weight.device
    dtype = self.lora_B[adapter].weight.dtype

    # Handle CPU + float16/bfloat16 case
    cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

    weight_A = self.lora_A[adapter].weight
    weight_B = self.lora_B[adapter].weight
    latent_weight = self.default_lora_latent_mapping.weight

    if cast_to_fp32:
        weight_A = weight_A.float()
        weight_B = weight_B.float()
        latent_weight = latent_weight.float()
    else:
        # Ensure all tensors are same dtype (bfloat16 in your case)
        weight_A = weight_A.to(dtype)
        weight_B = weight_B.to(dtype)
        latent_weight = latent_weight.to(dtype)

    output_tensor = transpose(
        weight_B @ latent_weight @ weight_A,
        self.fan_in_fan_out
    ) * self.scaling[adapter]

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)
        # cast back the weights
        self.lora_A[adapter].weight.data = weight_A.to(dtype)
        self.lora_B[adapter].weight.data = weight_B.to(dtype)
        self.default_lora_latent_mapping.weight.data = latent_weight.to(dtype)

    return output_tensor


def init_module_weights(target_module: torch.nn.Linear, sigma: float):
    # Initialize weights with Gaussian distribution
    torch.nn.init.normal_(target_module.weight, mean=0, std=sigma)
    if hasattr(target_module, "bias"):
        # Set bias to zeros
        if target_module.bias is not None:
            torch.nn.init.zeros_(target_module.bias)

def kaiming_uniform_init(matrix: torch.tensor):
    init.kaiming_uniform_(matrix, a=math.sqrt(5))
    return matrix

def find_and_initialize(model, peft_config, lora_rank, reconstr_type="svd"):
    """
    :param adapter_name: options: 'default'
    :param reconstr_type: options: 'svd'
    """
    half_init_dec = False
    replacement_module_random_init =  True #reconstruct_config['replacement_module_random_init']
    reconstruction_mode = "separated" #reconstruct_config['reconstr_mode']
    #lora_config = peft_config[adapter_name]
    lora_config = peft_config
    r_squared = True #reconstruct_config['r_squared']  # whether using r*r matrix between lora_A and lora_B or not
    is_target_modules_in_base_model = False
    key_list = [key for key, _ in model.named_modules()]
    assert (not isinstance(lora_config.target_modules, str))
    print("Iterating through model's specified modules to initialize A/B matrices.")
    for key in tqdm(key_list):
        target_module_found = any(key.endswith('.' + target_key) for target_key in lora_config.target_modules)
        if target_module_found:
            if not is_target_modules_in_base_model:
                is_target_modules_in_base_model = True
            _, target, target_name = _get_submodules(model, key)
           
            if reconstruction_mode == 'separated':
                replacement_encoder_weight, replacement_decoder_weight = get_replacement_module(weight=target.weight.T,
                                                                                                    module_name=key,
                                                                                                    type=reconstr_type,         
                                                                                                    lora_rank=lora_rank)

                    
                   
                if not isinstance(target, peft.tuners.lora.Linear):
                    raise NotImplementedError('Only initialization for peft.tuners.lora.Linear type is implemented.')
                    # TODO implement for Linear8bitLt
                else:
                    if half_init_dec:
                        pass
                        #kaiming_uniform_init_lower_half(replacement_decoder_weight)
                    if replacement_module_random_init:
                        #pass
                        kaiming_uniform_init(replacement_encoder_weight)
                        kaiming_uniform_init(replacement_decoder_weight)
                    replace_module_weights(target.lora_B.default, replacement_decoder_weight.T)
                    if r_squared:
                        target.forward = types.MethodType(forward_latent, target)
                        target.get_delta_weight = types.MethodType(get_delta_weight, target)
                        replace_module_weights(target.lora_A.default, replacement_encoder_weight.T)
                        target.default_lora_latent_mapping = torch.nn.Linear(lora_config.r, lora_config.r, bias=False)
                        init_module_weights(target.default_lora_latent_mapping, sigma=0.00001)
                        target.default_lora_latent_mapping.to(target.lora_A.default.weight.device)

                        target.lora_A.default.weight.requires_grad = False  # only the r*r matrix will be tuned
                        target.lora_B.default.weight.requires_grad = False  # only the r*r matrix will be tuned

                    else:
                        init_module_weights(target.lora_A.default, sigma=0.00001)

            else:
                raise NotImplementedError("The only supported mode is: separated.")

    if not is_target_modules_in_base_model:
        raise ValueError(
            f"Target modules {lora_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )
    

def federated_lora_fedsb(model, loss_name, criterion, lora_rank, train_graphs, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, opt_params, model_params, server_epoch):
    from optimizer.load_optimizer import load_optimizer
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
        client_model = copy.deepcopy(model)
        client_model = model #alias

        client_model.train()
        optimizer, lr_scheduler, _= load_optimizer(client_opt_name, client_model, client_lr, opt_params["client_momentum"], opt_params["client_weight_decay"], opt_params["lr_decay"], opt_params["epochs_lr_decay"], False, model_params, opt_params)
        #vector_to_parameters(old_params, client_model.parameters())
        for epoch in range(client_epoch):
            train(client_model, loss_name, criterion, device, train_loaders[client_id], optimizer, lr_scheduler, server_epoch, client_opt_params)
            
        for name, param in client_model.named_parameters():
            if param.requires_grad:
                #param_names.append(name)
                #server_adapter_name = name.replace("{}".format(adapter_name), opt_params["server_name"])
                if output_layer_name and output_layer_name in name:
                    output_weights[name] += param.data / client_num
                else:
                    if name in adapter_weights:
                        row, col = param.data.shape
                        adapter_weights[name][:row, :col] += param.data/client_num
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
    if server_lr_scheduler is not None:
        server_lr_scheduler.step()

    for group in server_optimizer.param_groups:
        print("server lr", group['lr'])