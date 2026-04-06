


from transformers import AutoModelForCausalLM
from peft import PeftModel
from peft import LoraConfig, get_peft_model, LoraModel
import torch
import copy

# 1. Define your local paths
base_model_path = "meta-llama/Llama-3.2-3B"  # or "meta-llama/Llama-2-7b-hf"
lora_adapter_path = "/u/lucmon/mlopt/results/oasst2/HF_CrossEntropy/federated/meta-llama/Llama-3.2-3B/task_name_en/lora_rank_16/lora_alpha_16/fedlora_avg_muonlora_v9/server_opt_sgd/client_opt_sgd/client_lr_1.0/client_momentum_0.0/client_weight_decay_0.0/client_num_64/client_epoch_1/client_round_0/sketch_size_-1/scheduler_cosine/lr_min_1e-05/lr_0.002/moment_0.7/wd_0.0/batch_size_2/epoch_10000/run_0"
#lora_adapter_path = "/u/lucmon/mlopt/results/oasst2/HF_CrossEntropy/federated/meta-llama/Llama-3.2-3B/task_name_en/lora_rank_-1/lora_alpha_16/server_opt_dion/client_opt_sgd/client_lr_1.0/client_momentum_0.0/client_weight_decay_0.0/client_num_64/client_epoch_1/client_round_0/sketch_size_-1/dion_rank_16/scheduler_cosine/lr_min_1e-05/lr_0.001/moment_0.95/wd_0.0/batch_size_2/epoch_10000/run_0/"
compare_with = "muon"


if compare_with == "muon":
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model = copy.deepcopy(base_model)
    # 2. Load the base model
    """
    model = AutoModelForCausalLM.from_pretrained(
        lora_adapter_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    """
    config = LoraConfig(
        r=16, 
        lora_alpha=16, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Ensure these match your training
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 3. Wrap the model with the config
    #model = get_peft_model(model, config)
    model = LoraModel(model, config, adapter_name="server")
    checkpoint = torch.load(lora_adapter_path + "/model.ckpt", weights_only=True)
    #base_model = PeftModel.from_pretrained(model, config)
    model.load_state_dict(checkpoint, strict=False)
    model.merge_and_unload()
    start = len('model.')
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    checkpoint = torch.load(lora_adapter_path + "/model.ckpt", weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    start = 0

# 3. Load the LoRA adapter from the local path
# This looks for 'adapter_config.json' and 'adapter_model.bin' in the folder
#model = PeftModel.from_pretrained(model, lora_adapter_path)

print("PEFT model weights:")
peft_weights = {}
for name, param in model.named_parameters():
    #print(name, param.shape, param.data.norm())
    peft_weights[name[start:]] = param

print("Base model weights:")
"""
for name, param in base_model.named_parameters():
    print(name, param.shape, param.data.norm())
"""

for name, param in base_model.named_parameters():
    if param.ndim != 2:
        continue
    print(name, param.shape, (param.data - peft_weights[name]).norm())
    weight_update = (param.data - peft_weights[name]).to(torch.float32)
    stable_rank = torch.linalg.norm(weight_update) / torch.linalg.matrix_norm(weight_update, ord=2)
    print(f"stable rank: {stable_rank}")
