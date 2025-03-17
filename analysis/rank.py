import torch

def compute_effecitve_rank(W, threshold=0.9):
    assert threshold < 1.0
    with torch.no_grad():
        U, Sigma, VT = torch.linalg.svd(W)
        sigma_sq = Sigma**2
        sigma_sq_sum = torch.sum(sigma_sq)
        sum = 0
        for i in range(sigma_sq.shape[0]):
            sum += sigma_sq[i]
            if sum >= threshold * sigma_sq_sum:
                break
    return i

def compute_stable_rank(W):
    return (torch.norm(W)**2 / torch.linalg.norm(W, ord=2)**2).item()

def compute_effective_rank(A, B, threshold=0.9):
    return compute_effecitve_rank(B @ A, threshold)

def compute_stable_rank_lora(A, B):
    return compute_stable_rank(B @ A)

def get_lora_eff_rank(graphs, model):
    adapter_weights = {}
    adapter_names = []

    for name, param in model.named_parameters():
        # select lora_A and lora_B
        if param.requires_grad and "lora" in name:
            adapter_names.append(name)
            adapter_weights[name] = param

    for i in range(0, len(adapter_names), 2):
        lora_A_name, lora_B_name = adapter_names[i], adapter_names[i+1]
        eff_rank = compute_effective_rank(adapter_weights[lora_A_name], adapter_weights[lora_B_name])
        graphs.effective_rank.append(eff_rank)
    print(graphs.effective_rank)

def get_lora_stable_rank(graphs, model):
    adapter_weights = {}
    adapter_names = []

    for name, param in model.named_parameters():
        # select lora_A and lora_B
        if param.requires_grad and "lora" in name:
            adapter_names.append(name)
            adapter_weights[name] = param

    for i in range(0, len(adapter_names), 2):
        lora_A_name, lora_B_name = adapter_names[i], adapter_names[i+1]
        eff_rank = compute_stable_rank_lora(adapter_weights[lora_A_name], adapter_weights[lora_B_name])
        graphs.stable_rank.append(eff_rank)
        print(eff_rank)
    print(graphs.stable_rank)