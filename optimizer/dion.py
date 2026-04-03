import torch
import copy
import math
from optimizer.load_optimizer import load_optimizer
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Utilities (mirroring the style of your federated helpers)
# ──────────────────────────────────────────────────────────────────────────────
 
def orthonormalize(P):
    """QR-based orthonormalization of columns of P[I, R]."""
    Q, _ = torch.linalg.qr(P)
    return Q
 
 
def col_norm(W):
    """Column-wise normalization of W[J, R] (ColNorm in the pseudocode)."""
    norms = torch.norm(W, dim=0, keepdim=True).clamp(min=1e-8)
    return W / norms
 
 
def power_iter1(M, V):
    """
    PowerIter1 from Algorithm 2.
 
    Args:
        M : momentum buffer  [I, J]
        V : semi-orthonormal  [J, R]
 
    Returns:
        U [I, R], W [J, R]
 
    In the real distributed setting lines 3 and 6 are AllReduce_Z calls
    (compressed DP-sync over mesh axis Z).  Here we *simulate* AllReduce by
    a plain sum — in a real multi-process run replace those lines with
    dist.all_reduce / your compression primitive.
    """
    # Line 2: P[I, R] = M[I, J] · V[J, R]
    P = M @ V                                  # [I, R]
 
    # Line 3: AllReduce_Z(P)  ── simulated (single process: no-op)
    # In distributed training: dist.all_reduce(P); P /= world_size
    P_reduced = P                              # placeholder for AllReduce
 
    # Line 4: U[I, R] = Orthonormalize(P[I, R])
    U = orthonormalize(P_reduced)              # [I, R]
 
    # Line 5: W[J, R] = M^T[J, I] · U[I, R]
    W = M.T @ U                                # [J, R]
 
    # Line 6: AllReduce_Z(W)  ── simulated (single process: no-op)
    # In distributed training: dist.all_reduce(W); W /= world_size
    W_reduced = W                              # placeholder for AllReduce
 
    return U, W_reduced
 
 
def dion_update_0d(G, M, V, beta=0.05):
    """
    DION^0D — Unsharded Dion update rule (Algorithm 2).
 
    Args:
        G    : gradient          [I, J]
        M    : momentum buffer   [I, J]  (modified in-place)
        V    : semi-orthonormal  [J, R]  (previous step)
        beta : error-feedback coefficient (default 0.05 per paper)
 
    Returns:
        O        : orthonormal update  [I, J]
        M_new    : updated momentum    [I, J]
        V_new    : updated V           [J, R]
    """
    # Line 2: accumulate gradient into momentum
    M_new = M + G                              # [I, J]
 
    # Line 3–4: rank-r approximation M ≈ U W^T via PowerIter1
    U, W = power_iter1(M_new, V)              # U: [I, R], W: [J, R]
 
    # Line 4 (main loop): error feedback
    # M_new[I, J] ← M_new[I, J] − β · U[I, R] ·_R W^T[R, J]
    M_new = M_new - beta * (U @ W.T)          # [I, J]
 
    # Line 5: column-normalise W → V_new
    V_new = col_norm(W)                        # [J, R]
 
    # Line 6: orthonormal update O = U ·_R V_new^T
    O = U @ V_new.T                            # [I, J]
 
    # Line 7: scale and return
    I_size, J_size = G.shape
    scale = math.sqrt(I_size / J_size)
    O_scaled = scale * O
 
    return O_scaled, M_new, V_new
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Simulated-distributed Dion optimizer (mirrors federated_lora_avg structure)
# ──────────────────────────────────────────────────────────────────────────────
 
def dion(
    model,
    loss_name,
    criterion,
    dion_rank,
    train_graphs,
    device,
    train_loaders,
    server_optimizer,       # used only for its lr/scheduler interface
    server_lr_scheduler,
    client_lr,
    opt_params,
    model_params,
    server_epoch,
):
    """
    Federated Dion optimizer.
 
    Each client performs `client_epoch` steps of local training, exactly as in
    federated_lora_avg.  The server then aggregates client gradients and applies
    the DION^0D update rule instead of simple FedAvg.
 
    Dion state (momentum M and semi-orthonormal V) is maintained in the
    dion_state dict keyed by server parameter name.
 
    Args mirror federated_lora_avg so the two functions are drop-in swappable.
    """
    from main import train          # same import as in your code
 
    client_num      = opt_params["client_num"]
    client_opt_name = opt_params["client_opt_name"]
    client_epoch    = opt_params["client_epoch"]
    beta = 1 - opt_params["server_momentum"]
 
    # ── 1. Collect server parameter names (same logic as federated_lora_avg) ──
    param_grads = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_grads[name] = torch.zeros_like(param)
            """
            if output_layer_name and output_layer_name in name:
                output_weights[name] = torch.zeros_like(param)
            else:
                adapter_names.append(name)
                adapter_weights[name] = torch.zeros_like(param)
            """
    # ── 2. Lazily initialise Dion state for each server parameter ────────────
    #    (momentum M and semi-orthonormal factor V)
    if not hasattr(model, "_dion_state"):
        model._dion_state = {}
 
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name not in model._dion_state:
                print(name, param.data.shape)
                I, J = param.data.shape
                R    = min(dion_rank, min(I, J))
                model._dion_state[name] = {
                    "M": torch.zeros(I, J, device=device),
                    "V": torch.nn.functional.normalize(
                        torch.randn(J, R, device=device), dim=0
                    ),
                }
 
    # ── 3. Local client training (identical to federated_lora_avg) ───────────
    client_opt_params = copy.deepcopy(opt_params)
    client_opt_params["train_stats"] = False
 
    for client_id in range(client_num):
        client_model = model   # alias
 
        client_model.train()
        optimizer, lr_scheduler, _ = load_optimizer(
            client_opt_name,
            client_model,
            client_lr,
            opt_params["client_momentum"],
            opt_params["client_weight_decay"],
            opt_params["lr_decay"],
            opt_params["epochs_lr_decay"],
            False,
            model_params,
            opt_params,
        )
        # different from muonlora, (changing adapters in place is fine)
        # notice we should not change the model in place! 
        # no optimizer.step() in train!
        assert client_opt_params["local_update_ON"] == False
        assert client_epoch == 1, "client_epoch must be 1 for dion"
        for epoch in range(client_epoch):
            try:
                train_graphs.loader_iter += 1
                assert iter(train_loaders[0]) == train_loaders[0]
                _, model_grad = train(client_model, 
                                    loss_name, 
                                    criterion, 
                                    device, 
                                    train_loaders[0], 
                                    optimizer, 
                                    lr_scheduler, 
                                    server_epoch, 
                                    client_opt_params)
            except StopIteration:
                # reinitialize iterator
                print("\nData Iterator is reloaded")
                train_graphs.loader_iter += 1
                train_loaders[0] = iter(train_loaders[1])
                _, model_grad = train(client_model, loss_name, criterion, device, train_loaders[0], optimizer, lr_scheduler, server_epoch, client_opt_params)
            
 
        # accumulate client parameters → simulate AllReduce / FedAvg
        for name, param in client_model.named_parameters():
            if param.requires_grad:
                param_grads[name] += model_grad[name]
                
 
    # average step is after the summation -- to provide more precisions
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_grads[name] = param_grads[name] / client_num

    # ── 4. Compute truncation error (same as federated_lora_avg) ─────────────
    server_optimizer.zero_grad()
    # ── 5. Apply DION^0D update rule per server parameter ────────────────────
    #
    #    In federated_lora_avg the "gradient" seen by the server is:
    #        param.data − averaged_client_param
    #    We re-use the same convention so Dion receives the same signal.
    #
    if opt_params.get("train_stats", False):
        grad_norm = 0.0
 
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        G = param_grads[name] # pseudo-gradient [I, ...]
 
        if param.dim() == 2:
            # ── Dion update for 2-D weight matrices ───────────────────────────
            state = model._dion_state[name]
            O_scaled, M_new, V_new = dion_update_0d(G, state["M"], state["V"], beta)
            state["M"] = M_new
            state["V"] = V_new
            param.grad = O_scaled.to(param.dtype)
        else:
            # ── Plain SGD pseudo-gradient for 1-D params (bias, layer norm) ───
            print("plain sgd pseudo-gradient for 1-D param", name)
            param.grad = G
 
        if opt_params.get("train_stats", False):
            grad_norm += torch.norm(param.grad).item() ** 2
 
    if opt_params.get("train_stats", False):
        train_graphs.grad_norm.append(grad_norm ** 0.5)
        print("grad norm:", train_graphs.grad_norm[-1])
 
    # ── 6. Server optimizer step (applies the Dion-computed .grad) ───────────
    server_optimizer.step()
 
    # ── 9. LR scheduler step ─────────────────────────────────────────────────
    if server_lr_scheduler is not None:
        server_lr_scheduler.step()
 
    for group in server_optimizer.param_groups:
        print("server lr", group["lr"])