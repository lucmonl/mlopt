import torch
import copy
import time
from optimizer.load_optimizer import load_optimizer


def federated_train(
    model,
    loss_name,
    criterion,
    train_graphs,
    device,
    train_loaders,
    server_optimizer,
    server_lr_scheduler,
    client_lr,
    opt_params,
    model_params,
    server_epoch,
):
    """
    FedAvg-style federated training using model_grad directly.

    Mirrors dion() in structure: instead of computing pseudo-gradients as
    (old_params - new_params), we call train() with local_update_ON=False and
    use the per-parameter gradients it returns via model_grad.

    This avoids deepcopy of the model and the parameter-vector round-trip.
    """
    #print("In single step federated train using model_grad.")

    from main import train  # same import as in dion()

    client_num      = opt_params["client_num"]
    client_opt_name = opt_params["client_opt_name"]
    client_epoch    = opt_params["client_epoch"]

    if opt_params["client_partial"] < 1:
        client_num = int(opt_params["client_partial"] * client_num)
        client_selected = list(
            __import__("numpy").random.choice(opt_params["client_num"], client_num, replace=False)
        )
    else:
        client_selected = list(range(client_num))

    # ── 1. Accumulator: one zero tensor per trainable param ─────────────────
    param_grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_grads[name] = torch.zeros_like(param)

    # ── 2. Local client training ─────────────────────────────────────────────
    client_opt_params = copy.deepcopy(opt_params)
    client_opt_params["train_stats"] = False
    # do NOT call optimizer.step() inside train(); we aggregate first
    assert client_opt_params.get("local_update_ON") == False, \
        "local_update_ON must be False for model_grad-based federated_train"

    training_time_accumulated = 0

    for client_id in client_selected:
        client_model = model  # alias – no deepcopy, mirroring dion()

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
            client_opt_params,
        )

        for epoch in range(client_epoch):
            try:
                train_graphs.loader_iter += 1
                assert iter(train_loaders[0]) == train_loaders[0]
                start_time = time.time()
                _, model_grad = train(
                    client_model,
                    loss_name,
                    criterion,
                    device,
                    train_loaders[0],
                    optimizer,
                    lr_scheduler,
                    server_epoch,
                    client_opt_params,
                )
                end_time = time.time()
                training_time_accumulated += end_time - start_time
                print(f"Time taken for client {client_id}: {end_time - start_time:.3f}s")
            except StopIteration:
                print("\nData iterator is reloaded")
                train_graphs.loader_iter += 1
                train_loaders[0] = iter(train_loaders[1])
                _, model_grad = train(
                    client_model,
                    loss_name,
                    criterion,
                    device,
                    train_loaders[0],
                    optimizer,
                    lr_scheduler,
                    server_epoch,
                    client_opt_params,
                )

        # accumulate per-parameter gradients
        for name, param in client_model.named_parameters():
            if param.requires_grad:
                param_grads[name] += model_grad[name]

    print(f"Total training time: {training_time_accumulated:.3f}s")

    # ── 3. Average across clients ────────────────────────────────────────────
    for name in param_grads:
        param_grads[name] = param_grads[name] / client_num

    # ── 4. Set averaged gradients and run server optimizer step ─────────────
    server_optimizer.zero_grad()

    if opt_params.get("train_stats", False):
        grad_norm = 0.0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        G = param_grads[name]

        if opt_params.get("clip_tau", -1) != -1:
            G = torch.clamp(G, min=-opt_params["clip_tau"], max=opt_params["clip_tau"])

        param.grad = G.to(param.dtype)

        if opt_params.get("train_stats", False):
            grad_norm += torch.norm(param.grad).item() ** 2

    if opt_params.get("train_stats", False):
        train_graphs.grad_norm.append(grad_norm ** 0.5)
        print("grad norm:", train_graphs.grad_norm[-1])

    server_optimizer.step()

    # ── 5. LR scheduler step ────────────────────────────────────────────────
    if server_lr_scheduler is not None:
        server_lr_scheduler.step()

    for group in server_optimizer.param_groups:
        print("server lr", group["lr"])