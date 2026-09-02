import torch
import copy
import time

import numpy as np

from optimizer.load_optimizer import load_optimizer


def collect_client_grads(
    model,
    loss_name,
    criterion,
    train_graphs,
    device,
    train_loaders,
    client_lr,
    opt_params,
    model_params,
    server_epoch,
    on_client_grad,
    exclude_from_copy=(),
):
    """Run one federated round and hand each client's gradients to a callback.

    Every selected client differentiates the *shared* model and never takes a
    local step (local_update_ON must be False), so `model_grad` is a clean
    stochastic gradient at the current server iterate.  `on_client_grad` is
    called once per client as on_client_grad(client_id, model_grad).

    Nothing is accumulated here on purpose: callers differ in what they want
    (a dense sum, a compressed message, a pair of thin factors) and in the
    dtype they want to accumulate in.

    `exclude_from_copy` names opt_params keys to keep out of the per-client
    deepcopy -- large server-side state that must not be duplicated.

    Returns the number of participating clients.
    """
    from main import train

    client_opt_name = opt_params["client_opt_name"]
    client_epoch = opt_params["client_epoch"]

    if opt_params["client_partial"] < 1:
        client_num = int(opt_params["client_partial"] * opt_params["client_num"])
        client_selected = np.random.choice(opt_params["client_num"], client_num, replace=False)
    else:
        client_num = opt_params["client_num"]
        client_selected = np.arange(client_num)

    assert opt_params.get("local_update_ON") is False, \
        "collect_client_grads needs local_update_ON=False so clients do not step"

    stashed = {k: opt_params.pop(k) for k in exclude_from_copy if k in opt_params}
    client_opt_params = copy.deepcopy(opt_params)
    opt_params.update(stashed)
    client_opt_params["train_stats"] = False

    training_time_accumulated = 0
    for client_id in client_selected:
        client_model = model  # alias -- no deepcopy
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
                # train_loaders[0] is an iterator in the LLM pipelines (hence the
                # StopIteration handler) and a plain DataLoader elsewhere.
                start_time = time.time()
                _, model_grad = train(
                    client_model, loss_name, criterion, device, train_loaders[0],
                    optimizer, lr_scheduler, server_epoch, client_opt_params,
                )
                end_time = time.time()
                training_time_accumulated += end_time - start_time
                print(f"Time taken for client {client_id}: {end_time - start_time:.3f}s")
            except StopIteration:
                print("\nData Iterator is reloaded")
                train_graphs.loader_iter += 1
                train_loaders[0] = iter(train_loaders[1])
                _, model_grad = train(
                    client_model, loss_name, criterion, device, train_loaders[0],
                    optimizer, lr_scheduler, server_epoch, client_opt_params,
                )

        on_client_grad(client_id, model_grad)
        del model_grad

    print(f"Total training time: {training_time_accumulated:.3f}s")
    return client_num


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

    assert opt_params["client_early_stop"] == 0

    # ── 1. Accumulator: one zero tensor per trainable param ─────────────────
    # fp32 regardless of the model dtype: summing many bf16 gradients loses a
    # large fraction of the small ones to round-off.
    param_grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_grads[name] = torch.zeros_like(param, dtype=torch.float32)

    # ── 2. Local client training ─────────────────────────────────────────────
    def _accumulate(client_id, model_grad):
        for name in param_grads:
            param_grads[name] += model_grad[name].float()

    client_num = collect_client_grads(
        model, loss_name, criterion, train_graphs, device, train_loaders,
        client_lr, opt_params, model_params, server_epoch, _accumulate,
    )

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