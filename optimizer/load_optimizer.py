import torch
import torch.optim as optim

def load_optimizer(opt_name, model, lr, momentum, weight_decay, lr_decay, epochs_lr_decay, model_params, **kwargs):
    if opt_name == "sgd" or opt_name == "gd":
        optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    elif opt_name == "adam":
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), lr=lr, betas=(momentum, 0.99), weight_decay=weight_decay)
    elif opt_name == "sam":
        from optimizer.sam import SAM
        if kwargs["base_opt"] == "sgd":
            base_optimizer = torch.optim.SGD
            optimizer = SAM(model.parameters(), base_optimizer, rho=kwargs["sam_rho"], adaptive=kwargs["sam_adaptive"], lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif kwargs["base_opt"] == "adam":
            base_optimizer = torch.optim.Adam
            optimizer = SAM(model.parameters(), base_optimizer, rho=kwargs["sam_rho"], adaptive=kwargs["sam_adaptive"], lr=lr, betas=(momentum, 0.99), weight_decay=weight_decay)
        else:
            raise NotImplementedError
        model_params = model_params | {"base_opt": kwargs["base_opt"], "sam_rho": kwargs["sam_rho"]} 
        if kwargs["sam_adaptive"]:
            model_params = model_params | {"sam": "adaptive"}
    elif opt_name == "norm-sgd":
        from optimizer.normalized_sgd import Normalized_Optimizer
        #norm_sgd_lr = args.norm_sgd_lr
        if kwargs["base_opt"] == "sgd":
            base_optimizer = torch.optim.SGD
            optimizer = Normalized_Optimizer(model.parameters(), base_optimizer, kwargs["norm_sgd_lr"],
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
        elif kwargs["base_opt"] == "adam":
            base_optimizer = torch.optim.Adam
            optimizer = Normalized_Optimizer(model.parameters(), base_optimizer, kwargs["norm_sgd_lr"],
                                lr=lr,
                                betas=(momentum, 0.99),
                                weight_decay=weight_decay)
        model_params = model_params | {"base_opt": kwargs["base_opt"], "norm_lr": kwargs["norm_sgd_lr"]}
    elif opt_name == "goldstein":
        base_optimizer = torch.optim.SGD
        #gold_delta = args.gold_delta
        from optimizer.goldstein import Goldstein
        optimizer = Goldstein(model.parameters(), base_optimizer, delta=kwargs["gold_delta"], lr=lr, momentum=momentum, weight_decay=weight_decay)
        model_params = model_params | {"gold_delta": kwargs["gold_delta"]}

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=epochs_lr_decay,
                                                gamma=lr_decay)
    return optimizer, lr_scheduler, model_params


def load_optimizer_from_args(opt_name, model, lr, momentum, weight_decay, lr_decay, epochs_lr_decay, model_params, kwargs):
    if opt_name == "sgd" or opt_name == "gd":
        optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    elif opt_name == "sam":
        from optimizer.sam import SAM
        #if kwargs["base_opt"] == "sgd":
        if kwargs.base_opt == "sgd":
            base_optimizer = torch.optim.SGD
            optimizer = SAM(model.parameters(), base_optimizer, rho=kwargs.sam_rho, adaptive=kwargs.sam_adaptive, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif kwargs.base_opt == "adam":
            base_optimizer = torch.optim.Adam
            optimizer = SAM(model.parameters(), base_optimizer, rho=kwargs.sam_rho, adaptive=kwargs.sam_adaptive, lr=lr, betas=(momentum, 0.99), weight_decay=weight_decay)
        else:
            raise NotImplementedError
        model_params = model_params | {"base_opt": kwargs.base_opt, "sam_rho": kwargs.sam_rho} 
    elif opt_name == "norm-sgd":
        from optimizer.normalized_sgd import Normalized_Optimizer
        #norm_sgd_lr = args.norm_sgd_lr
        if kwargs.base_opt == "sgd":
            base_optimizer = torch.optim.SGD
            optimizer = Normalized_Optimizer(model.parameters(), base_optimizer, kwargs.norm_sgd_lr,
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
        elif kwargs.base_opt == "adam":
            base_optimizer = torch.optim.Adam
            optimizer = Normalized_Optimizer(model.parameters(), base_optimizer, kwargs.norm_sgd_lr,
                                lr=lr,
                                betas=(momentum, 0.99),
                                weight_decay=weight_decay)
        model_params = model_params | {"base_opt": kwargs.base_opt, "norm_lr": kwargs.norm_sgd_lr}
    elif opt_name == "goldstein":
        base_optimizer = torch.optim.SGD
        #gold_delta = args.gold_delta
        from optimizer.goldstein import Goldstein
        optimizer = Goldstein(model.parameters(), base_optimizer, delta=kwargs.gold_delta, lr=lr, momentum=momentum, weight_decay=weight_decay)
        model_params = model_params | {"gold_delta": kwargs.gold_delta}

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=epochs_lr_decay,
                                                gamma=lr_decay)
    return optimizer, lr_scheduler, model_params