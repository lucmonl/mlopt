import torch
import torch.optim as optim
import warmup_scheduler

def load_optimizer(opt_name, model, lr, momentum, weight_decay, lr_decay, epochs_lr_decay, model_params, **kwargs):
    if opt_name == "sgd" or opt_name == "gd":
        optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    elif opt_name == "adam":
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    elif opt_name == "sam":
        from optimizer.sam import SAM
        if kwargs["base_opt"] == "sgd":
            base_optimizer = torch.optim.SGD
            optimizer = SAM(model.parameters(), base_optimizer, rho=kwargs["sam_rho"], adaptive=kwargs["sam_adaptive"], train_stats=kwargs["train_stats"], lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif kwargs["base_opt"] == "adam":
            base_optimizer = torch.optim.Adam
            optimizer = SAM(model.parameters(), base_optimizer, rho=kwargs["sam_rho"], adaptive=kwargs["sam_adaptive"], train_stats=kwargs["train_stats"], lr=lr, betas=(momentum, 0.99), weight_decay=weight_decay)
        else:
            raise NotImplementedError
        model_params = model_params | {"base_opt": kwargs["base_opt"], "sam_rho": kwargs["sam_rho"]} 
        if kwargs["sam_adaptive"]:
            model_params = model_params | {"sam": "adaptive"}
    elif opt_name == "sam_on":
        from optimizer.sam_on import SAM_ON
        if kwargs["base_opt"] == "sgd":
            base_optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer = SAM_ON(optimizer=base_optimizer, model=model, rho=kwargs["sam_rho"], adaptive=kwargs["sam_adaptive"], only_norm=True)
        model_params = model_params | {"base_opt": kwargs["base_opt"], "sam_rho": kwargs["sam_rho"]} 
        if kwargs["sam_adaptive"]:
            model_params = model_params | {"sam": "adaptive"}
    elif opt_name == "replay_sam":
        from optimizer.sam import Replay_SAM
        if kwargs["base_opt"] == "sgd":
            base_optimizer = torch.optim.SGD
            optimizer = Replay_SAM(model.parameters(), base_optimizer, rho=kwargs["sam_rho"], adaptive=kwargs["sam_adaptive"], train_stats=kwargs["train_stats"], lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif kwargs["base_opt"] == "adam":
            base_optimizer = torch.optim.Adam
            optimizer = Replay_SAM(model.parameters(), base_optimizer, rho=kwargs["sam_rho"], adaptive=kwargs["sam_adaptive"], train_stats=kwargs["train_stats"], lr=lr, betas=(momentum, 0.99), weight_decay=weight_decay)
        else:
            raise NotImplementedError
        model_params = model_params | {"base_opt": kwargs["base_opt"], "sam_rho": kwargs["sam_rho"]} 
        if kwargs["sam_adaptive"]:
            model_params = model_params | {"sam": "adaptive"}
    elif opt_name.startswith("look_sam"):
        from optimizer.sam import LookSAM
        if kwargs["base_opt"] == "sgd":
            base_optimizer = torch.optim.SGD
            optimizer = LookSAM(alpha=kwargs["look_alpha"], params=model.parameters(), base_optimizer=base_optimizer, rho=kwargs["sam_rho"], lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif kwargs["base_opt"] == "adam":
            base_optimizer = torch.optim.Adam
            optimizer = LookSAM(alpha=kwargs["look_alpha"], params=model.parameters(), base_optimizer=base_optimizer, rho=kwargs["sam_rho"], lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise NotImplementedError
        model_params = model_params | {"base_opt": kwargs["base_opt"], "sam_rho": kwargs["sam_rho"], "look_alpha": kwargs["look_alpha"]} 
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
    elif opt_name == "federated":
        optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=kwargs['client_momentum'],
                            weight_decay=weight_decay)
        model_params = model_params | {'server_opt': kwargs['server_opt_name'], 'client_opt': kwargs['client_opt_name'], 'client_lr': kwargs['client_lr'], 'client_momentum': kwargs['client_momentum'],
                                       "client_num": kwargs['client_num'], 'client_epoch': kwargs['client_epoch'], 'sketch_size': kwargs['sketch_size']}
    else:
        raise NotImplementedError
    """
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=epochs_lr_decay,
                                                gamma=lr_decay)
    """
    if kwargs["scheduler_name"] == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs["epoch"], eta_min=kwargs["lr_min"])
        model_params = model_params | {"scheduler": "cosine", "lr_min": kwargs["lr_min"]} 
    elif kwargs["scheduler_name"] == "multistep":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=epochs_lr_decay, gamma=lr_decay)
        model_params = model_params | {"scheduler": "cosine", "lr_decay": lr_decay}
    else:
        lr_scheduler = None
    #self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=self.hparams.warmup_epoch, after_scheduler=self.base_scheduler)
    lr_scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=5, after_scheduler=lr_scheduler)
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