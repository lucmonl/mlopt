import torch
import torch.optim as optim
import warmup_scheduler
import sys

def load_optimizer_param(opt_name, model, lr, momentum, weight_decay, lr_decay, epochs_lr_decay, warm_start, model_params, **kwargs):
    if kwargs['cls_lr'] == -1:
        return load_optimizer(opt_name, model, lr, momentum, weight_decay, lr_decay, epochs_lr_decay, warm_start, model_params, **kwargs)

    if opt_name == "federated":
        opt_name = kwargs["server_opt_name"]
        cls_lr = kwargs['cls_lr']
        model_params = model_params | {"cls_lr": kwargs['cls_lr'], 'server_opt': kwargs['server_opt_name'], 'client_opt': kwargs['client_opt_name'], 'client_lr': kwargs['client_lr'], 'client_momentum': kwargs['client_momentum'],
                                       "client_weight_decay": kwargs['client_weight_decay'], "client_num": kwargs['client_num'], 'client_epoch': kwargs['client_epoch'], 'sketch_size': kwargs['sketch_size']}
    else:
        cls_lr = lr

    cls_params = list(map(lambda x: x[1],list(filter(lambda kv: kwargs["output_layer_name"] in kv[0] and kv[1].requires_grad, model.named_parameters()))))
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kwargs["output_layer_name"] not in kv[0] and kv[1].requires_grad, model.named_parameters()))))

    if opt_name == "sgd" or opt_name == "gd":
        optimizer = optim.SGD([{'params': base_params}, {'params': cls_params, 'lr': cls_lr}],
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    else:
        raise NotImplementedError

    if kwargs["scheduler_name"] == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs["epoch"], eta_min=kwargs["lr_min"])
        model_params = model_params | {"scheduler": "cosine", "lr_min": kwargs["lr_min"]} 
    elif kwargs["scheduler_name"] == "multistep":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=epochs_lr_decay, gamma=lr_decay)
        model_params = model_params | {"scheduler": "cosine", "lr_decay": lr_decay}
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler, model_params


def load_optimizer(opt_name, model, lr, momentum, weight_decay, lr_decay, epochs_lr_decay, warm_start, model_params, kwargs):
    if opt_name == "federated":
        if kwargs["server_opt_name"] == "clip_sgd":
            opt_name = "sgd"
            assert kwargs["clip_tau"] != -1
            #model_params = model_params | {'clip': kwargs['clip_tau']}
        else:
            opt_name = kwargs["server_opt_name"]

        if kwargs['use_ef'] != 0:
            kwargs["error_feedback"] = {}
            model_params = model_params | {"use_ef": kwargs['use_ef']}
        #opt_name = "sgd" if kwargs["server_opt_name"] == "clip_sgd" else kwargs["server_opt_name"] 
        #weight_decay = 0.0
        model_params = model_params | {'server_opt': kwargs['server_opt_name'], 'client_opt': kwargs['client_opt_name'], 'client_lr': kwargs['client_lr'], 'client_momentum': kwargs['client_momentum'],
                                       "client_weight_decay": kwargs['client_weight_decay'], "client_num": kwargs['client_num'], 'client_epoch': kwargs['client_epoch'], 'sketch_size': kwargs['sketch_size']}
        if kwargs["client_partial"] < 1:
            model_params = model_params | {"client_partial": kwargs["client_partial"]}
        if kwargs["privacy_clip"] != -1:
            model_params = model_params | {"privacy_clip": kwargs["privacy_clip"], "privacy_noise": kwargs["privacy_noise"]}
    
    if kwargs["clip_tau"] != -1:
        model_params = model_params | {'clip': kwargs['clip_tau']}
    if kwargs["switch_epoch"] != -1:
        model_params = model_params | {"switch_epoch": kwargs["switch_epoch"]}
    
    if opt_name == "sgd" or opt_name == "gd":
        optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    elif opt_name in ["adam", "adam_ctsk"]: 
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    elif opt_name in ["amsgrad", "amsgrad_ctsk"]:
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay, amsgrad=True)
    elif opt_name == "amsgradw":
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay, amsgrad=True)
    elif opt_name == "adamw":
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    elif opt_name == "adams_v1":
        from optimizer.sam_adam import AdamS_v1
        optimizer = AdamS_v1(model.parameters(), alpha=kwargs["sam_alpha"], lr=lr, momentum=momentum, weight_decay=weight_decay)
        model_params = model_params | {"alpha":kwargs["sam_alpha"], "sam_rho": kwargs["sam_rho"]} 
    elif opt_name == "sophia":
        from optimizer.sophia import SophiaG
        optimizer = SophiaG(model.parameters(), lr=lr,betas=(momentum, 0.99),rho=kwargs["sophia_rho"], weight_decay=weight_decay)
        model_params = model_params | {'sophia_rho': kwargs['sophia_rho'], 'hess_interval': kwargs['hess_interval']}
    elif opt_name == "sophus":
        from optimizer.sophus import Sophus
        optimizer = Sophus(model.parameters(), lr=lr,betas=(momentum, 0.99),rho=kwargs["sophia_rho"], weight_decay=weight_decay, rank=kwargs["sophus_rank"])
        model_params = model_params | {'sophia_rho': kwargs['sophia_rho'], "sophus_rank": kwargs['sophus_rank'], 'hess_interval': kwargs['hess_interval']}
    elif opt_name == "dom_sgd":
        """from paper Does SGD really happen in tiny subspaces? https://arxiv.org/pdf/2405.16002"""
        from optimizer.subspace_sgd import DOM_SGD
        optimizer = DOM_SGD(model, model.parameters(), kwargs["num_classes"], 
                            kwargs["criterion_summed"], kwargs["batch_size"], 
                            kwargs["num_classes"], kwargs["device"], 
                            use_hf_model=kwargs["hf_model"], start=kwargs["eig_start"], end=kwargs["eig_end"], 
                            eigs_pattern= kwargs["eigs_pattern"], lr=lr, momentum=momentum, weight_decay=weight_decay)
        if kwargs["eigs_pattern"] != "LM":
            model_params = model_params | {"eig_pattern": kwargs["eigs_pattern"]}
        if kwargs["eig_start"] != 0 or kwargs["eig_end"] != -1:
            model_params = model_params | {"start": kwargs["eig_start"], "end": kwargs["eig_end"]}
    elif opt_name == "bulk_sgd":
        """from paper Does SGD really happen in tiny subspaces? https://arxiv.org/pdf/2405.16002"""
        from optimizer.subspace_sgd import BULK_SGD
        optimizer = BULK_SGD(model, model.parameters(), kwargs["num_classes"], 
                            kwargs["criterion_summed"], kwargs["batch_size"], 
                            kwargs["num_classes"], kwargs["device"], 
                            use_hf_model=kwargs["hf_model"], end=kwargs["eig_end"], lr=lr, momentum=momentum, weight_decay=weight_decay)
        if kwargs["eig_end"] != -1:
            model_params = model_params | {"end": kwargs["eig_end"]}
    elif opt_name == "gn_dom_sgd":
        from optimizer.subspace_sgd import GN_DOM_SGD
        optimizer = GN_DOM_SGD(model,  model.parameters(), kwargs["num_classes"], kwargs["criterion"], kwargs["criterion_summed"], kwargs["device"],
                               lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == "gn_bulk_sgd":
        from optimizer.subspace_sgd import GN_BULK_SGD
        optimizer = GN_BULK_SGD(model,  model.parameters(), kwargs["num_classes"], kwargs["criterion"], kwargs["criterion_summed"], kwargs["device"],
                               lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == "sketch_adam":
        from optimizer.sketch_adam import Adam
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
    elif opt_name == "alternate_sam":
        from optimizer.sam import Alternate_SAM
        if kwargs["base_opt"] == "sgd":
            base_optimizer = torch.optim.SGD
            optimizer = Alternate_SAM(model.parameters(), base_optimizer, alpha=kwargs["sam_alpha"], rho=kwargs["sam_rho"], adaptive=kwargs["sam_adaptive"], train_stats=kwargs["train_stats"], lr=lr, momentum=momentum, weight_decay=weight_decay)
        model_params = model_params | {"base_opt": kwargs["base_opt"], "sam_rho": kwargs["sam_rho"], "look_alpha": kwargs["sam_alpha"]} 
    elif opt_name == "alternate_sam_v2":
        from optimizer.sam import Alternate_SAM_v2
        if kwargs["base_opt"] == "sgd":
            base_optimizer = torch.optim.SGD
            optimizer = Alternate_SAM_v2(model.parameters(), base_optimizer, alpha=kwargs["sam_alpha"], rho=kwargs["sam_rho"], adaptive=kwargs["sam_adaptive"], train_stats=kwargs["train_stats"], lr=lr, momentum=momentum, weight_decay=weight_decay)
        model_params = model_params | {"base_opt": kwargs["base_opt"], "sam_rho": kwargs["sam_rho"], "look_alpha": kwargs["sam_alpha"]} 
    elif opt_name == "alternate_sam_v3":
        from optimizer.sam import Alternate_SAM_v3
        if kwargs["base_opt"] == "sgd":
            base_optimizer = torch.optim.SGD
            optimizer = Alternate_SAM_v3(model.parameters(), base_optimizer, alpha=kwargs["sam_alpha"], rho=kwargs["sam_rho"], adaptive=kwargs["sam_adaptive"], train_stats=kwargs["train_stats"], lr=lr, momentum=momentum, weight_decay=weight_decay)
        model_params = model_params | {"base_opt": kwargs["base_opt"], "sam_rho": kwargs["sam_rho"], "look_alpha": kwargs["sam_alpha"]} 
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
            optimizer = LookSAM(alpha=kwargs["sam_alpha"], params=model.parameters(), base_optimizer=base_optimizer, rho=kwargs["sam_rho"], lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif kwargs["base_opt"] == "adam":
            base_optimizer = torch.optim.Adam
            optimizer = LookSAM(alpha=kwargs["sam_alpha"], params=model.parameters(), base_optimizer=base_optimizer, rho=kwargs["sam_rho"], lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise NotImplementedError
        model_params = model_params | {"base_opt": kwargs["base_opt"], "sam_rho": kwargs["sam_rho"], "look_alpha": kwargs["sam_alpha"]} 
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
    elif opt_name == "adahessian":
        from optimizer.adahessian import Adahessian
        optimizer = Adahessian(model.parameters(), lr=lr, betas=(momentum, 0.999), eps=1e-4, weight_decay=weight_decay)
    elif opt_name == "federated":
        optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=kwargs['client_momentum'],
                            weight_decay=weight_decay)
        model_params = model_params | {'server_opt': kwargs['server_opt_name'], 'client_opt': kwargs['client_opt_name'], 'client_lr': kwargs['client_lr'], 'client_momentum': kwargs['client_momentum'],
                                       "client_num": kwargs['client_num'], 'client_epoch': kwargs['client_epoch'], 'sketch_size': kwargs['sketch_size']}
    elif opt_name == "fetchsgd":
        from optimizer.fetchsgd import fetchSGD
        optimizer = fetchSGD(model, model.parameters(), lr=lr, sketch_size=kwargs["sketch_size"], momentum=momentum)
    elif opt_name == "onebit":
        from torch.nn.utils import parameters_to_vector
        p = len(parameters_to_vector(model.parameters()))
        from optimizer.onebit_adam import OneBit_Adam
        optimizer = OneBit_Adam(model.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
        kwargs["server_error_feedback"] = 0
        kwargs["client_error_feedback"] = torch.zeros(kwargs["client_num"], p).to(kwargs["device"])
        #model_params = model_params | {"switch_epoch": kwargs["switch_epoch"]}
    elif opt_name == "onebit_v2":
        from torch.nn.utils import parameters_to_vector
        p = len(parameters_to_vector(model.parameters()))
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
        kwargs["server_error_feedback"] = 0
        kwargs["client_error_feedback"] = torch.zeros(kwargs["client_num"], p).to(kwargs["device"])
        #model_params = model_params | {"switch_epoch": kwargs["switch_epoch"]}
    elif opt_name == "cdadam":
        from torch.nn.utils import parameters_to_vector
        p = len(parameters_to_vector(model.parameters()))
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
        kwargs["g_tilde"] = 0 
        kwargs["g_hat_server"] = 0
        kwargs["g_hat"] = torch.zeros(kwargs["client_num"], p).to(kwargs["device"])
    elif opt_name in ["cocktailsgd", "cocktailsgd2"]:
        kwargs["local_model"] = {}
        kwargs["server_error_feedback"] = 0
        "just placeholder optimizer, not used in fact"
        optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    elif opt_name == "marina":
        assert kwargs["marina_prob"] > 0 and kwargs["marina_prob"] <= 1
        model_params = model_params | {"base_opt": kwargs["base_opt"], "full_prob": kwargs["marina_prob"]}
        if kwargs["base_opt"] == "sgd":
            optimizer = optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=kwargs['client_momentum'],
                                weight_decay=weight_decay)
        elif kwargs["base_opt"] == "adam":
            from torch.optim import Adam
            optimizer = Adam(model.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    elif opt_name in ["cams", "paq"]:
        if kwargs["base_opt"] == "sgd":
             optimizer = optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
        elif kwargs["base_opt"] == "adam":
            from torch.optim import Adam
            optimizer = Adam(model.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
        else:
            raise NotImplementedError
        model_params = model_params | {"base_opt": kwargs["base_opt"]}
    elif opt_name in ["lora_rite", "lora_rite_v2"]:
        from optimizer.lora_rite import LORA_RITE
        if opt_name == "lora_rite":
            optimizer = LORA_RITE(model, lr=lr, beta=momentum, output_layer_name=kwargs["output_layer_name"], version="v1")
        elif opt_name == "lora_rite_v2":
            optimizer = LORA_RITE(model, lr=lr, beta=momentum, output_layer_name=kwargs["output_layer_name"], version="v2")
        #model_params = model_params | {"base_opt": kwargs["base_opt"]}
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
    #if warm_start:
    #    lr_scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=5, after_scheduler=lr_scheduler)    
    return optimizer, lr_scheduler, model_params

def load_fake_scheduler(lr, **kwargs):
    from torch.nn.parameter import Parameter
    optimizer = optim.SGD([Parameter(torch.tensor(0.0))], lr=lr)
    if kwargs["scheduler_name"] == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs["epoch"], eta_min=kwargs["lr_min"])
        #model_params = model_params | {"scheduler": "cosine", "lr_min": kwargs["lr_min"]} 
    else:
        lr_scheduler = None
    return lr_scheduler



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