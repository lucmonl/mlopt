import os
import sys

import torch
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import gc
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models

from tqdm import tqdm
from collections import OrderedDict
#os.environ["SCIPY_USE_PROPACK"] =  "1"
from scipy.sparse.linalg import svds

from IPython import embed

from optimizer.sam import disable_running_stats, enable_running_stats

torch.manual_seed(32)
 
# setting path
#current_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir = os.path.dirname(current_dir)
#sys.path.append(parent_dir)
#print(parent_dir)
#from utilities import get_hessian_eigenvalues_weight_decay, get_hessian_eigenvalues     

def train(model, loss_name, criterion, device, num_classes, train_loader, optimizer, epoch):
    model.train()
    
    pbar = tqdm(total=len(train_loader), position=0, leave=True)
    accuracy = 0
    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        if data.shape[0] != batch_size:
                continue
        
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        out = model(data)
        if loss_name == 'CrossEntropyLoss':
            loss = criterion(out, target)
        elif loss_name == 'MSELoss':
            #if transform_to_one_hot:
            #    loss = criterion(out, F.one_hot(target, num_classes=num_classes).float()) * num_classes
            #else:
            loss = criterion(out, target)
        
        loss.backward()
        if opt_name == "sam":
            optimizer.first_step(zero_grad=True)
            # second forward-backward step
            disable_running_stats(model)
            out = model(data)
            #loss = criterion(out, F.one_hot(target, num_classes=num_classes).float()) * num_classes
            loss = criterion(out, target).float()
            loss.backward()
            optimizer.second_step(zero_grad=True)
            enable_running_stats(model)
        elif opt_name == "goldstein":
            gold_iters = 0
            while gold_iters < 1:
                gold_iters += 1
                optimizer.first_step(zero_grad=True)
                disable_running_stats(model)
                out = model(data)
                #loss = criterion(out, F.one_hot(target, num_classes=num_classes).float()) * num_classes
                loss = criterion(out, target).float()
                loss.backward()
                optimizer.second_step(zero_grad=False)
            optimizer.third_step(zero_grad=True)
            enable_running_stats(model)
        else:
            optimizer.step()

        if out.dim() > 1:
            accuracy = torch.mean((torch.argmax(out,dim=1)==target).float()).item()

        pbar.update(1)
        pbar.set_description(
            'Train\t\tEpoch: {} [{}/{} ({:.0f}%)] \t'
            'Batch Loss: {:.6f} \t'
            'Batch Accuracy: {:.6f}'.format(
                epoch,
                batch_idx,
                len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item(),
                accuracy))
        
        if debug and batch_idx > 20:
            break

    pbar.close()

def analysis(graphs, analysis_list, model, model_name, criterion_summed, device, num_classes, train_loader, test_loader, analysis_loader, analysis_test_loader):
    if 'loss' in analysis_list:
        from analysis.loss import compute_loss
        compute_loss(graphs, model, loss_name, criterion, criterion_summed, device, num_classes, train_loader, test_loader)

    if 'eigs' in analysis_list:
        from analysis.eigs import compute_eigenvalues
        compute_eigenvalues(graphs, model, criterion_summed, weight_decay, analysis_loader, analysis_test_loader, num_classes, device)
    
    if 'nc' in analysis_list:
        from analysis.nc import get_nc_statistics
        get_nc_statistics(graphs, model, features, classifier, loss_name, criterion_summed, weight_decay, num_classes, analysis_loader, analysis_test_loader, device, debug=False)

    if 'weight_norm' in analysis_list:
        from analysis.weight_norm import get_min_weight_norm, get_min_weight_norm_with_g, get_grad_loss_ratio
        get_min_weight_norm(graphs, model, C=num_classes, model_name=model_name)
        get_min_weight_norm_with_g(graphs, model, C=num_classes, model_name=model_name)
        get_grad_loss_ratio(graphs, model, loss_name, analysis_loader, criterion, criterion_summed, num_classes, device)


class features:
    pass

def hook(self, input, output):
    features.value = input[0].clone()

class graphs:
    def __init__(self):
        self.log_epochs   = []
        self.accuracy     = []
        self.loss         = []
        self.eigs         = []
        self.eigs_test    = []

        self.test_loss    = []
        self.test_accuracy = []

        self.reg_loss     = []
        self.test_reg_loss     = []

        # NC1
        self.Sw_invSb     = []

        # NC2
        self.norm_M_CoV   = []
        self.norm_W_CoV   = []
        self.cos_M        = []
        self.cos_W        = []

        # NC3
        self.W_M_dist     = []

        # NC4
        self.NCC_mismatch = []

        # Decomposition
        self.MSE_wd_features = []
        self.LNC1 = []
        self.LNC23 = []
        self.Lperp = []

        # NC1
        self.test_Sw_invSb     = []

        # NC2
        self.test_norm_M_CoV   = []
        self.test_norm_W_CoV   = []
        self.test_cos_M        = []
        self.test_cos_W        = []

        # NC3
        self.test_W_M_dist     = []

        # NC4
        self.test_NCC_mismatch = []

        # Decomposition
        self.test_MSE_wd_features = []
        self.test_LNC1 =         []
        self.test_LNC23 =        []
        self.test_Lperp =        []

        # weight norm statsitics
        self.wn_grad_loss_ratio = []
        self.wn_norm_min        = []
        self.wn_norm_min_with_g = []


def get_lookup_directory(lr, dataset_name, opt_name, model_name, weight_decay, batch_size, **kwargs):
    results_dir = "results"
    directory = f"{results_dir}/{dataset_name}/{opt_name}/{model_name}/"
    for key, value in kwargs.items():
        directory += f"{key}_{value}/"
    directory += f"lr_{lr}/wd_{weight_decay}/batch_size_{batch_size}/"
    return directory

def get_directory(lr, dataset_name, opt_name, model_name, weight_decay, batch_size, epochs, **kwargs):
    #results_dir = "results"
    #directory = f"{results_dir}/{model_name}/{dataset_name}/{opt_name}/lr_{lr}/wd_{weight_decay}/batch_size_{batch_size}/epoch_{epochs}/"
    directory = get_lookup_directory(lr, dataset_name, opt_name, model_name, weight_decay, batch_size, **kwargs) + f"epoch_{epochs}/"
    return directory

def continue_training(lr, dataset_name, opt_name, model_name, weight_decay, batch_size, epochs, **kwargs):
    #results_dir = "results"
    #lookup_dir = f"{results_dir}/{model_name}/{dataset_name}/{opt_name}/lr_{lr}/wd_{weight_decay}/batch_size_{batch_size}/"
    lookup_dir = get_lookup_directory(lr, dataset_name, opt_name, model_name, weight_decay, batch_size, **kwargs)
    if not os.path.exists(lookup_dir):
        return 0
    epoch_dir = os.listdir(lookup_dir)
    trained_epochs = [int(x.split("_")[-1]) for x in epoch_dir]
    trained_epochs.sort(reverse=True)
    load_from_epoch = 0
    for trained_epoch in trained_epochs:
        if trained_epoch == epochs:
            print(lookup_dir)
            yes_or_no = input("The path already exists. Are you sure to overwrite? [y/n]")
            if yes_or_no == 'n':
                sys.exit()
        if trained_epoch < epochs:
            load_from_epoch = trained_epoch
            break
    return load_from_epoch
    
if __name__ == "__main__":
    DATASETS = ["spurious", "cifar", "mnist"]
    MODELS = ["2-mlp-sim-bn", "weight_norm_torch", "weight_norm", "resnet18", "WideResNet"]
    INIT_MODES = ["O(1)", "O(1/sqrt{m})"]
    LOSSES = ['MSELoss', 'CrossEntropyLoss']
    OPTIMIZERS = ['goldstein','sam', 'sgd', 'norm-sgd']

    parser = argparse.ArgumentParser(description="Train Configuration.")
    parser.add_argument("--debug", type=bool, default=False, help="only run first 20 batches per epoch if set to True")
    parser.add_argument("--dataset",  type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("--model",  type=str, choices=MODELS, help="which model to train")
    parser.add_argument("--width", type=int, default=512, help="network width for weight norm")
    parser.add_argument("--init_mode",  type=str, default="O(1)", choices=INIT_MODES, help="Initialization Mode")
    parser.add_argument("--basis_var", type=float, default=5, help="variance for initialization")
    parser.add_argument("--wn_scale", type=float, default=10, help="scaling coef for weight_norm model")

    parser.add_argument("--loss",  type=str, choices=LOSSES, help="Training Loss")
    parser.add_argument("--opt",  type=str, choices=OPTIMIZERS, help="Training Optimizer")
    parser.add_argument('--analysis', nargs='+', type=str, help="quantities that will be analyzed")
    parser.add_argument("--lr", type=float, help="the learning rate")
    parser.add_argument("--lr_decay", type=float, default=1, help="the learning rate decat. default: no decay")
    parser.add_argument("--momentum", type=float, default=0, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=0, help ="weight decay")
    parser.add_argument("--epoch", type=int, help="total training epoches")
    parser.add_argument("--batch_size", type=int, help="batch size in training, also the number of samples in analysis dataset")
    
    # data
    parser.add_argument("--sp_train_size", type=int, default=4096, help="training size for spurious dataset")
    parser.add_argument("--sp_feat_dim", type=int, default=20, help="dimension for spurious data")

    # optimizer hyperparameters
    parser.add_argument("--sam_rho", type=float, default=0.2, help="rho for SAM")
    parser.add_argument("--sam_adaptive", type=bool, default=False, help="use adaptive SAM")
    parser.add_argument("--gold_delta", type=float, default=1, help="delta for goldstein")
    args = parser.parse_args()


    debug = args.debug # Only runs 20 batches per epoch for debugging
    model_params = {}

    # dataset parameters
    dataset_name        = args.dataset #"spurious" #"cifar"\
    sp_train_size       = args.sp_train_size
    sp_feat_dim         = args.sp_feat_dim

    # model parameters
    model_name          = args.model #"2-mlp-sim-bn"#"weight_norm_torch" #"weight_norm" #"resnet18"
    wn_width            = args.width #2048#512 #, 1024
    wn_init_mode        = args.init_mode  # "O(1)" # "O(1/sqrt{m})"
    wn_basis_var        = args.basis_var
    wn_scale            = args.wn_scale

    # Optimization Criterion
    # loss_name = 'CrossEntropyLoss'
    loss_name           = args.loss
    opt_name            = args.opt
    analysis_list       = args.analysis # ['loss', 'eigs'] #['loss','eigs','nc',''weight_norm']

    # Optimization hyperparameters
    lr_decay            = args.lr_decay #1# 0.1

    epochs              = args.epoch
    epochs_lr_decay     = [epochs//3, epochs*2//3]

    batch_size          = args.batch_size #512 # 128

    #hyperparameters for sam
    sam_rho = args.sam_rho #0.1
    sam_adaptive = args.sam_adaptive #False

    if debug:
        torch.autograd.set_detect_anomaly(True)

    #hyperparameters for gold
    if opt_name == "goldstein":
        gold_delta = args.gold_delta
        model_params = model_params | {"gold_delta": gold_delta}

    # Best lr after hyperparameter tuning
    if loss_name == 'CrossEntropyLoss':
        lr = args.lr #0.0679
    elif loss_name == 'MSELoss':
        lr = args.lr #0.0184
    momentum            = args.momentum #0 # 0.9
    weight_decay        = args.weight_decay #0 # 5e-4 * 10


    if dataset_name == "cifar":
        from data.cifar import load_cifar
        train_loader, test_loader, analysis_loader, analysis_test_loader, input_ch, num_pixels, C, transform_to_one_hot = load_cifar(loss_name, batch_size)
    elif dataset_name == "mnist":
        from data.mnist import load_mnist
        train_loader, test_loader, analysis_loader, input_ch, C, transform_to_one_hot = load_mnist(loss_name, batch_size)
    elif dataset_name == "spurious":
        from data.spurious import load_spurious_data
        train_loader, test_loader, analysis_loader, analysis_test_loader, num_pixels, C, transform_to_one_hot = load_spurious_data(loss_name, sp_feat_dim, sp_train_size, batch_size)
        model_params = model_params | {"feat_dim": sp_feat_dim, "train_size": sp_train_size}

    if model_name == "resnet18":
        model = models.resnet18(pretrained=False, num_classes=C)
        model_params = {} | model_params
    elif model_name == "WideResNet":
        from arch.wide_resnet import WideResNet
        model = WideResNet(depth=16, width_factor=8, dropout=0.0, in_channels=input_ch, labels=C)
    elif model_name == "weight_norm":
        from arch.weight_norm import weight_norm_net
        model = weight_norm_net(num_pixels, [wn_width, wn_width], wn_init_mode, wn_basis_var, wn_scale)
        model_params = {"width": wn_width, "init": wn_init_mode, "var": wn_basis_var, "scale": wn_scale} | model_params
    elif model_name == "weight_norm_torch":
        from arch.weight_norm import weight_norm_torch
        model = weight_norm_torch(num_pixels, [wn_width, wn_width])
        model_params = {"width": wn_width} | model_params
    elif model_name == "2-mlp-sim-bn":
        from arch.mlp_sim_bn import mlp_sim_bn
        model = mlp_sim_bn(num_pixels, C)
        model_params = {} | model_params
    else:
        raise NotImplementedError

    


    # analysis parameters
    """
    epoch_list          = [1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11,
                        12,  13,  14,  16,  17,  19,  20,  22,  24,  27,   29,
                        32,  35,  38,  42,  45,  50,  54,  59,  65,  71,   77,
                        85,  92,  101, 110, 121, 132, 144, 158, 172, 188,  206,
                        225, 245, 268, 293, 320, 350]
    """
    #epoch_list = np.arange(1, epochs+1, 100).tolist()
    train_graphs = graphs()

    if dataset_name == "mnist":
        model.conv1 = nn.Conv2d(input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias=False) # Small dataset filter size used by He et al. (2015)
        model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)



    # register hook that saves last-layer input into features
    if "nc" in analysis_list:
        classifier = model.fc
        classifier.register_forward_hook(hook)

    if loss_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
        criterion_summed = nn.CrossEntropyLoss(reduction='sum')
    elif loss_name == 'MSELoss':
        if transform_to_one_hot:
            def mse_sum_with_one_hot(out, target):
                return nn.MSELoss(reduction='sum')(out, F.one_hot(target, num_classes=C).float()) * C

            def mse_with_one_hot(out, target):
                return nn.MSELoss()(out, F.one_hot(target, num_classes=C).float()) * C

            criterion = mse_with_one_hot
            criterion_summed = mse_sum_with_one_hot
        else:
            criterion = nn.MSELoss()
            criterion_summed = nn.MSELoss(reduction='sum')


    if opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    elif opt_name == "sam":
        base_optimizer = torch.optim.SGD
        from optimizer.sam import SAM
        optimizer = SAM(model.parameters(), base_optimizer, rho=sam_rho, adaptive=sam_adaptive, lr=lr, momentum=momentum, weight_decay=weight_decay)
        model_params = model_params | {"sam_rho": sam_rho} 
    elif opt_name == "norm-sgd":
        base_optimizer = torch.optim.SGD
        from optimizer.normalized_sgd import Normalized_Optimizer
        optimizer = Normalized_Optimizer(model.parameters(), base_optimizer,
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    elif opt_name == "goldstein":
        base_optimizer = torch.optim.SGD
        from optimizer.goldstein import Goldstein
        optimizer = Goldstein(model.parameters(), base_optimizer, delta=gold_delta, lr=lr, momentum=momentum, weight_decay=weight_decay)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=epochs_lr_decay,
                                                gamma=lr_decay)

    load_from_epoch = continue_training(lr, dataset_name, opt_name, model_name, weight_decay, batch_size, epochs, **model_params)
    epoch_list = np.arange(load_from_epoch+1, epochs+1, 200).tolist()
    if load_from_epoch != 0:
        print("loading from trained epoch {}".format(load_from_epoch))
        load_from_dir = get_directory(lr, dataset_name, opt_name, model_name, weight_decay, batch_size, load_from_epoch, **model_params)
        model.load_state_dict(torch.load(os.path.join(load_from_dir, "model.ckpt")))
        with open(f'{load_from_dir}/train_graphs.pk', 'rb') as f:
            train_graphs = pickle.load(f)
    model = model.to(device)

    directory = get_directory(lr, dataset_name, opt_name, model_name, weight_decay, batch_size, epochs, **model_params)
    os.makedirs(directory, exist_ok=True)

    import pickle

    cur_epochs = []
    for epoch in range(load_from_epoch+1, epochs + 1):
        train(model, loss_name, criterion, device, C, train_loader, optimizer, epoch)
        #lr_scheduler.step()

        if epoch in epoch_list:
            train_graphs.log_epochs.append(epoch)
            #analysis(train_graphs, model, criterion_summed, device, C, analysis_loader, test_loader)
            analysis(train_graphs, analysis_list, model, model_name, criterion_summed, device, C, train_loader, test_loader, analysis_loader, analysis_test_loader)
            pickle.dump(train_graphs, open(f"{directory}/train_graphs.pk", "wb"))
            torch.save(model.state_dict(), f"{directory}/model.ckpt")