import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from optimizer.load_optimizer import load_optimizer
from main import train
from utilities import tensor_topk

class QSGDCompressor(object):
    "adapted from https://github.com/xinyandai/gradient-quantization/blob/master/compressors/qsgd_compressor.py"
    def __init__(self, size, shape, random, n_bit, c_dim, no_cuda):
        self.random = random
        self.bit = n_bit
        c_dim = c_dim
        assert self.bit > 0

        self.cuda = not no_cuda
        self.s = 2 ** self.bit
        self.size = size
        self.shape = shape

        if c_dim == 0 or self.size < c_dim:
            self.dim = self.size
        else:
            self.dim = c_dim
            for i in range(0, 10):
                if size % self.dim != 0:
                    self.dim = self.dim // 2 * 3

        if c_dim != self.dim:
            print("alternate dimension form"
                  " {} to {}, size {} shape {}"
                  .format(c_dim, self.dim, size, shape))

        assert self.dim != 0, \
            "0 sub dimension size {}  " \
            "c_dim {} self.dim {}"\
                .format(size, c_dim, self.dim)
        assert size % self.dim == 0, \
            "not divisible size {} " \
            " c_dim {} self.dim {}"\
                .format(size, c_dim, self.dim)

        self.M = size // self.dim
        self.code_dtype = torch.int32


    def compress(self, vec):
        """
        :param vec: torch tensor
        :return: norm, signs, quantized_intervals
        """
        vec = vec.view(-1, self.dim)
        # norm = torch.norm(vec, dim=1, keepdim=True)
        norm = torch.max(torch.abs(vec), dim=1, keepdim=True)[0]
        normalized_vec = vec / norm

        scaled_vec = torch.abs(normalized_vec) * self.s
        l = torch.clamp(scaled_vec, 0, self.s-1).type(self.code_dtype)

        if self.random:
            # l[i] <- l[i] + 1 with probability |v_i| / ||v|| * s - l
            probabilities = scaled_vec - l.type(torch.float32)
            r = torch.rand(l.size())
            if self.cuda:
                r = r.cuda()
            l[:] += (probabilities > r).type(self.code_dtype)

        signs = torch.sign(vec) > 0
        return [norm, signs.view(self.shape), l.view(self.shape)]

    def decompress(self, signature):
        [norm, signs, l] = signature
        assert l.shape == signs.shape
        scaled_vec = l.type(torch.float32) * (2 * signs.type(torch.float32) - 1)
        compressed = (scaled_vec.view((-1, self.dim))) * norm / self.s
        return compressed.view(self.shape)

def cocktail_compress(v):
    v = v.clone()
    p = v.numel()
    comp = QSGDCompressor(size=p, shape=p, random=False, n_bit=8, c_dim=p, no_cuda=True)
    #randomly select
    random_sample_prob = 0.1
    v = v * torch.bernoulli(random_sample_prob * torch.ones(p)).to(v)
    #top-k
    topk_size = int(0.2 * p)
    v = tensor_topk(v, k=topk_size)
    # quantization
    v = comp.decompress(comp.compress(v))
    return v


def cocktail_compress_2(v):
    v = v.clone()
    p = v.numel()
    comp = QSGDCompressor(size=p, shape=p, random=False, n_bit=8, c_dim=p, no_cuda=True)
    #randomly select
    random_sample_prob = 0.1
    v = v * torch.bernoulli(random_sample_prob * torch.ones(p)).to(v)
    #top-k
    topk_size = int(0.2 * p)
    v = tensor_topk(v, k=topk_size)
    ind = torch.where(v != 0)
    v_nz = v[ind]
    # quantization
    v_nz_comp = comp.decompress(comp.compress(v_nz))
    v[ind] = v_nz_comp
    return v


def federated_cocktail_train(model, loss_name, criterion, device, train_loaders, server_optimizer, server_lr_scheduler, client_lr, epochs_lr_decay, lr_decay, model_params, opt_params, server_epoch):
    if opt_params["server_opt_name"] == "cocktailsgd2":
        cocktail_compress = cocktail_compress_2
    client_num, client_opt_name, client_epoch = opt_params["client_num"], opt_params["client_opt_name"], opt_params["client_epoch"]
    vector_m = 0
    model_diff_comp, client_model_temp = {}, {}
    import copy

    #from utilities import state_dict_to_vector, vector_to_state_dict
    # initialize client models, optimizers
    server_params = parameters_to_vector(model.parameters())
    device = server_params.get_device()

    client_opt_params = copy.deepcopy(opt_params)
    client_opt_params["train_stats"] = False
    for client_id in range(client_num):
        # update client models
        if client_id not in opt_params["local_model"]:
            opt_params["local_model"][client_id] = copy.deepcopy(model)
        client_model = opt_params["local_model"][client_id]
        old_params = parameters_to_vector(client_model.parameters())
        client_model.train()
        optimizer, lr_scheduler, _= load_optimizer(client_opt_name, client_model, client_lr, opt_params["client_momentum"], opt_params["client_weight_decay"], lr_decay, epochs_lr_decay, False, model_params, opt_params)
        #vector_to_parameters(old_params, client_model.parameters())
        for epoch in range(client_epoch):
            train(client_model, loss_name, criterion, device, train_loaders[client_id], optimizer, lr_scheduler, server_epoch, client_opt_params)
            
        new_params = parameters_to_vector(client_model.parameters())

        model_diff_comp[client_id] = cocktail_compress(old_params - server_params)
        client_model_temp[client_id] = new_params - model_diff_comp[client_id]
        vector_m += model_diff_comp[client_id].detach() #* min(1, opt_params["clip_tau"] / param_norm.item())
        

    vector_m = vector_m / client_num
    vector_m = vector_m + opt_params["server_error_feedback"]
    vector_m_compress = cocktail_compress(vector_m)
    opt_params["server_error_feedback"] = vector_m - vector_m_compress

    server_params += vector_m_compress
    vector_to_parameters(server_params, model.parameters())
    for client_id in range(client_num):
        vector_to_parameters(client_model_temp[client_id] + vector_m_compress, opt_params["local_model"][client_id].parameters())