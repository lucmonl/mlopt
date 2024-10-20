import torch
import sys
import numpy as np

def compute_residual(graphs, model, signals, signal_index, train_loader, device):
    assert len(signals) == 1
    batch_size = train_loader.batch_size
    residuals = 0
    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        #print(torch.sum(signals[0]))
        #print((model(data.to(device)) - target.to(device)).shape)
        #print(torch.sign(data[range(batch_size),signal_index[batch_size*(batch_idx-1):batch_size*batch_idx], :].to(device) @ model.w_plus).shape)
        #print(model(data.to(device)) - target.to(device))
        # ensure that the residual and sign vectors have the same dimension!
        print(model(data.to(device)))
        residuals += torch.sum((model(data.to(device)) - target.to(device)) * torch.sign(data[range(batch_size),signal_index[batch_size*(batch_idx-1):batch_size*batch_idx], :].to(device) @ model.w_plus.squeeze()))
    graphs.residuals.append(residuals.item())
    return

def compute_weight_signal_alignment(graphs, model, signals, signal_index, train_loader):
    for idx, signal in enumerate(signals, start=1):
        align_plus = (signal @ model.w_plus.detach().cpu()).numpy() # d @ (d * m) = m
        #align_minus = (signal @ model.w_minus.detach().cpu()).numpy()

        stored_align_signal = getattr(graphs, "align_signal_{}".format(idx)) 
        #stored_align_signal.append(np.concatenate([align_plus, align_minus])) #comment
        stored_align_signal.append(align_plus)
        setattr(graphs, "align_signal_{}".format(idx), stored_align_signal) 
        #graphs.align_signal.append(np.concatenate([align_plus, align_minus])) # 2*m
        setattr(graphs, "signal_{}".format(idx), signal)

    batch_size = train_loader.batch_size
    align_noise_list = []
    for batch_idx, (data, _) in enumerate(train_loader, start=1):
        align_noise_plus = torch.sum((data @ model.w_plus.detach().cpu()), dim=[1,2]).numpy() # (B*P*d) @ (d*m) --sum--> (B)
        align_noise_plus -= torch.sum(data[range(batch_size),signal_index[batch_size*(batch_idx-1):batch_size*batch_idx], :] @ model.w_plus.detach().cpu(), dim=-1).numpy()
        #align_noise_minus = torch.sum((data @ model.w_minus.detach().cpu()), dim=[1,2]).numpy() # (B*P*d) @ (d*m)
        #align_noise_minus -= torch.sum(data[range(batch_size),signal_index[batch_size*(batch_idx-1):batch_size*batch_idx], :] @ model.w_minus.detach().cpu(), dim=-1).numpy()
        #align_noise_list.append(np.concatenate([align_noise_plus, align_noise_minus])) #2*B
        align_noise_list.append(align_noise_plus) #2*B    
    graphs.align_noise.append(np.concatenate(align_noise_list, axis=-1).reshape(-1))
    return

def get_linear_coefs(graphs, model):
    graphs.linear_coefs.append([model.v_plus.item(), model.v_minus.item()])
    return