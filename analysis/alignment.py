import torch
import sys
import numpy as np

def compute_weight_signal_alignment(graphs, model, signal, signal_index, train_loader):
    align_plus = (signal @ model.w_plus.detach().cpu()).numpy() # d @ (d * m) = m
    align_minus = (signal @ model.w_minus.detach().cpu()).numpy()
    graphs.align_signal.append(np.concatenate([align_plus, align_minus])) # 2*m
    
    batch_size = train_loader.batch_size
    align_noise_list = []
    for batch_idx, (data, _) in enumerate(train_loader, start=1):      
        align_noise_plus = torch.sum((data @ model.w_plus.detach().cpu()), dim=[1,2]).numpy() # (B*P*d) @ (d*m) --sum--> (B)
        align_noise_plus -= torch.sum(data[range(batch_size),signal_index[batch_size*(batch_idx-1):batch_size*batch_idx], :] @ model.w_plus.detach().cpu(), dim=-1).numpy()
        align_noise_minus = torch.sum((data @ model.w_minus.detach().cpu()), dim=[1,2]).numpy() # (B*P*d) @ (d*m)
        align_noise_minus -= torch.sum(data[range(batch_size),signal_index[batch_size*(batch_idx-1):batch_size*batch_idx], :] @ model.w_minus.detach().cpu(), dim=-1).numpy()
        align_noise_list.append(np.concatenate([align_noise_plus, align_noise_minus])) #2*B
    
    graphs.align_noise.append(np.concatenate(align_noise_list, axis=-1).reshape(-1))

    graphs.out_layer.append([model.v_plus.item(), model.v_minus.item()])
    return