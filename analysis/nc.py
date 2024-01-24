import numpy as np
from scipy.sparse.linalg import svds
import torch
from tqdm import tqdm
import torch.nn.functional as F

def get_nc_statistics(graphs, model, features, classifier, loss_name, criterion_summed, weight_decay, num_classes, loader, loader_test, device, debug=False):
    graphs_lists = [graphs.NCC_mismatch, graphs.reg_loss, graphs.norm_M_CoV, graphs.norm_W_CoV, graphs.MSE_wd_features, \
                    graphs.LNC1, graphs.LNC23, graphs.Lperp, graphs.Sw_invSb, graphs.W_M_dist, graphs.cos_M, graphs.cos_W]
    get_nc_statistics_mode(graphs_lists, model, features, classifier, loss_name, criterion_summed, weight_decay, device, num_classes, loader, debug=debug)

    graphs_lists = [graphs.test_NCC_mismatch, graphs.test_reg_loss, graphs.test_norm_M_CoV, graphs.test_norm_W_CoV, graphs.test_MSE_wd_features, \
                    graphs.test_LNC1, graphs.test_LNC23, graphs.test_Lperp, graphs.test_Sw_invSb, graphs.test_W_M_dist, graphs.test_cos_M, graphs.test_cos_W]
    get_nc_statistics_mode(graphs_lists, model, features, classifier, loss_name, criterion_summed, weight_decay, device, num_classes, loader_test, debug=debug)

def get_nc_statistics_mode(graphs_lists, model, features, classifier, loss_name, criterion_summed, weight_decay, device, num_classes, loader, debug=False):
    graphs_NCC_mismatch, graphs_reg_loss, graphs_norm_M_CoV, graphs_norm_W_CoV, graphs_MSE_wd_features, \
                    graphs_LNC1, graphs_LNC23, graphs_Lperp, graphs_Sw_invSb, graphs_W_M_dist, graphs_cos_M, graphs_cos_W = graphs_lists


    model.eval()

    C             = num_classes
    N             = [0 for _ in range(C)]
    mean          = [0 for _ in range(C)]
    Sw            = 0

    loss          = 0
    net_correct   = 0
    NCC_match_net = 0
    #eigs, _ = get_hessian_eigenvalues_weight_decay(model, criterion_summed, weight_decay, loader_abridged, neigs=10, num_classes=num_classes, device=device)
    #graphs.eigs.append(eigs[0].item())
    #print("in analysis 2:", eigs)


    for computation in ['Mean','Cov']:
        pbar = tqdm(total=len(loader), position=0, leave=True)
        for batch_idx, (data, target) in enumerate(loader, start=1):

            data, target = data.to(device), target.to(device)

            output = model(data)
            h = features.value.data.view(data.shape[0],-1) # B CHW
            
            # during calculation of class means, calculate loss
            if computation == 'Mean':
                if str(criterion_summed) == 'CrossEntropyLoss()':
                    loss += criterion_summed(output, target).item()
                elif str(criterion_summed) == 'MSELoss()':
                    loss += criterion_summed(output, F.one_hot(target, num_classes=num_classes).float()).item()

            for c in range(C):
                # features belonging to class c
                idxs = (target == c).nonzero(as_tuple=True)[0]
                
                if len(idxs) == 0: # If no class-c in this batch
                    continue

                h_c = h[idxs,:] # B CHW

                if computation == 'Mean':
                    # update class means
                    mean[c] += torch.sum(h_c, dim=0) # CHW
                    N[c] += h_c.shape[0]
                    
                elif computation == 'Cov':
                    # update within-class cov

                    z = h_c - mean[c].unsqueeze(0) # B CHW
                    cov = torch.matmul(z.unsqueeze(-1), # B CHW 1
                                       z.unsqueeze(1))  # B 1 CHW
                    Sw += torch.sum(cov, dim=0)

                    # during calculation of within-class covariance, calculate:
                    # 1) network's accuracy
                    net_pred = torch.argmax(output[idxs,:], dim=1)
                    net_correct += sum(net_pred==target[idxs]).item()

                    # 2) agreement between prediction and nearest class center
                    NCC_scores = torch.stack([torch.norm(h_c[i,:] - M.T,dim=1) \
                                              for i in range(h_c.shape[0])])
                    NCC_pred = torch.argmin(NCC_scores, dim=1)
                    NCC_match_net += sum(NCC_pred==net_pred).item()

            pbar.update(1)
            pbar.set_description(
                'Analysis {}\t'
                '[{}/{} ({:.0f}%)]'.format(
                    computation,
                    batch_idx,
                    len(loader),
                    100. * batch_idx/ len(loader)))
            
            if debug and batch_idx > 20:
                break
        pbar.close()
        
        if computation == 'Mean':
            for c in range(C):
                mean[c] /= N[c]
                M = torch.stack(mean).T
            loss /= sum(N)
        elif computation == 'Cov':
            Sw /= sum(N)
    
    #graphs.loss.append(loss)
    #graphs_accuracy.append(net_correct/sum(N))
    graphs_NCC_mismatch.append(1-NCC_match_net/sum(N))

    # loss with weight decay
    reg_loss = loss
    for param in model.parameters():
        reg_loss += 0.5 * weight_decay * torch.sum(param**2).item()
    graphs_reg_loss.append(reg_loss)

    # global mean
    muG = torch.mean(M, dim=1, keepdim=True) # CHW 1
    
    # between-class covariance
    M_ = M - muG
    Sb = torch.matmul(M_, M_.T) / C

    # avg norm
    W  = classifier.weight
    M_norms = torch.norm(M_,  dim=0)
    W_norms = torch.norm(W.T, dim=0)

    graphs_norm_M_CoV.append((torch.std(M_norms)/torch.mean(M_norms)).item())
    graphs_norm_W_CoV.append((torch.std(W_norms)/torch.mean(W_norms)).item())

    # Decomposition of MSE #
    if loss_name == 'MSELoss':

        wd = 0.5 * weight_decay # "\lambda" in manuscript, so this is halved
        St = Sw+Sb
        size_last_layer = Sb.shape[0]
        eye_P = torch.eye(size_last_layer).to(device)
        eye_C = torch.eye(C).to(device)

        St_inv = torch.inverse(St + (wd/(wd+1))*(muG @ muG.T) + wd*eye_P)

        w_LS = 1 / C * (M.T - 1 / (1 + wd) * muG.T) @ St_inv
        b_LS = (1/C * torch.ones(C).to(device) - w_LS @ muG.T.squeeze(0)) / (1+wd)
        w_LS_ = torch.cat([w_LS, b_LS.unsqueeze(-1)], dim=1)  # c x n
        b  = classifier.bias
        w_ = torch.cat([W, b.unsqueeze(-1)], dim=1)  # c x n

        LNC1 = 0.5 * (torch.trace(w_LS @ (Sw + wd*eye_P) @ w_LS.T) + wd*torch.norm(b_LS)**2)
        LNC23 = 0.5/C * torch.norm(w_LS @ M + b_LS.unsqueeze(1) - eye_C) ** 2

        A1 = torch.cat([St + muG @ muG.T + wd*eye_P, muG], dim=1)
        A2 = torch.cat([muG.T, torch.ones([1,1]).to(device) + wd], dim=1)
        A = torch.cat([A1, A2], dim=0)
        Lperp = 0.5 * torch.trace((w_ - w_LS_) @ A @ (w_ - w_LS_).T)

        MSE_wd_features = loss + 0.5* weight_decay * (torch.norm(W)**2 + torch.norm(b)**2).item()
        MSE_wd_features *= 0.5

        graphs_MSE_wd_features.append(MSE_wd_features)
        graphs_LNC1.append(LNC1.item())
        graphs_LNC23.append(LNC23.item())
        graphs_Lperp.append(Lperp.item())

    # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=C-1)
    inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T 
    graphs_Sw_invSb.append(np.trace(Sw @ inv_Sb))

    # ||W^T - M_||
    normalized_M = M_ / torch.norm(M_,'fro')
    normalized_W = W.T / torch.norm(W.T,'fro')
    graphs_W_M_dist.append((torch.norm(normalized_W - normalized_M)**2).item())

    # mutual coherence
    def coherence(V): 
        G = V.T @ V
        G += torch.ones((C,C),device=device) / (C-1)
        G -= torch.diag(torch.diag(G))
        return torch.norm(G,1).item() / (C*(C-1))

    graphs_cos_M.append(coherence(M_/M_norms))
    graphs_cos_W.append(coherence(W.T/W_norms))