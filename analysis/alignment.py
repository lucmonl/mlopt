import torch
import sys
import numpy as np

def compute_weight_signal_alignment(graphs, model, signal):
    align_plus = (signal @ model.w_plus.detach().cpu()).numpy().squeeze()
    align_minus = (signal @ model.w_minus.detach().cpu()).numpy().squeeze()
    graphs.align.append(np.concatenate([align_plus, align_minus]))
    return