import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from data.group_dro.models import model_attributes
from torch.utils.data import Dataset, Subset
#from data.celebA_dataset import CelebADataset
from data.group_dro.cub_dataset import CUBDataset
from data.group_dro.dro_dataset import DRODataset
#from data.multinli_dataset import MultiNLIDataset

################
### SETTINGS ###
################
"""
confounder_settings = {
    
    'CelebA':{
        'constructor': CelebADataset
    },
    'CUB':{
        'constructor': CUBDataset
    },
    'MultiNLI':{
        'constructor': MultiNLIDataset
    }
}
"""
confounder_settings = {
    'CUB':{
        'constructor': CUBDataset
    },
}
########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(dataset, root_dir, target_name, confounder_names, model_name, augment_data, train, return_full_dataset=False):
    full_dataset = confounder_settings[dataset]['constructor'](
        root_dir=root_dir,
        target_name=target_name,
        confounder_names=confounder_names,
        model_type=model_name,
        augment_data=augment_data)
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str)
    if train:
        splits = ['train', 'val', 'test']
    else:
        splits = ['test']
    subsets = full_dataset.get_splits(splits, train_frac=1.0)
    print("making DRODataset...")
    dro_subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
                              n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str) \
                   for split in splits]
    return dro_subsets