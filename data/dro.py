import os
import torch
import numpy as np
from torch.utils.data import Subset
#from data.label_shift_utils import prepare_label_shift_data
#from data.confounder_utils import prepare_confounder_data

from data.group_dro.confounder_utils import prepare_confounder_data

dataset_attributes = {
    'CelebA': {
        'root_dir': 'celebA'
    },
    'CUB': {
        'root_dir': '/projects/dali/'
    },
    'CIFAR10': {
        'root_dir': 'CIFAR10/data'
    },
    'MultiNLI': {
        'root_dir': 'multinli'
    }
}

def prepare_data(dataset, root_dir, target_name, confounder_names, model_name, augment_data, train, return_full_dataset=False):
    shift_type ='confounder'
    # Set root_dir to defaults if necessary
    if root_dir is None:
        root_dir = dataset_attributes[dataset]['root_dir']
    if shift_type=='confounder':
        return prepare_confounder_data(dataset, root_dir, target_name, confounder_names, model_name, augment_data, train, return_full_dataset)
    
    """
    elif args.shift_type.startswith('label_shift'):
        assert not return_full_dataset
        return prepare_label_shift_data(args, train)
    """

def load_dro(batch_size, dataset, target_name, confounder_names, model_name):
    train_data, val_data, test_data = prepare_data(dataset=dataset, root_dir=dataset_attributes[dataset]['root_dir'], 
                                                   target_name=target_name, 
                                                   confounder_names=confounder_names, 
                                                   model_name = model_name, augment_data=False, train=True)
    C = train_data.n_classes
    data_params = {"compute_acc": True}
    transform_to_one_hot = True

    loader_kwargs = {'batch_size':batch_size, 'num_workers':1, 'pin_memory':True}
    train_loader = train_data.get_loader(train=True, reweight_groups=False, **loader_kwargs)
    test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)

    analysis_loader = None
    analysis_test_loader = None

    n_groups = train_data.n_groups
    group_counts = train_data.group_counts().cuda()
    group_str = train_data.group_str

    return train_loader, test_loader, analysis_loader, analysis_test_loader, n_groups, group_counts, group_str, C, transform_to_one_hot, data_params



