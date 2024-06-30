from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
import numpy as np
import torch

def load_wilds(batch_size, task_name):
    C = 2
    transform_to_one_hot = True
    data_params = {"compute_acc": True}

    dataset = get_dataset(dataset=task_name, download=True, root_dir='/projects/dali/data')
    train_data = dataset.get_subset(
        "train",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    train_loader = get_train_loader("standard", train_data, batch_size=batch_size)

    test_data = dataset.get_subset(
        'test',
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    test_loader = get_train_loader("standard", test_data, batch_size=batch_size)

    analysis_loader = None
    analysis_test_loader = None

    return train_loader, test_loader, analysis_loader, analysis_test_loader, C, transform_to_one_hot, data_params


def load_wilds_federated(batch_size, task_name, client_num=1):
    C = 2
    transform_to_one_hot = True
    data_params = {"compute_acc": True}

    dataset = get_dataset(dataset=task_name, download=True, root_dir='/projects/dali/data')
    train_data = dataset.get_subset(
        "train",
        transform=transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        ),
    )
    randperm = np.random.permutation(len(train_data))
    train_data = torch.utils.data.Subset(train_data, randperm[:50000])

    #train_loader = get_train_loader("standard", train_data, batch_size=batch_size)
    train_loader = torch.utils.data.DataLoader(
                        train_data,
                        batch_size=batch_size, shuffle=True)
    
    randperm = np.random.permutation(len(train_data))
    client_loaders = []
    for i in range(client_num):
        data_index = randperm[i:-1:client_num]
        client_train = torch.utils.data.Subset(train_data, data_index)
        client_loaders.append(torch.utils.data.DataLoader(
                        client_train,
                        batch_size=batch_size, shuffle=True))

    test_data = dataset.get_subset(
        'test',
        transform=transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        ),
    )
    #randperm = np.random.permutation(len(test_data))
    #test_data = torch.utils.data.Subset(test_data, randperm[:5000])
    #test_loader = get_train_loader("standard", test_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
                        test_data,
                        batch_size=batch_size, shuffle=False)

    analysis_loader = None
    analysis_test_loader = None

    return train_loader, test_loader, analysis_loader, analysis_test_loader, client_loaders, C, transform_to_one_hot, data_params