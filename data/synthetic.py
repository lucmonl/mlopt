import numpy as np
import torch
import os
from torch.utils.data.dataset import TensorDataset

DATA_FOLDER = "/projects/dali/data/synthetic/"

def generate_weight_norm(train_size, num_pixels, C):
    from arch.weight_norm import weight_norm_net

    width = 4096
    wn_init_mode = "O(1/sqrt{m})"
    wn_basis_var = 5
    wn_scale = 1

    data_path = DATA_FOLDER + f"teacher_weight_norm/{num_pixels}/{train_size}/"

    model = weight_norm_net(num_pixels, [width, width], wn_init_mode, wn_basis_var, wn_scale, C)
    X_train, X_test = torch.randn(train_size, num_pixels), torch.randn(train_size, num_pixels)
    y_train = 2*((model(X_train) > 0).float() - 0.5)
    y_test = 2*((model(X_test) > 0).float() - 0.5)

    os.makedirs(data_path, exist_ok=True)
    torch.save(X_train, data_path + "X_train.pt")
    torch.save(X_test, data_path + "X_test.pt")
    torch.save(y_train, data_path + "y_train.pt")
    torch.save(y_test, data_path + "y_test.pt")

    
def load_weight_norm_teacher(num_pixels, train_size, batch_size):
    data_params = {"compute_acc": True}
    C = 1
    transform_to_one_hot = False

    data_path = DATA_FOLDER + f"teacher_weight_norm/{num_pixels}/{train_size}/"
    if not os.path.exists(data_path):
        generate_weight_norm(train_size, num_pixels, C)

    X_train, X_test = torch.load(data_path + "X_train.pt"), torch.load(data_path + "X_test.pt")
    y_train, y_test = torch.load(data_path + "y_train.pt"), torch.load(data_path + "y_test.pt")

    train = TensorDataset(X_train, y_train)
    test = TensorDataset(X_test, y_test)

    analysis_size = max(batch_size, 128)
    analysis = torch.utils.data.Subset(train, range(analysis_size))
    analysis_test = torch.utils.data.Subset(test, range(analysis_size))
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size, shuffle=False)
    analysis_loader = torch.utils.data.DataLoader(
        analysis,
        batch_size=analysis_size, shuffle=False)
    analysis_test_loader = torch.utils.data.DataLoader(
        analysis_test,
        batch_size=analysis_size, shuffle=False)
    return train_loader, test_loader, analysis_loader, analysis_test_loader, num_pixels, C, transform_to_one_hot, data_params
