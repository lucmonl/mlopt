import os
import sys

def get_lookup_directory(lr, dataset_name, loss_name, opt_name, model_name, momentum, weight_decay, batch_size, **kwargs):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    results_dir = "results"
    directory = os.path.join(dir_path, f"{results_dir}/{dataset_name}/{loss_name}/{opt_name}/{model_name}/")
    for key, value in kwargs.items():
        directory += f"{key}_{value}/"
    directory += f"lr_{lr}/moment_{momentum}/wd_{weight_decay}/batch_size_{batch_size}/"
    return directory

def get_running_directory(lr, dataset_name, loss_name, opt_name, model_name, momentum, weight_decay, batch_size, epochs, **kwargs):
    return get_lookup_directory(lr, dataset_name, loss_name, opt_name, model_name, momentum, weight_decay, batch_size, **kwargs) + f"epoch_{epochs}/"

def get_directory(lr, dataset_name, loss_name, opt_name, model_name, momentum, weight_decay, batch_size, epochs, multi_run, **kwargs):
    #results_dir = "results"
    #directory = f"{results_dir}/{model_name}/{dataset_name}/{opt_name}/lr_{lr}/wd_{weight_decay}/batch_size_{batch_size}/epoch_{epochs}/"
    directory = get_running_directory(lr, dataset_name, loss_name, opt_name, model_name, momentum, weight_decay, batch_size, epochs, **kwargs)
    if not multi_run or not os.path.exists(directory):
        return directory + "run_0/"
    run_dir = os.listdir(directory)
    prev_runs = [int(x.split("_")[-1]) for x in run_dir if x.startswith("run")]
    return directory + "run_{}".format(len(prev_runs))
    #return directory

def continue_training(lr, dataset_name, loss_name, opt_name, model_name, momentum, weight_decay, batch_size, epochs, multi_run, **kwargs):
    #results_dir = "results"
    #lookup_dir = f"{results_dir}/{model_name}/{dataset_name}/{opt_name}/lr_{lr}/wd_{weight_decay}/batch_size_{batch_size}/"
    lookup_dir = get_lookup_directory(lr, dataset_name, loss_name, opt_name, model_name, momentum, weight_decay, batch_size, **kwargs)
    if not os.path.exists(lookup_dir) or multi_run:
        return 0
    epoch_dir = os.listdir(lookup_dir)
    trained_epochs = [int(x.split("_")[-1]) for x in epoch_dir]
    trained_epochs.sort(reverse=True)
    load_from_epoch = 0
    for trained_epoch in trained_epochs:
        if trained_epoch == epochs:
            print(lookup_dir)
            yes_or_no = input("The path already exists. Are you sure to overwrite? [y/n]")
            if yes_or_no != 'y':
                sys.exit()
        if trained_epoch < epochs:
            load_from_epoch = trained_epoch
            break
    return load_from_epoch

def vit_directory(size, patch_size, img_size, opt_name=None, pretrain="in21k"):
    if opt_name == "sam":
        return "/projects/dali/models/vit/vit_{}_patch{}_{}.sam_{}/pytorch_model.bin".format(size, patch_size, img_size, pretrain)
    elif opt_name == "clip":
        return "/projects/dali/models/vit/vit_{}_patch{}_clip_{}.openai_ft_{}/pytorch_model.bin".format(size, patch_size, img_size, pretrain)
    elif opt_name == "sbb":
        return "/projects/dali/models/vit/vit_{}_patch{}_reg4_gap_{}.sbb_{}/pytorch_model.bin".format(size, patch_size, img_size, pretrain)
    return "/projects/dali/models/vit/vit_{}_patch{}_{}.augreg_{}/pytorch_model.bin".format(size, patch_size, img_size, pretrain)

def pretain_num_classes(pretrain):
    if pretrain == "in21k":
        return 21843
    elif pretrain == "in1k":
        return 1000