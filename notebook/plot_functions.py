import sys
sys.path.append('..')
from main import graphs
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

from path_manage import get_directory

print(graphs)

def plot_figures_opts(opts, model_params, opt_params):
    plt.figure(figsize=(15,5))
    for opt_name in opts:
        model_param = model_params[opt_name]
        directory = get_directory(opt_params[opt_name]['lr'], 
                                opt_params[opt_name]['dataset_name'],
                                opt_params[opt_name]['loss'],
                                opt_params[opt_name]['opt'], 
                                opt_params[opt_name]['model_name'], 
                                opt_params[opt_name]['momentum'], 
                                opt_params[opt_name]['weight_decay'], 
                                opt_params[opt_name]['batch_size'], 
                                opt_params[opt_name]['epochs'], 
                                multi_run = False,
                                **model_param
                                )
        print(directory)
        with open(f'../{directory}train_graphs.pk', 'rb') as f:
            train_graphs = pickle.load(f)

        #if len(train_graphs.log_epochs) != 0:
        #    cur_epochs = train_graphs.log_epochs
        #else:
        #cur_epochs = np.arange(len(train_graphs.loss))
        cur_epochs = train_graphs.log_epochs
        plt.subplot(2,6,1)
        #print(cur_epochs)
        #print(train_graphs.loss)
        plt.semilogy(cur_epochs, train_graphs.loss)
        #plt.legend(['Loss + Weight Decay'])
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Loss')


        plt.subplot(2,6,2)
        #print(train_graphs.eigs)
        plt.semilogy(cur_epochs, train_graphs.eigs)
        #plt.legend(['Loss + Weight Decay'])
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Sharpness')

        plt.subplot(2,6,3)
        plt.semilogy(cur_epochs, train_graphs.test_loss)
        #.legend(['Loss + Weight Decay'])
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Testing Loss')

        plt.subplot(2,6,4)
        plt.semilogy(cur_epochs, train_graphs.test_accuracy)
        #.legend(['Loss + Weight Decay'])
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Testing Accuracy')

        plt.subplot(2,6,5)
        plt.semilogy(cur_epochs, train_graphs.eigs_test)
        #.legend(['Loss + Weight Decay'])
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Testing Sharpness')
        
        if "adv_eta" in opt_params[opt_name]:
            plt.subplot(2,6,6)
            plt.semilogy(cur_epochs, train_graphs.adv_eigs[opt_params[opt_name]["adv_eta"]])
            #.legend(['Loss + Weight Decay'])
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Adversarial Sharpness')

    plt.legend(opts)
    plt.tight_layout()
    plt.show()

def get_attr(opt_name, model_params, opt_params, attr):
    model_param = model_params[opt_name]
    directory = get_directory(opt_params[opt_name]['lr'], 
                            opt_params[opt_name]['dataset_name'],
                            opt_params[opt_name]['loss'],
                            opt_params[opt_name]['opt'], 
                            opt_params[opt_name]['model_name'], 
                            opt_params[opt_name]['momentum'], 
                            opt_params[opt_name]['weight_decay'], 
                            opt_params[opt_name]['batch_size'], 
                            opt_params[opt_name]['epochs'], 
                            multi_run = False,
                            **model_param
                            )
    print(directory)
    with open(f'../{directory}train_graphs.pk', 'rb') as f:
        train_graphs = pickle.load(f)
    return getattr(train_graphs, attr)

def plot_max_2d(array, k=1, start=0, end=None, xaxis=None):

    assert k >= 1 
    if k > len(array[0]):
        print("WARNING: exceeds the second dim of array. Force round to max dim")
        k = len(array[0])
    for j in range(1,k+1):
        if not xaxis:
            plt.plot([np.partition(np.array(array[start+i]).flatten(), -j)[-j] for i in range(len(array[start:end]))])
        else:
            plt.plot(xaxis, [np.partition(np.array(array[start+i]).flatten(), -j)[-j] for i in range(len(array[start:end]))])

def plot_figures_align(opts, model_params, opt_params, signal_nums=1):
    plt.figure(figsize=(15,5))
    for opt_name in opts:
        model_param = model_params[opt_name]
        directory = get_directory(opt_params[opt_name]['lr'], 
                                opt_params[opt_name]['dataset_name'],
                                opt_params[opt_name]['loss'],
                                opt_params[opt_name]['opt'], 
                                opt_params[opt_name]['model_name'], 
                                opt_params[opt_name]['momentum'], 
                                opt_params[opt_name]['weight_decay'], 
                                opt_params[opt_name]['batch_size'], 
                                opt_params[opt_name]['epochs'], 
                                multi_run = False,
                                **model_param
                                )
        print(directory)
        with open(f'../{directory}train_graphs.pk', 'rb') as f:
            train_graphs = pickle.load(f)

        #if len(train_graphs.log_epochs) != 0:
        #    cur_epochs = train_graphs.log_epochs
        #else:
        #cur_epochs = np.arange(len(train_graphs.loss))
        cur_epochs = train_graphs.log_epochs

        for i in range(1, signal_nums+1):
            plt.subplot(2,6,i)
            #print(cur_epochs)
            #print(train_graphs.loss)
            #plt.semilogy(cur_epochs, train_graphs.loss)
            align = getattr(train_graphs, "align_signal_{}".format(i))
            plot_max_2d(align, 4, xaxis=cur_epochs)
            #plt.legend(['Loss + Weight Decay'])
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Align Signal {}'.format(i))


        plt.subplot(2,6,i+1)
        #print(train_graphs.eigs)
        align = getattr(train_graphs, "align_noise")
        plot_max_2d(align, 4, xaxis=cur_epochs)
        #plt.legend(['Loss + Weight Decay'])
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Align Noise')

        plt.subplot(2,6,i+2)
        out_layer = getattr(train_graphs, "linear_coefs") # get_attr('sgd-0.05', model_params, opt_params, "out_layer")
        plot_max_2d(out_layer, 2, xaxis=cur_epochs)
        #.legend(['Loss + Weight Decay'])
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Output Layer')

        plt.subplot(2,6,i+3)
        model_output = getattr(train_graphs, "model_output") # get_attr('sgd-0.05', model_params, opt_params, "out_layer")
        plt.plot(cur_epochs, [np.sum(model_output[i] > 0) for i in range(len(model_output))])
        #.legend(['Loss + Weight Decay'])
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title("(out-target)*target > 0")
      

    plt.legend(opts)
    plt.tight_layout()
    plt.show()

def plot_attr_figure(opts, model_params, opt_params, attr_name):
    plt.figure(figsize=(4,4))
    for opt_name in opts:
        model_param = model_params[opt_name]
        directory = get_directory(opt_params[opt_name]['lr'], 
                                opt_params[opt_name]['dataset_name'],
                                opt_params[opt_name]['loss'],
                                opt_params[opt_name]['opt'], 
                                opt_params[opt_name]['model_name'], 
                                opt_params[opt_name]['momentum'], 
                                opt_params[opt_name]['weight_decay'], 
                                opt_params[opt_name]['batch_size'], 
                                opt_params[opt_name]['epochs'], 
                                multi_run = False,
                                **model_param
                                )
        print(directory)
        with open(f'../{directory}train_graphs.pk', 'rb') as f:
            train_graphs = pickle.load(f)

        #if len(train_graphs.log_epochs) != 0:
        #    cur_epochs = train_graphs.log_epochs
        #else:
        #cur_epochs = np.arange(len(train_graphs.loss))
        cur_epochs = train_graphs.log_epochs
        if attr_name == "activation_pattern":
            activation = getattr(train_graphs, "activation_pattern")
            plt.plot(cur_epochs, [np.sum(activation[i][1]) for i in range(len(activation))])
            #.legend(['Loss + Weight Decay'])
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title("# of activated neurons")
        elif attr_name == "loss":
            train_loss = getattr(train_graphs, "loss")
            plt.semilogy(cur_epochs, train_loss)
            #.legend(['Loss + Weight Decay'])
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title("Loss")
    plt.tight_layout()
    plt.legend(opts)
    plt.show()

