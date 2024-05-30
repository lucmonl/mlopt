import sys
sys.path.append('..')
from main import graphs
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

from path_manage import get_directory, get_running_directory

#print(graphs)

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
    #print(directory)
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

def plot_train_loss(ax, xaxis, yaxis):
    ax.semilogy(xaxis, yaxis)
    ax.set_xlabel('Epoch')
    ax.set_title('Training Loss')

def plot_train_acc(ax, xaxis, yaxis):
    ax.plot(xaxis, yaxis)
    ax.set_xlabel('Epoch')
    ax.set_title('Training Accuracy')

def plot_train_eig(ax, xaxis, yaxis):
    ax.plot(xaxis, yaxis)
    ax.set_xlabel('Epoch')
    ax.set_title('Sharpness')

def plot_test_loss(ax, xaxis, yaxis):
    ax.semilogy(xaxis, yaxis)
    ax.set_xlabel('Epoch')
    ax.set_title('Test Loss')

def plot_test_acc(ax, xaxis, yaxis):
    ax.plot(xaxis, yaxis)
    ax.set_xlabel('Epoch')
    ax.set_title('Test Accuracy')

def plot_test_eigs(ax, xaxis, yaxis):
    ax.plot(xaxis, yaxis)
    ax.set_xlabel('Epoch')
    ax.set_title('Test Sharpness')

def plot_ascent_step_diff(ax, yaxis):
    ax.plot(yaxis)
    ax.set_xlabel('Epoch')
    ax.set_title('Ascent_step_diff')

def plot_xy(ax, xaxis, yaxis, name, alpha=1.0):
    line = ax.plot(xaxis, yaxis, alpha=alpha)[0]
    ax.set_xlabel('Epoch')
    ax.set_title(name)
    return line

def plot_xlogy(ax, xaxis, yaxis, name):
    line = ax.semilogy(xaxis, yaxis)[0]
    ax.set_xlabel('Epoch')
    ax.set_title(name)
    return line

def plot_y(ax, yaxis, name):
    line = ax.plot(yaxis)[0]
    ax.set_xlabel('Epoch')
    ax.set_title(name)
    return line

def get_attr_from_graph(train_graph, attr):
    if attr == "test_err":
        return 1- np.array(getattr(train_graph, "test_accuracy"))
    else:
        return getattr(train_graph, attr)

def plot_attr(ax, train_graphs, attr, start=None, end=None):
    data = []
    xaxis = None
    for train_graph in train_graphs:
        cur_epochs = train_graph.log_epochs
        if xaxis and len(cur_epochs) == len(xaxis):
            data.append(get_attr_from_graph(train_graph, attr)[start: end])
        elif xaxis is None:
            xaxis = train_graph.log_epochs
            data.append(get_attr_from_graph(train_graph, attr)[start: end])
        else:
            # still running
            continue
    xaxis = xaxis[start: end]
    yaxis = np.mean(np.array(data), axis=0)
    stds = np.std(np.array(data), axis=0)
    #plt.plot(x, means, label="Estimated Mean")

    if attr in ['test_err']:
        #yaxis = 1- np.array(getattr(train_graphs, "test_accuracy")[start:end])
        line = plot_xlogy(ax, xaxis, yaxis, name=attr)
        ax.fill_between(xaxis, yaxis - stds, yaxis + stds, alpha=0.3, label="Confidence Interval")
    elif attr in ['loss', 'train_err', 'test_loss']:
        #yaxis = getattr(train_graphs, attr)[start:end]
        line = plot_xlogy(ax, xaxis, yaxis, name=attr)
        ax.fill_between(xaxis, yaxis - stds, yaxis + stds, alpha=0.3, label="Confidence Interval")
    elif attr in ['accuracy', 'eigs', 'test_accuracy', 'eigs_test']:
        #yaxis = getattr(train_graphs, attr)[start:end]
        line = plot_xy(ax, xaxis, yaxis, name=attr)
        ax.fill_between(xaxis, yaxis - stds, yaxis + stds, alpha=0.3, label="Confidence Interval")
    elif attr in ['cos_descent_ascent', 'progress_dir', 'ascent_semi_cos', 'ascent_step_diff', 'descent_step_diff', 'descent_norm']:
        if hasattr(train_graph, attr):
            #yaxis = getattr(train_graphs, attr)[start:end]
            line = plot_y(ax=ax, yaxis=yaxis, name=attr)
            ax.fill_between(range(len(yaxis)), yaxis - stds, yaxis + stds, alpha=0.3, label="Confidence Interval")
        else:
            line = plot_y(ax=ax, yaxis=[], name=attr)
    else:
        if hasattr(train_graph, attr):
            line = plot_y(ax=ax, yaxis=yaxis, name=attr)
            ax.fill_between(range(len(yaxis)), yaxis - stds, yaxis + stds, alpha=0.3, label="Confidence Interval")
        else:
            line = plot_y(ax=ax, yaxis=[], name=attr)
    return line


def plot_figures_opts_attrs(opts, model_params, opt_params, attrs, start=None, end=None):
    rows, cols = (len(attrs) - 1) // 6 + 1, min(len(attrs), 6)
    fig, axs = plt.subplots(rows,cols, figsize=(cols*2.5, rows*2))
    axs = axs.reshape(-1)
    for opt_name in opts:
        model_param = model_params[opt_name]
        """
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
        """
        directory = get_running_directory(opt_params[opt_name]['lr'], 
                                        opt_params[opt_name]['dataset_name'],
                                        opt_params[opt_name]['loss'],
                                        opt_params[opt_name]['opt'], 
                                        opt_params[opt_name]['model_name'], 
                                        opt_params[opt_name]['momentum'], 
                                        opt_params[opt_name]['weight_decay'], 
                                        opt_params[opt_name]['batch_size'], 
                                        opt_params[opt_name]['epochs'], **model_param)
        run_dir = os.listdir("../" + directory)
        train_graphs = []
        for run_id in run_dir[::-1]:
            with open(f'../{directory}/{run_id}/train_graphs.pk', 'rb') as f:
                train_graphs.append(pickle.load(f))
        #if len(train_graphs.log_epochs) != 0:
        #    cur_epochs = train_graphs.log_epochs
        #else:
        #cur_epochs = np.arange(len(train_graphs.loss))
        #cur_epochs = train_graphs.log_epochs[start:end]
        ax_ptr = 0
        lines = []
        for attr in attrs:
            lines.append(plot_attr(ax=axs[ax_ptr], train_graphs=train_graphs, attr=attr, start=start, end=end))
            ax_ptr += 1
        """
        if 'loss' in attrs:
            plot_train_loss(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=train_graphs.loss[start:end])
            ax_ptr += 1
        
        if 'acc' in attrs:
            plot_train_acc(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=train_graphs.accuracy[start:end])
            ax_ptr += 1

        if 'train_err' in attrs:
            plot_xlogy(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=1-np.array(train_graphs.accuracy[start:end]), name="Train Error")
            ax_ptr += 1

        if 'eigs' in attrs:
            plot_train_eig(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=train_graphs.eigs[start:end])
            ax_ptr += 1

        if 'test_loss' in attrs:
            plot_test_loss(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=train_graphs.test_loss[start:end])
            ax_ptr += 1

        if 'test_acc' in attrs:
            plot_test_acc(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=train_graphs.test_accuracy[start:end])
            ax_ptr += 1

        if 'test_err' in attrs:
            plot_xlogy(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=1-np.array(train_graphs.test_accuracy[start:end]), name="Test Error")
            ax_ptr += 1

        if 'test_eigs' in attrs:
            plot_test_eigs(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=train_graphs.eigs_test[start:end])
            ax_ptr += 1

        if 'progress_dir' in attrs:
            if hasattr(train_graphs, 'progress_dir'):
                plot_y(ax=axs[ax_ptr], yaxis=train_graphs.progress_dir[start:end], name="progress dir")
            else:
                plot_y(ax=axs[ax_ptr], yaxis=[], name="progress dir")
            ax_ptr += 1

        if "ascent_semi_cos" in attrs:
            if hasattr(train_graphs, "ascent_semi_cos"):
                plot_y(ax=axs[ax_ptr], yaxis=train_graphs.ascent_semi_cos[start:end], name="ascent_semi_cos")
            else:
                plot_y(ax=axs[ax_ptr], yaxis=[], name="ascent_semi_cos")
            ax_ptr += 1

        if 'ascent_diff' in attrs:
            if hasattr(train_graphs, 'ascent_step_diff'):
                plot_ascent_step_diff(ax=axs[ax_ptr], yaxis=train_graphs.ascent_step_diff[start:end])
            else:
                plot_ascent_step_diff(ax=axs[ax_ptr], yaxis=[])
            ax_ptr += 1

        if 'descent_diff' in attrs:
            if hasattr(train_graphs, 'descent_step_diff'):
                plot_y(ax=axs[ax_ptr], yaxis=train_graphs.descent_step_diff[start:end], name="descent diff")
            else:
                plot_y(ax=axs[ax_ptr], yaxis=[], name="descent diff")
            ax_ptr += 1

        if 'descent_norm' in attrs:
            if hasattr(train_graphs, 'descent_norm'):
                plot_y(ax=axs[ax_ptr], yaxis=train_graphs.descent_norm[start:end], name="descent_norm")
            else:
                plot_y(ax=axs[ax_ptr], yaxis=[], name="descent_norm")
            ax_ptr += 1

        if 'grad_loss_ratio' in attrs:
            plot_y(ax=axs[ax_ptr], yaxis=train_graphs.wn_grad_loss_ratio[start:end], name="grad_loss_ratio")
            ax_ptr += 1

        if 'wn_norm_min' in attrs:
            plot_y(ax=axs[ax_ptr], yaxis=train_graphs.wn_norm_min[start:end], name="wn_norm_min")
            ax_ptr += 1
        
        """

    axs[0].legend(lines, opts)
    plt.tight_layout()
    plt.show()

def plot_figures_opts_hists(opts, model_params, opt_params, attr, epochs):
    rows, cols = (len(epochs) - 1) // 6 + 1, min(len(epochs), 6)
    fig, axs = plt.subplots(rows,cols, figsize=(cols*2.5, rows*2))
    axs = axs.reshape(-1)
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
        #print(directory)
        with open(f'../{directory}train_graphs.pk', 'rb') as f:
            train_graphs = pickle.load(f)

        ax_ptr = 0
        for epoch in epochs:
            grad_norm = get_attr(opt_name, model_params, opt_params, attr)
            axs[ax_ptr].hist(np.array(grad_norm[epoch]).reshape(-1), bins=20, density=True, alpha=0.7)
            axs[ax_ptr].set_title(epoch)
            ax_ptr += 1

    axs[0].legend(opts)
    plt.tight_layout()
    plt.show()

def plot_figures_opts_attr(opts_list, model_params, opt_params, attr, start=None, end=None, alpha=1.0, legends=[], titles=[], save_dir=None):
    #rows, cols = (len(attrs) - 1) // 6 + 1, min(len(attrs), 6)
    rows, cols = 1, len(opts_list)
    fig, axs = plt.subplots(rows,cols, figsize=(cols*4, rows*3))
    axs = axs.reshape(-1)
    ax_ptr = 0
    for opts, legend, title in zip(opts_list, legends, titles):
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
            cur_epochs = train_graphs.log_epochs[start:end]
            
            if 'loss' == attr:
                plot_train_loss(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=train_graphs.loss[start:end])
            
            if 'acc' == attr:
                plot_train_acc(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=train_graphs.accuracy[start:end])

            if 'train_err' == attr:
                plot_xlogy(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=1-np.array(train_graphs.accuracy[start:end]), name="Train Error")

            if 'test_loss' == attr:
                plot_test_loss(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=train_graphs.test_loss[start:end])

            if 'test_acc' == attr:
                plot_test_acc(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=train_graphs.test_accuracy[start:end])

            if 'test_err' == attr:
                import matplotlib.ticker as mtick
                axs[ax_ptr].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
                plot_xy(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=1-1*np.array(train_graphs.test_accuracy[start:end]), name=title, alpha=alpha)

        axs[ax_ptr].legend(legend)
        ax_ptr += 1
    axs[0].set_ylabel("Test Err")
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir)

def plot_figure_cos_descent_ascent(opts, model_params, opt_params):
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

        cur_epochs = train_graphs.log_epochs
        plt.plot(train_graphs.cos_descent_ascent)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('cos_descent_ascent')