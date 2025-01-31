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

def cosine_similarity(u, v, ret_abs=False):
    cosine =  (u @ v / np.linalg.norm(u) / np.linalg.norm(v))
    if ret_abs:
        return np.abs(cosine)
    else:
        return cosine

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

def get_attr(opt_name, model_params, opt_params, attr, eval_graph=False):
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
    if not eval_graph:
        with open(f'../{directory}train_graphs.pk', 'rb') as f:
            train_graphs = pickle.load(f)
    else:
        with open(f'../{directory}eval_graphs.pk', 'rb') as f:
            train_graphs = pickle.load(f)
    return getattr(train_graphs, attr)

def get_attr_eval(opt_name, model_params, opt_params, attr):
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
    with open(f'../{directory}eval_graphs.pk', 'rb') as f:
        train_graphs = pickle.load(f)
    return getattr(train_graphs, attr)

def plot_max_2d(array, k=1, start=0, end=None, xaxis=None):
    assert k >= 1 
    if k > array[0].shape[0]:
        print("WARNING: exceeds the second dim of array. Force round to max dim")
        k = array[0].shape[0]
    for j in range(1,k+1):
        if not xaxis:
            plt.plot([np.partition(np.array(array[start+i]).flatten(), -j)[-j] for i in range(len(array[start:end]))])
        else:
            plt.plot(xaxis, [np.partition(np.array(array[start+i]).flatten(), -j)[-j] for i in range(len(array[start:end]))])

def plot_figures_align(opts, model_params, opt_params, signal_nums=1, topk=1):
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
            plot_max_2d(align, topk, xaxis=cur_epochs)
            #plt.legend(['Loss + Weight Decay'])
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Align Signal {}'.format(i))


        plt.subplot(2,6,i+1)
        #print(train_graphs.eigs)
        align = getattr(train_graphs, "align_noise")
        plot_max_2d(align, 20, xaxis=cur_epochs)
        #plt.legend(['Loss + Weight Decay'])
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Align Noise')

        grads = getattr(train_graphs, "grads")
        signal = getattr(train_graphs, "signal_1").cpu().numpy()
        evecs = getattr(train_graphs, "evec")

        ax = plt.subplot(2, 6, i+2)
        yaxis = [cosine_similarity(grads[i], signal, ret_abs=True) for i in range(len(grads))]
        plot_xy(ax, xaxis=cur_epochs, yaxis=yaxis, name="cos_grad_signal", alpha=1.0)

        ax = plt.subplot(2, 6, i+3)
        yaxis = [cosine_similarity(grads[i], evecs[i], ret_abs=True) for i in range(len(grads))]
        plot_xy(ax, xaxis=cur_epochs, yaxis=yaxis, name="cos_grad_evec", alpha=1.0)

        ax = plt.subplot(2, 6, i+4)
        yaxis = [cosine_similarity(evecs[i], signal, ret_abs=True) for i in range(len(grads))]
        plot_xy(ax, xaxis=cur_epochs, yaxis=yaxis, name="cos_evec_signal", alpha=1.0)
        """
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
        """

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

def plot_xy(ax, xaxis, yaxis, name, alpha=1.0, linewidth=1.0):
    line = ax.plot(xaxis, yaxis, alpha=alpha, linewidth=linewidth)[0]
    ax.set_xlabel('Epoch')
    ax.set_title(name)
    return line

def plot_xlogy(ax, xaxis, yaxis, name, alpha=1.0):
    line = ax.semilogy(xaxis, yaxis, alpha=alpha, linewidth=0.5)[0]
    ax.set_xlabel('Epoch')
    ax.set_title(name)
    return line

def plot_y(ax, yaxis, name):
    line = ax.plot(yaxis, linewidth=0.5)[0]
    ax.set_xlabel('Epoch')
    ax.set_title(name)
    return line

def get_attr_from_graph(train_graph, attr):
    if attr == "test_err":
        return 1- np.array(getattr(train_graph, "test_accuracy"))
    elif attr == "loss_ratio":
        loss = np.array(getattr(train_graph, "loss"))
        return loss[1:] / loss[:-1]
    else:
        return getattr(train_graph, attr)
    


def plot_attr(ax, train_graphs, attr, start=None, end=None):
    data = []
    xaxis = None
    print(len(train_graphs))
    if attr == "gen_loss":
        for train_graph in train_graphs:
            cur_epochs = train_graph.log_epochs
            if xaxis and len(cur_epochs) == len(xaxis):
                data.append(np.array(get_attr_from_graph(train_graph, 'test_loss')[start: end]) - np.array(get_attr_from_graph(train_graph, 'loss')[start: end]))
            elif xaxis is None:
                xaxis = train_graph.log_epochs
                data.append(np.array(get_attr_from_graph(train_graph, 'test_loss')[start: end]) - np.array(get_attr_from_graph(train_graph, 'loss')[start: end]))
            else:
                # still running
                continue
    else:
        for train_graph in train_graphs:
            cur_epochs = train_graph.log_epochs
            if xaxis and len(cur_epochs) == len(xaxis):
                data.append(get_attr_from_graph(train_graph, attr)[start: end])
            elif xaxis is None:
                xaxis = train_graph.log_epochs
                data.append(get_attr_from_graph(train_graph, attr)[start: end])
            else:
                # still running
                print("skip ongoing runs")
                continue
    #if attr == "loss":
    #    print(data)
    #plt.plot(x, means, label="Estimated Mean")
    xaxis = xaxis[start: end]
    yaxis = np.mean(np.array(data), axis=0).reshape(-1)
    stds = np.std(np.array(data), axis=0).reshape(-1)

    if attr in ['test_err']:
        #yaxis = 1- np.array(getattr(train_graphs, "test_accuracy")[start:end])
        line = plot_xlogy(ax, xaxis, yaxis, name=attr)
        ax.fill_between(xaxis, yaxis - stds, yaxis + stds, alpha=0.3, label="Confidence Interval")
    elif attr in ['loss', 'train_err', 'test_loss']:
        #yaxis = getattr(train_graphs, attr)[start:end]
        line = plot_xlogy(ax, xaxis, yaxis, name=attr)
        ax.fill_between(xaxis, yaxis - stds, yaxis + stds, alpha=0.3, label="Confidence Interval")
    elif attr in ['accuracy',  'test_accuracy', 'wn_grad_loss_ratio', 'wn_norm_min', "grad_evecs_cos", "residuals", 'eigs',  'eigs_test', 'gen_loss']:
        #yaxis = getattr(train_graphs, attr)[start:end]
        line = plot_xy(ax, xaxis, yaxis, name=attr)
        ax.fill_between(xaxis, yaxis - stds, yaxis + stds, alpha=0.3, label="Confidence Interval")
    elif attr in ['loss_ratio']:
        line = plot_xy(ax, xaxis[:-1], yaxis, name=attr)
        ax.fill_between(xaxis[:-1], yaxis - stds, yaxis + stds, alpha=0.3, label="Confidence Interval")
    elif attr in ['cos_descent_ascent', 'progress_dir', 'ascent_semi_cos', \
                  'ascent_step_diff', 'descent_step_diff', 'descent_norm', \
                  'dominant_alignment', 'batch_loss', "fedlora_A_align", "fedlora_B_align", \
                  "fedlora_A_cosine", "fedlora_B_cosine", "lora_A_norm", "lora_B_norm", "grad_norm", "truncate_err"]:
        if hasattr(train_graph, attr):
            #yaxis = getattr(train_graphs, attr)[start:end]
            line = plot_y(ax=ax, yaxis=yaxis, name=attr)
            ax.fill_between(range(len(yaxis)), yaxis - stds, yaxis + stds, alpha=0.3, label="Confidence Interval")
        else:
            line = plot_y(ax=ax, yaxis=[], name=attr)
    else:
        raise NotImplementedError
        """
        if hasattr(train_graph, attr):
            line = plot_y(ax=ax, yaxis=yaxis, name=attr)
            ax.fill_between(range(len(yaxis)), yaxis - stds, yaxis + stds, alpha=0.3, label="Confidence Interval")
        else:
            line = plot_y(ax=ax, yaxis=[], name=attr)
        """
    return line

def plot_single_attr(opts, model_params, opt_params, attr, savefig=None, title=None, xlabel=None, ylabel=None, eval=False):
    fig, ax = plt.subplots(figsize=(6, 5))
    lines = []
    for opt_name in opts:
        model_param = model_params[opt_name]
        directory = get_running_directory(opt_params[opt_name]['lr'], 
                                        opt_params[opt_name]['dataset_name'],
                                        opt_params[opt_name]['loss'],
                                        opt_params[opt_name]['opt'], 
                                        opt_params[opt_name]['model_name'], 
                                        opt_params[opt_name]['momentum'], 
                                        opt_params[opt_name]['weight_decay'], 
                                        opt_params[opt_name]['batch_size'], 
                                        opt_params[opt_name]['epochs'], **model_param)
        print(directory)
        run_dir = os.listdir("../" + directory)
        train_graphs = []
        for run_id in run_dir[::-1]:
            if not eval: 
                with open(f'../{directory}/{run_id}/train_graphs.pk', 'rb') as f:
                    train_graphs.append(pickle.load(f))
            else:
                print("loading eval graph")
                with open(f'../{directory}/{run_id}/eval_graphs.pk', 'rb') as f:
                    train_graphs.append(pickle.load(f))
        lines.append(plot_attr(ax=ax, train_graphs=train_graphs, attr=attr))
    ax.legend(lines, opts)

    if title is not None:
        plt.title(title)
    else:
        plt.title("")
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig)
    plt.show()


def plot_figures_opts_attrs(opts, model_params, opt_params, attrs, start=None, end=None, eval=False, return_last=False):
    rows, cols = (len(attrs) - 1) // 6 + 1, min(len(attrs), 6)
    cols = 2 if cols == 1 else cols # avoid error in subplots
    dpi_scale = 1
    fig, axs = plt.subplots(rows,cols, figsize=(cols*2.5*dpi_scale, rows*2*dpi_scale), dpi=500)
    axs = axs.reshape(-1)
    lines = []
    last_vals = []
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
        print(directory)
        #run_dir = os.listdir("../" + directory)
        run_dir = os.listdir(directory)
        train_graphs = []
        for run_id in run_dir[::-1]:
            if not eval: 
                with open(f'{directory}/{run_id}/train_graphs.pk', 'rb') as f:
                    train_graphs.append(pickle.load(f))
            else:
                print("loading eval graph")
                with open(f'{directory}/{run_id}/eval_graphs.pk', 'rb') as f:
                    train_graphs.append(pickle.load(f))
        #if len(train_graphs.log_epochs) != 0:
        #    cur_epochs = train_graphs.log_epochs
        #else:
        #cur_epochs = np.arange(len(train_graphs.loss))
        #cur_epochs = train_graphs.log_epochs[start:end]
        ax_ptr = 0
        #print(opt_name)
        for attr in attrs:
            line = plot_attr(ax=axs[ax_ptr], train_graphs=train_graphs, attr=attr, start=start, end=end)
            ax_ptr += 1
            if attr == attrs[0]:
                lines.append(line)
            if attr == "test_err":
                last_vals.append(line.get_data()[-1][-1])
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
    if return_last:
        return last_vals

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


def plot_figures_attrs_hists(opt, model_params, opt_params, attrs, epochs):
    rows, cols = (len(epochs) - 1) // 6 + 1, min(len(epochs), 6)
    fig, axs = plt.subplots(rows,cols, figsize=(cols*2.5, rows*2))
    axs = axs.reshape(-1)
    opt_name = opt
    for attr in attrs:
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

    axs[0].legend(attrs)
    plt.tight_layout()
    plt.show()

def plot_figures_opts_attr(opts_list, model_params, opt_params, attrs, start=None, end=None, alpha=1.0, linewidth=1.0, legends=[], titles=[], yaxis=[], save_dir=None, return_last=False):
    #rows, cols = (len(attrs) - 1) // 6 + 1, min(len(attrs), 6)
    import matplotlib.ticker as mtick
    rows, cols = 1, len(opts_list)
    fig, axs = plt.subplots(rows,cols, figsize=(cols*4, rows*3))
    last_val = []
    if len(opts_list) > 1:
        axs = axs.reshape(-1)
    else:
        axs = [axs]
    ax_ptr = 0

    if isinstance(attrs, str):
        attrs = [attrs for i in range(len(opts_list))]
    if isinstance(yaxis, str):
        yaxis = [yaxis for i in range(len(opts_list))]

    for opts, legend, title, attr in zip(opts_list, legends, titles, attrs):
        last_val.append([])
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
            with open(f'{directory}train_graphs.pk', 'rb') as f:
                train_graphs = pickle.load(f)

            #if len(train_graphs.log_epochs) != 0:
            #    cur_epochs = train_graphs.log_epochs
            #else:
            #cur_epochs = np.arange(len(train_graphs.loss))
            cur_epochs = train_graphs.log_epochs[start:end]
            
            if 'loss' == attr:
                plot_train_loss(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=train_graphs.loss[start:end])
            
            if 'acc' == attr:
                axs[ax_ptr].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
                plot_train_acc(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=train_graphs.accuracy[start:end])

            if 'train_err' == attr:
                line=plot_xy(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=1-1*np.array(train_graphs.accuracy[start:end]), name=title, alpha=alpha)
                axs[ax_ptr].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
                #axs[ax_ptr].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
                #plot_xlogy(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=1-np.array(train_graphs.accuracy[start:end]), name="Train Error")

            if 'test_loss' == attr:
                plot_test_loss(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=train_graphs.test_loss[start:end])

            if 'test_acc' == attr:
                axs[ax_ptr].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
                plot_test_acc(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=train_graphs.test_accuracy[start:end])

            if 'test_err' == attr:
                line=plot_xy(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=1-1*np.array(train_graphs.test_accuracy[start:end]), name=title, alpha=alpha)
                axs[ax_ptr].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
                last_val[-1].append(line.get_data()[-1][-1])

        axs[ax_ptr].legend(legend)
        ax_ptr += 1
    if yaxis == []:
        axs[0].set_ylabel("Val Error")
    else:
        for i in range(len(yaxis)):
            axs[i].set_ylabel(yaxis[i])
    axs[0].set_ylim([0.01, 0.07])

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir)
    if return_last:
        return last_val

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

def plot_eval_attr_keys(opt_name, model_params, opt_params, attrs, keys, zero_out_top, zero_out_selfattn):
    #rows, cols = (len(attr) - 1) // 6 + 1, min(len(epochs), 6)
    rows, cols = 1, 2
    fig, axs = plt.subplots(rows,cols, figsize=(cols*2.5, rows*2))
    axs = axs.reshape(-1)

    ax_ptr = 0
    for attr in attrs:
        attr_dict = get_attr(opt_name, model_params, opt_params, attr, eval_graph=True)
        print(attr_dict[zero_out_top].keys())
        for key in keys:
            if key != -1:
                attr_list = attr_dict[zero_out_top][zero_out_selfattn][key]
            else:
                attr_list = attr_dict[key]
            axs[ax_ptr].plot(attr_list)
        axs[ax_ptr].set_title(attr)
        ax_ptr += 1
        

    axs[0].legend(keys)
    plt.tight_layout()
    plt.show()

def plot_eigen_density(model_param, opt_param, start=None, end=None, save_dir=None):
    from utilities import get_esd_plot
    directory = get_directory(opt_param['lr'], 
                            opt_param['dataset_name'],
                            opt_param['loss'],
                            opt_param['opt'], 
                            opt_param['model_name'], 
                            opt_param['momentum'], 
                            opt_param['weight_decay'], 
                            opt_param['batch_size'], 
                            opt_param['epochs'], 
                            multi_run = False,
                            **model_param
                            )
    print(directory)
    with open(f'../{directory}train_graphs.pk', 'rb') as f:
        eval_graphs = pickle.load(f)
    
    cur_epochs = eval_graphs.log_epochs
    density_eigens, density_weights = eval_graphs.density_eigen, eval_graphs.density_weight
    density_eigens, density_weights, epochs = density_eigens[start:end], density_weights[start:end], cur_epochs[start:end]
    rows, cols = (len(density_eigens)-1) // 4 + 1, min(len(density_eigens), 4)
    fig, axs = plt.subplots(rows,cols, figsize=(cols*2.5, rows*2))
    axs = axs.reshape(-1)
    for i in range(len(density_eigens)):
        ax, density_eigen, density_weight = axs[i], density_eigens[i], density_weights[i]
    #for ax, density_eigen, density_weight in zip(axs, density_eigens, density_weights):
        #plt.figure(figsize=(3,3))
        get_esd_plot(ax, density_eigen, density_weight, epochs[i], ylabel=i%6)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir)

def plot_attention_map(model_param, opt_param, depths=None):
    directory = get_directory(opt_param['lr'], 
                            opt_param['dataset_name'],
                            opt_param['loss'],
                            opt_param['opt'], 
                            opt_param['model_name'], 
                            opt_param['momentum'], 
                            opt_param['weight_decay'], 
                            opt_param['batch_size'], 
                            opt_param['epochs'], 
                            multi_run = False,
                            **model_param
                            )
    print(directory)
    with open(f'../{directory}eval_graphs.pk', 'rb') as f:
        eval_graphs = pickle.load(f)
    if depths is None:
        depths = np.arange(len(eval_graphs.attention_map))
    print(depths)
    for depth in depths:
        attention_maps = eval_graphs.attention_map[depth]
        rows, cols = (len(attention_maps)+1) // 6 + 1, min(len(attention_maps)+2, 6)
        #rows, cols = (len(epochs) - 1) // 6 + 1, min(len(epochs), 6)
        ax_itr = 0
        fig, axs = plt.subplots(rows,cols, figsize=(cols*2.5, rows*2))
        axs = axs.reshape(-1)
        ax = axs[ax_itr]
        raw_image = np.array(eval_graphs.test_img[0])#.transpose([1,2,0])
        ax.imshow(raw_image)

        for i in range(len(attention_maps)):
            ax_itr+=1
            attention_map = np.array(attention_maps[i])
            axs[ax_itr].imshow(attention_map)

        output_norm = eval_graphs.output_norm[0]
        ax_itr+=1
        axs[ax_itr].imshow(output_norm[0])
    return raw_image, eval_graphs.attention_map, output_norm

def plot_loss_ratio_vs_grad(opts, model_params, opt_params, savefig=None):
    fig, ax1 = plt.subplots(figsize=[6, 5])
    ax2 = ax1.twinx()

    lines = []
    line_styles = ["solid", "dotted"]
    colors = ["#69b3a2", "red"]

    for i, opt_name in enumerate(opts):
        model_param = model_params[opt_name]
        directory = get_running_directory(opt_params[opt_name]['lr'], 
                                        opt_params[opt_name]['dataset_name'],
                                        opt_params[opt_name]['loss'],
                                        opt_params[opt_name]['opt'], 
                                        opt_params[opt_name]['model_name'], 
                                        opt_params[opt_name]['momentum'], 
                                        opt_params[opt_name]['weight_decay'], 
                                        opt_params[opt_name]['batch_size'], 
                                        opt_params[opt_name]['epochs'], **model_param)
        run_dir = os.listdir(f'../{directory}')
        prev_runs = [int(x.split("_")[-1]) for x in run_dir if x.startswith("run")]
        loss_list, acc_list, grad_loss_ratio_list, test_acc_list, wn_norm_list = [], [], [],[],[]
        
        for run in prev_runs:
            with open(f'../{directory}run_{run}/train_graphs.pk', 'rb') as f:
                train_graphs = pickle.load(f)
                loss_list.append(train_graphs.loss)
                acc_list.append(train_graphs.accuracy)
                grad_loss_ratio_list.append(train_graphs.wn_grad_loss_ratio)
                test_acc_list.append(train_graphs.test_accuracy)
                wn_norm_list.append(train_graphs.wn_norm_min)
                cur_epochs = train_graphs.log_epochs
        
        #print(np.array(loss_list).shape)
        print(loss_list[0][-1])
        loss_list = np.array(loss_list)[:,:50]
        loss_ratio = loss_list[:,1:]/loss_list[:,:-1]
        print(len(cur_epochs))
        print(loss_ratio.shape)
        ax1.plot(cur_epochs[:-1], loss_ratio[0], linestyle=line_styles[i], color=colors[0])
        
        #plt.legend(['Loss + Weight Decay'])
        ax1.set_ylabel(r"$L(\theta_{t+1}) / L(\theta_t)$")

        #print(train_graphs.wn_grad_loss_ratio)
        means = np.mean(np.array(grad_loss_ratio_list), axis=0)[:50]
        lines.append(ax2.plot(cur_epochs[:-1], means[:-1], linestyle=line_styles[i], color=colors[1], label=opt_name)[0])
        ax2.set_ylabel(r"$\Vert \nabla L(\theta_t) \Vert^2$ / $L(\theta_t)$")
        ax1.set_xlabel("Epochs")
    ax1.yaxis.label.set_color(colors[0])
    ax2.yaxis.label.set_color(colors[1])
    print(lines)
    ax2.legend(lines, [l.get_label() for l in lines], loc='center right')
    leg = ax2.get_legend()
    leg.legendHandles[0].set_color('black')
    leg.legendHandles[1].set_color('black')

    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig)