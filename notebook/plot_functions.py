import sys
sys.path.append('..')
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

def get_attr(opt_name, model_params, opt_params, attr, eval_graph=False, return_x =False):
    model_param = model_params[opt_name]
    directory = get_running_directory(opt_params[opt_name]['lr'], 
                            opt_params[opt_name]['dataset_name'],
                            opt_params[opt_name]['loss'],
                            opt_params[opt_name]['opt'], 
                            opt_params[opt_name]['model_name'], 
                            opt_params[opt_name]['momentum'], 
                            opt_params[opt_name]['weight_decay'], 
                            opt_params[opt_name]['batch_size'], 
                            opt_params[opt_name]['epochs'], 
                            **model_param
                            )
    print(directory)
    run_dir = os.listdir(directory)
    xaxis = []
    return_attr = []
    for run_id in run_dir:
        if not eval_graph:
            with open(f'{directory}/{run_id}/train_graphs.pk', 'rb') as f:
                train_graphs = pickle.load(f)
        else:
            with open(f'{directory}/{run_id}/eval_graphs.pk', 'rb') as f:
                train_graphs = pickle.load(f)
        xaxis.append(train_graphs.log_epochs)
        return_attr.append(getattr(train_graphs, attr))
    if return_x:
        return xaxis, return_attr
    else:
        return return_attr

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
    ax.set_xlabel('Communication Round')
    #ax.set_title('Training Accuracy')

def plot_train_eig(ax, xaxis, yaxis):
    ax.plot(xaxis, yaxis)
    ax.set_xlabel('Epoch')
    ax.set_title('Sharpness')

def plot_test_loss(ax, xaxis, yaxis, xlabel, title):
    #ax.semilogy(xaxis, yaxis)
    ax.plot(xaxis, yaxis)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

def plot_test_acc(ax, xaxis, yaxis):
    ax.plot(xaxis, yaxis)
    #ax.set_xlabel('Epoch')
    ax.set_xlabel('Communication Round')
    ax.set_title('Test Accuracy')

def plot_test_eigs(ax, xaxis, yaxis):
    ax.plot(xaxis, yaxis)
    ax.set_xlabel('Epoch')
    ax.set_title('Test Sharpness')

def plot_ascent_step_diff(ax, yaxis):
    ax.plot(yaxis)
    ax.set_xlabel('Epoch')
    ax.set_title('Ascent_step_diff')

def plot_xy(ax, xaxis, yaxis, name, alpha=1.0, linewidth=1.0, linestyle='solid'):
    color = 'gray' if linestyle == 'dashed' else None
    alpha = 0.5 if linestyle == 'dashed' else alpha
    line = ax.plot(xaxis, yaxis, alpha=alpha, linewidth=linewidth, linestyle=linestyle, color=color)[0]
    ax.set_xlabel('Communication Round')
    ax.set_title(name)
    return line

def plot_xlogy(ax, xaxis, yaxis, name, alpha=1.0, linewidth=1.0, linestyle='solid'):
    color = 'gray' if linestyle == 'dashed' else None
    alpha = 0.5 if linestyle == 'dashed' else alpha
    line = ax.semilogy(xaxis, yaxis, alpha=alpha, linewidth=linewidth, linestyle=linestyle, color=color)[0]
    ax.set_xlabel('Epoch')
    ax.set_title(name)
    return line

def plot_y(ax, yaxis, name, linestyle='solid'):
    color = 'gray' if linestyle == 'dashed' else None
    alpha = 0.5 if linestyle == 'dashed' else 1.0
    line = ax.plot(yaxis, linewidth=0.5, linestyle=linestyle, color=color, alpha=alpha)[0]
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
    


def plot_attr(ax, train_graphs, attr, start=None, end=None, alpha=1.0, linewidth=1.0, linestyle='solid'):
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
        line = plot_xlogy(ax, xaxis, yaxis, name=attr, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
        ax.fill_between(xaxis, yaxis - stds, yaxis + stds, alpha=alpha, label="Confidence Interval")
    elif attr in ['loss', 'train_err', 'test_loss']:
        #yaxis = getattr(train_graphs, attr)[start:end]
        line = plot_xlogy(ax, xaxis, yaxis, name=attr, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
        ax.fill_between(xaxis, yaxis - stds, yaxis + stds, alpha=alpha, label="Confidence Interval")
    elif attr in ['accuracy',  'test_accuracy', 'wn_grad_loss_ratio', 'wn_norm_min', "grad_evecs_cos", "residuals", 'eigs',  'eigs_test', 'gen_loss']:
        #yaxis = getattr(train_graphs, attr)[start:end]
        line = plot_xy(ax, xaxis, yaxis, name=attr, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
        ax.fill_between(xaxis, yaxis - stds, yaxis + stds, alpha=alpha, label="Confidence Interval")
    elif attr in ['loss_ratio']:
        line = plot_xy(ax, xaxis[:-1], yaxis, name=attr, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
        ax.fill_between(xaxis[:-1], yaxis - stds, yaxis + stds, alpha=alpha, label="Confidence Interval")
    elif attr in ['cos_descent_ascent', 'progress_dir', 'ascent_semi_cos', \
                  'ascent_step_diff', 'descent_step_diff', 'descent_norm', \
                  'dominant_alignment', 'batch_loss', "fedlora_A_align", "fedlora_B_align", \
                  "fedlora_A_cosine", "fedlora_B_cosine", "lora_A_norm", "lora_B_norm", "grad_norm", "truncate_err"]:
        if hasattr(train_graph, attr):
            #yaxis = getattr(train_graphs, attr)[start:end]
            line = plot_y(ax=ax, yaxis=yaxis, name=attr, linestyle=linestyle)
            ax.fill_between(range(len(yaxis)), yaxis - stds, yaxis + stds, alpha=alpha, label="Confidence Interval")
        else:
            line = plot_y(ax=ax, yaxis=[], name=attr, linestyle=linestyle)
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

def plot_attr_overlap(ax, train_graphs, attr, start=None, end=None, alpha=1.0, linewidth=1.0, linestyle='solid'):
    """Like plot_attr but truncates all runs to their shortest overlapping range."""
    data = []
    xaxes = []
    print(len(train_graphs))
    if attr == "gen_loss":
        for train_graph in train_graphs:
            raw = np.array(get_attr_from_graph(train_graph, 'test_loss')[start:end]) - \
                  np.array(get_attr_from_graph(train_graph, 'loss')[start:end])
            data.append(raw)
            xaxes.append(train_graph.log_epochs[start:end])
    else:
        for train_graph in train_graphs:
            data.append(np.array(get_attr_from_graph(train_graph, attr)[start:end]))
            xaxes.append(train_graph.log_epochs[start:end])

    min_len = min(len(d) for d in data)
    data = [d[:min_len] for d in data]
    xaxis = xaxes[0][:min_len]

    yaxis = np.mean(np.array(data), axis=0).reshape(-1)
    stds = np.std(np.array(data), axis=0).reshape(-1)# * 3

    if attr in ['test_err', 'loss', 'train_err', 'test_loss']:
        line = plot_xy(ax, xaxis, yaxis, name=attr, alpha=alpha, linestyle=linestyle)
        ax.fill_between(xaxis, yaxis - stds, yaxis + stds, alpha=0.3, label="Confidence Interval")
    elif attr in ['accuracy', 'test_accuracy', 'wn_grad_loss_ratio', 'wn_norm_min', "grad_evecs_cos",
                  "residuals", 'eigs', 'eigs_test', 'gen_loss']:
        line = plot_xy(ax, xaxis, yaxis, name=attr, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
        ax.fill_between(xaxis, yaxis - stds, yaxis + stds, alpha=0.3, label="Confidence Interval")
    elif attr in ['loss_ratio']:
        line = plot_xy(ax, xaxis[:-1], yaxis, name=attr, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
        ax.fill_between(xaxis[:-1], yaxis - stds, yaxis + stds, alpha=0.3, label="Confidence Interval")
    elif attr in ['cos_descent_ascent', 'progress_dir', 'ascent_semi_cos',
                  'ascent_step_diff', 'descent_step_diff', 'descent_norm',
                  'dominant_alignment', 'batch_loss', "fedlora_A_align", "fedlora_B_align",
                  "fedlora_A_cosine", "fedlora_B_cosine", "lora_A_norm", "lora_B_norm", "grad_norm", "truncate_err"]:
        if hasattr(train_graphs[0], attr):
            line = plot_y(ax=ax, yaxis=yaxis, name=attr, linestyle=linestyle)
            ax.fill_between(range(len(yaxis)), yaxis - stds, yaxis + stds, alpha=0.3, label="Confidence Interval")
        else:
            line = plot_y(ax=ax, yaxis=[], name=attr, linestyle=linestyle)
    else:
        raise NotImplementedError
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
            if attr == "accuracy":
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

def plot_figures_opts_attr(opts_list, model_params, opt_params, attrs, start=None, end=None, alpha=1.0, linewidth=1.0, legend_fontsize=None, ylabel=None, legends=[], titles=[], yaxis=[], xlabels=[], save_dir=None, return_last=False, return_max=False):
    #rows, cols = (len(attrs) - 1) // 6 + 1, min(len(attrs), 6)
    import matplotlib.ticker as mtick
    rows, cols = 1, len(opts_list)
    fig, axs = plt.subplots(rows,cols, figsize=(cols*4, rows*3))
    last_val = []
    max_val = []
    if len(opts_list) > 1:
        axs = axs.reshape(-1)
    else:
        axs = [axs]
    ax_ptr = 0

    if isinstance(attrs, str):
        attrs = [attrs for i in range(len(opts_list))]
    if isinstance(yaxis, str):
        yaxis = [yaxis for i in range(len(opts_list))]
    if xlabels == []:
        xlabels = ["Epoch" for i in range(len(opts_list))]

    for opts, legend, title, attr, xlabel in zip(opts_list, legends, titles, attrs, xlabels):
        last_val.append([])
        max_val.append([])
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
                max_val[-1].append(np.max(train_graphs.accuracy[start:end]))

            if 'train_err' == attr:
                line=plot_xy(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=1-1*np.array(train_graphs.accuracy[start:end]), name=title, alpha=alpha)
                axs[ax_ptr].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
                #axs[ax_ptr].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
                #plot_xlogy(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=1-np.array(train_graphs.accuracy[start:end]), name="Train Error")

            if 'test_loss' == attr:
                plot_test_loss(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=train_graphs.test_loss[start:end], xlabel=xlabel, title=title)

            if 'test_acc' == attr:
                line = plot_xy(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=np.array(train_graphs.test_accuracy[start:end]), name=title, alpha=alpha)
                axs[ax_ptr].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
                last_val[-1].append(line.get_data()[-1][-1])
                max_val[-1].append(np.max(train_graphs.test_accuracy[start:end]))
                #plot_test_acc(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=train_graphs.test_accuracy[start:end])
                

            if 'test_err' == attr:
                #print(1-1*np.array(train_graphs.test_accuracy[start:end]))
                line=plot_xy(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=1-1*np.array(train_graphs.test_accuracy[start:end]), name=title, alpha=alpha)
                axs[ax_ptr].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
                last_val[-1].append(line.get_data()[-1][-1])

            if 'truncate_err' == attr:
                cur_epochs = np.arange(len(train_graphs.truncate_err[start:end]))
                line=plot_xy(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=np.array(train_graphs.truncate_err[start:end]), name=title, alpha=alpha)
                #axs[ax_ptr].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
                #axs[ax_ptr].yaxis.set_ylabel("Truncation Error")
                last_val[-1].append(line.get_data()[-1][-1])

            if 'truncate_err_ratio' == attr:
                cur_epochs = np.arange(len(train_graphs.truncate_err_ratio[start:end]))
                line=plot_xy(ax=axs[ax_ptr], xaxis=cur_epochs, yaxis=np.array(train_graphs.truncate_err_ratio[start:end]), name=title, alpha=alpha)
                #axs[ax_ptr].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
                #axs[ax_ptr].yaxis.set_ylabel("Truncation Error Ratio")
                last_val[-1].append(line.get_data()[-1][-1])

        axs[ax_ptr].legend(legend, fontsize=legend_fontsize)
        ax_ptr += 1
    if yaxis == []:
        if ylabel:
            axs[0].set_ylabel(ylabel)
        else:
            axs[0].set_ylabel("Test Accuracy")
    else:
        for i in range(len(yaxis)):
            axs[i].set_ylabel(yaxis[i])
    #axs[0].set_ylim([0.01, 0.07])

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir)
    if return_last:
        return last_val
    if return_max:
        return max_val

def plot_figures_opts_attr_ci(opts_list, model_params, opt_params, attrs, start=None, end=None, alpha=1.0, linewidth=1.0, legend_fontsize=None, ylabel=None, legends=[], titles=[], yaxis=[], xlabels=[], save_dir=None, return_last=False, return_max=False, overlap=False, use_seaborn=False, linestyles=[]):
    import matplotlib.ticker as mtick
    rows, cols = 1, len(opts_list)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    last_val = []
    max_val = []
    if len(opts_list) > 1:
        axs = axs.reshape(-1)
    else:
        axs = [axs]
    ax_ptr = 0

    if isinstance(attrs, str):
        attrs = [attrs for _ in range(len(opts_list))]
    if isinstance(yaxis, str):
        yaxis = [yaxis for _ in range(len(opts_list))]
    if xlabels == []:
        xlabels = ["Epoch" for _ in range(len(opts_list))]
    if linestyles == []:
        linestyles = [[] for _ in range(len(opts_list))]

    if use_seaborn:
        import seaborn as sns
        import pandas as pd

    for opts, legend, title, attr, xlabel, linestyle in zip(opts_list, legends, titles, attrs, xlabels, linestyles):
        last_val.append([])
        max_val.append([])
        line_handles = []
        for opt_idx, opt_name in enumerate(opts):
            print("linestyle: ", linestyle)
            optlinestyle = linestyle[opt_idx] if opt_idx < len(linestyle) else 'solid'
            model_param = model_params[opt_name]
            directory = get_running_directory(
                opt_params[opt_name]['lr'],
                opt_params[opt_name]['dataset_name'],
                opt_params[opt_name]['loss'],
                opt_params[opt_name]['opt'],
                opt_params[opt_name]['model_name'],
                opt_params[opt_name]['momentum'],
                opt_params[opt_name]['weight_decay'],
                opt_params[opt_name]['batch_size'],
                opt_params[opt_name]['epochs'],
                **model_param
            )
            print(directory)
            run_dir = sorted(os.listdir(directory))
            train_graphs = []
            for run_id in run_dir:
                if 'run' not in run_id:
                    continue
                run_path = f'{directory}{run_id}/train_graphs.pk'
                if os.path.exists(run_path):
                    with open(run_path, 'rb') as f:
                        train_graphs.append(pickle.load(f))

            if attr in ['test_err', 'train_err', 'test_acc', 'acc']:
                axs[ax_ptr].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

            if use_seaborn:
                label = legend[opt_idx] if opt_idx < len(legend) else opt_name
                rows = []
                for tg in train_graphs:
                    vals = get_attr_from_graph(tg, attr)[start:end]
                    xaxis = tg.log_epochs[start:end]
                    for x, y in zip(xaxis, vals):
                        rows.append({'x': x, 'value': y, 'run': id(tg)})
                df = pd.DataFrame(rows)
                sns.lineplot(
                    data=df, x='x', y='value',
                    ax=axs[ax_ptr], label=label,
                    errorbar='sd', linewidth=linewidth,
                    alpha=0.5 if optlinestyle == 'dashed' else alpha,
                    linestyle=optlinestyle,
                    color='gray' if optlinestyle == 'dashed' else None,
                )
                line = axs[ax_ptr].lines[-1]
                line_handles.append(line)
                mean_vals = df.groupby('x')['value'].mean()
                last_val[-1].append(float(mean_vals.iloc[-1]))
                max_val[-1].append(float(mean_vals.max()))
            else:
                plot_fn = plot_attr_overlap if overlap else plot_attr
                line = plot_fn(ax=axs[ax_ptr], train_graphs=train_graphs, attr=attr, start=start, end=end, alpha=alpha, linewidth=linewidth, linestyle=optlinestyle)
                line_handles.append(line)
                ydata = line.get_ydata()
                last_val[-1].append(ydata[-1])
                max_val[-1].append(np.max(ydata))

        axs[ax_ptr].set_xlabel(xlabel)
        axs[ax_ptr].set_title(title)
        if use_seaborn:
            axs[ax_ptr].legend(fontsize=legend_fontsize)
        else:
            axs[ax_ptr].legend(line_handles, legend, fontsize=legend_fontsize)
        ax_ptr += 1

    if yaxis == []:
        if ylabel:
            axs[0].set_ylabel(ylabel)
        else:
            axs[0].set_ylabel("Test Accuracy")
    else:
        for i in range(len(yaxis)):
            axs[i].set_ylabel(yaxis[i])

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir)
    if return_last:
        return last_val
    if return_max:
        return max_val

def plot_figures_opts_attr_ci_beautify(opts_list, model_params, opt_params, attrs, start=None, end=None, alpha=0.9, linewidth=2.0, legend_fontsize=11, ylabel=None, legends=[], titles=[], yaxis=[], xlabels=[], save_dir=None, return_last=False, return_max=False, overlap=False, use_seaborn=True, linestyles=[]):
    """Beautified version of plot_figures_opts_attr_ci with improved aesthetics."""
    import matplotlib.ticker as mtick
    import seaborn as sns
    import pandas as pd

    # Colorblind-friendly palette (Wong 2011)
    PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000']

    rows, cols = 1, len(opts_list)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.8))
    last_val = []
    max_val = []
    if len(opts_list) > 1:
        axs = axs.reshape(-1)
    else:
        axs = [axs]
    ax_ptr = 0

    if isinstance(attrs, str):
        attrs = [attrs for _ in range(len(opts_list))]
    if isinstance(yaxis, str):
        yaxis = [yaxis for _ in range(len(opts_list))]
    if xlabels == []:
        xlabels = ["Epoch" for _ in range(len(opts_list))]
    if linestyles == []:
        linestyles = [[] for _ in range(len(opts_list))]

    for opts, legend, title, attr, xlabel, linestyle in zip(opts_list, legends, titles, attrs, xlabels, linestyles):
        last_val.append([])
        max_val.append([])
        ax = axs[ax_ptr]

        # Background and grid
        ax.set_facecolor('#f7f7f7')
        ax.grid(True, color='white', linewidth=1.2, linestyle='-', zorder=0)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#cccccc')
            ax.spines[spine].set_linewidth(0.8)

        if attr in ['test_err', 'train_err', 'test_acc', 'acc']:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

        line_handles = []
        for opt_idx, opt_name in enumerate(opts):
            optlinestyle = linestyle[opt_idx] if opt_idx < len(linestyle) else 'solid'
            color = PALETTE[opt_idx % len(PALETTE)]
            # dashed lines: same hue, reduced alpha
            opt_alpha = 0.55 if optlinestyle == 'dashed' else alpha

            model_param = model_params[opt_name]
            directory = get_running_directory(
                opt_params[opt_name]['lr'],
                opt_params[opt_name]['dataset_name'],
                opt_params[opt_name]['loss'],
                opt_params[opt_name]['opt'],
                opt_params[opt_name]['model_name'],
                opt_params[opt_name]['momentum'],
                opt_params[opt_name]['weight_decay'],
                opt_params[opt_name]['batch_size'],
                opt_params[opt_name]['epochs'],
                **model_param
            )
            print(directory)
            run_dir = sorted(os.listdir(directory))
            train_graphs = []
            for run_id in run_dir:
                if 'run' not in run_id:
                    continue
                run_path = f'{directory}{run_id}/train_graphs.pk'
                if os.path.exists(run_path):
                    with open(run_path, 'rb') as f:
                        train_graphs.append(pickle.load(f))

            if use_seaborn:
                label = legend[opt_idx] if opt_idx < len(legend) else opt_name
                df_rows = []
                for tg in train_graphs:
                    vals = get_attr_from_graph(tg, attr)[start:end]
                    xax = tg.log_epochs[start:end]
                    for x, y in zip(xax, vals):
                        df_rows.append({'x': x, 'value': y, 'run': id(tg)})
                df = pd.DataFrame(df_rows)
                sns.lineplot(
                    data=df, x='x', y='value',
                    ax=ax, label=label,
                    errorbar='sd', linewidth=linewidth,
                    alpha=opt_alpha, linestyle=optlinestyle, color=color,
                )
                line = ax.lines[-1]
                line_handles.append(line)
                mean_vals = df.groupby('x')['value'].mean()
                last_val[-1].append(float(mean_vals.iloc[-1]))
                max_val[-1].append(float(mean_vals.max()))
            else:
                plot_fn = plot_attr_overlap if overlap else plot_attr
                line = plot_fn(ax=ax, train_graphs=train_graphs, attr=attr, start=start, end=end,
                               alpha=opt_alpha, linewidth=linewidth, linestyle=optlinestyle)
                line.set_color(color)
                line_handles.append(line)
                ydata = line.get_ydata()
                last_val[-1].append(ydata[-1])
                max_val[-1].append(np.max(ydata))

        ax.set_xlabel(xlabel, fontsize=12, labelpad=4)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=6)
        ax.tick_params(axis='both', labelsize=10)
        if use_seaborn:
            ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='#cccccc')
        else:
            ax.legend(line_handles, legend, fontsize=legend_fontsize, framealpha=0.9, edgecolor='#cccccc')
        ax_ptr += 1

    if yaxis == []:
        ylabel_text = ylabel if ylabel else "Test Accuracy"
        axs[0].set_ylabel(ylabel_text, fontsize=12)
    else:
        for i in range(len(yaxis)):
            axs[i].set_ylabel(yaxis[i], fontsize=12)

    plt.tight_layout(pad=1.5)
    if save_dir:
        plt.savefig(save_dir, dpi=150, bbox_inches='tight')
    if return_last:
        return last_val
    if return_max:
        return max_val


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

def plot_eigen_density(model_param, opt_param, start=None, end=None, step=None, save_dir=None, transparent=False):
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
    with open(f'{directory}train_graphs.pk', 'rb') as f:
        eval_graphs = pickle.load(f)
    
    cur_epochs = eval_graphs.log_epochs
    density_eigens, density_weights = eval_graphs.density_eigen, eval_graphs.density_weight
    density_eigens, density_weights, epochs = density_eigens[start:end:step], density_weights[start:end:step], cur_epochs[start:end:step]
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
        plt.savefig(save_dir, transparent=transparent)

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