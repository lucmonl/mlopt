"""
from scipy.stats import levy_stable
from plot_functions import get_attr, plot_figures_opts_attrs
from main import graphs
import matplotlib.pyplot as plt
import numpy as np

lr, dataset_name, loss_name, momentum, weight_decay, batch_size, smooth, epochs = 1.0, "cifar", 'CrossEntropyLoss', 0.0, 1e-4, 128, 0.1, 200
model_name, depth = "resnet_fixup", 18
client_opt, client_lr, client_momentum, client_num, client_epoch, sketch_size, scheduler, lr_min  ="sgd", 0.1, 0.0, 80, 1, 80000, "cosine", 1e-5
model_params, opt_params = {}, {}
model_params["f-sgd-niid"] = {"depth": depth, "non_iid": 0.8, 'server_opt': 'sgd', 'client_opt': client_opt, 'client_lr': client_lr, 'client_momentum': client_momentum, "client_num": client_num, 'client_epoch': client_epoch, "sketch_size": -1}
opt_params['f-sgd-niid'] = {'lr': lr, 'dataset_name':dataset_name, 'loss': loss_name, 'opt':'federated', 'model_name':model_name, 'momentum':momentum, 'weight_decay':weight_decay, 'batch_size':batch_size, 'epochs':epochs}

#plot_figures_opts_attrs(["f-sgd-niid", "f-clip-niid"], model_params, opt_params, attrs=["loss", "acc", "test_loss", "test_err"])
#grad_norm = get_attr("f-sgd", model_params, opt_params, "grad_norm")
#plt.hist(np.array(grad_norm)[-1].reshape(-1))
#plt.show()
plt.figure(figsize=(2.5,2))
grad_norm = get_attr("f-sgd-niid", model_params, opt_params, "grad_norm")

data_points = np.array(grad_norm)[:2400].reshape(-1)
print("start to fit")
#alpha, beta, loc, scale = levy_stable.fit(data_points)
alpha, beta, loc, scale = 1.5137136049717719, 0.9999893138626956, 2.3271825648881856, 0.54501460768667
print("finish fitting")
pdf_levy = [levy_stable.pdf(x, alpha, beta, loc, scale) for x in np.arange(0,14,0.1)]
plt.figure(figsize=(2.5,2))
plt.hist(np.array(grad_norm)[:2400].reshape(-1), bins=40, density=True)
plt.plot(np.arange(0,14,0.1), pdf_levy)
plt.xlabel("gradient norm")
plt.ylabel("density")
plt.tight_layout()
plt.savefig("../plots/neurips24/heavy_tail.pdf")
print(alpha, beta, loc, scale)
"""

from plot_functions import get_attr, plot_figures_opts_attrs
from main import graphs
import matplotlib.pyplot as plt
import numpy as np

lr, dataset_name, loss_name, momentum, weight_decay, batch_size, smooth, epochs = 1.0, "cifar", 'CrossEntropyLoss', 0.0, 1e-4, 8, 0.1, 200
model_name, depth = "resnet_fixup", 18
client_opt, client_lr, client_momentum, client_num, client_epoch, sketch_size, scheduler, lr_min  ="sgd", 0.1, 0.0, 80, 1, 80000, "cosine", 1e-5
model_params, opt_params = {}, {}
model_params["f-sgd-niid"] = {"depth": depth, "non_iid": 0.6, 'server_opt': 'sgd', 'client_opt': client_opt, 'client_lr': client_lr, 'client_momentum': client_momentum, 'client_weight_decay': 0.0, "client_num": client_num, 'client_epoch': client_epoch, "sketch_size": -1, "scheduler": scheduler, "lr_min": lr_min}
opt_params['f-sgd-niid'] = {'lr': lr, 'dataset_name':dataset_name, 'loss': loss_name, 'opt':'federated', 'model_name':model_name, 'momentum':momentum, 'weight_decay':weight_decay, 'batch_size':batch_size, 'epochs':epochs}

grad_norm = get_attr("f-sgd-niid", model_params, opt_params, "minibatch_grad_norm")
#plt.hist(np.array(grad_norm).reshape(-1))
#plt.show()
print(len(grad_norm))

heavy_ind = [500, 700, 1000, 3400, 4400, 4500, 5000, 5200, 5500,6500, 6700, 9000, 9500, 10200, 10500]

from scipy.stats import levy_stable
print("start to fit")
rows, cols=3, 5
plt_i = 0
fig, axs = plt.subplots(rows,cols, figsize=(cols*2.5, rows*2))
axs = axs.reshape(-1)
for i in heavy_ind:
    data_points = np.array(grad_norm[i]).reshape(-1)
    alpha, beta, loc, scale = levy_stable.fit(data_points)
    pdf_levy = [levy_stable.pdf(x, alpha, beta, loc, scale) for x in np.arange(np.min(grad_norm[i]),np.max(grad_norm[i]),0.1)]
    axs[plt_i].hist(np.array(grad_norm[i]), bins=20, density=True)
    axs[plt_i].plot(np.arange(np.min(grad_norm[i]),np.max(grad_norm[i]),0.1), pdf_levy)
    axs[plt_i].set_title(r"$\alpha=$"+str(alpha.round(3)))
    axs[plt_i].set_xlabel("gradient norm")
    if plt_i % cols == 0:
        axs[plt_i].set_ylabel("density")
    plt_i += 1
plt.tight_layout()
print("finish fitting")
plt.savefig("../plots/iclr25/non_iid_alpha.pdf")