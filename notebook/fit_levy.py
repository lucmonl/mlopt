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