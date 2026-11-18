import numpy as np
from matplotlib import pyplot as plt
# hook that will catptuer the percentage of dead ReLUs in each layer
def dead_percentage_hook(name, dead_percent):
    def hook(module, input, output):
        with torch.no_grad():
            percentage = (output == 0).float().mean().item() *100
            dead_percent[name] = percentage
    return hook

def plot_gradient_flow(model , grad_norm , iterations_per_epochs, title,x_label = '', y_label = ''):
    grad_norm_array = np.array(grad_norm)
    parameters_num = grad_norm_array.shape[0]
    epochs_num = grad_norm_array.shape[1] // iterations_per_epochs

    # truncate the gradient based on the number of iterartions per epoch to ignore the incomplete epochs where the number of iterations is smaller than the previouse epochs
    # i do this for dynamic sampler but as i train CIFAR, it's a fixed size dataset so the number of iterations  per epoch is constant 
    valid_length = iterations_per_epochs * epochs_num
    grad_norm_array = grad_norm_array[:, :valid_length]
    grad_norm_array = grad_norm_array.reshape(-1,epochs_num, iterations_per_epochs).mean(axis = 2)

    layer_id = 1
    param_id = 0
    colors = plt.cm.tab20.colors
    
    for layer_idx, layer in enumerate(model.layers):
        params = layer.parameters()
    
        if len(params) == 0:
            continue
    
        color = colors[layer_id % len(colors)]
    
        for i, _ in enumerate(params):
            # first curve gets the legend label
            plt.plot(
                grad_norm_array[param_id],
                color=color,
                label=f"{layer.__class__.__name__}{layer_id}" if i==0 else None
                )
            param_id += 1
        layer_id += 1

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
