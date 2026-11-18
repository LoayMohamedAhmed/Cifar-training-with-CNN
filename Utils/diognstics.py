from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import torch
g = torch.Generator().manual_seed(2882001)
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

#---------------------------------------------------------------------------------------
def gradient_ascent(model, layer_idx,channel_idx,LR = 0.05,steps = 150, tv_weight=1e-2):
    # freeze the weights as we only ant the gradient of output ith respect to input not the weights
    # but when we finish we restore the original state of the model
    model.eval()
    orig_requires_grad = [p.requires_grad for p in model.parameters().values()]
    device = list(model.parameters().values())[0].device
    
    activation_out = {}
    def hook(module, input, output):
        activation_out['activation'] = output
    handler = model.layers[layer_idx].register_forward_hook(hook)
    
    for param in model.parameters().values():
        param.requires_grad = False
        
    # 1. Start from low-variance gray noise instead of loud std=1 noise
    pattern = torch.randn((1,3,32,32), generator= g, device = device, requires_grad= True) *0.005
    try:
        for i in tqdm(range(steps),desc = 'gradient_iterations'):
            # 2. Random spatial jitter (±2 pixels) to suppress high-frequency static
            ox, oy = torch.randint(-2, 3, (2,)).tolist()
            jittered = torch.roll(pattern, shifts=(ox, oy), dims=(2, 3))
            model(jittered)
            activation = activation_out['activation'][0,channel_idx].mean()
            # 3. Total Variation penalty: discourages neighboring pixels from having wildly different colors
            tv_loss = (
                ((pattern[:, :, 1:, :] - pattern[:, :, :-1, :]) ** 2).mean() +
                ((pattern[:, :, :, 1:] - pattern[:, :, :, :-1]) ** 2).mean()
            )
            # Maximize activation while minimizing TV noise
            objective = activation - tv_weight * tv_loss
            pattern.grad = None
            output.backward()
            with torch.no_grad():
                # normalize the grad to so the step size still robust 
                grad = pattern.grad / (pattern.grad.std() + 1e-8)
                pattern += LR * grad
                # add some regularization to prevent the input from exploding to infinity
                pattern *= 0.99
                
    finally:
        del model.layers[layer_idx]._forward_hooks[handler]
        for param, grad_state in zip(model.parameters().values(), orig_requires_grad):
            param.requires_grad = grad_state
    return pattern

#-------------------------------------------------------------------------------------------------
def senity_check(data,tr_engine):
    losses  = []
    for i in range(len(data['images'])):
        loss, logits = tr_engine.step(data['images'] , data['classes'])
        losses.append(loss)
    plt.plot(losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('senity check')
    plt.show()
