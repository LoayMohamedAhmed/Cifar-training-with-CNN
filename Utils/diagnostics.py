from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import torch
from Src.layers import ReLU

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
def gradient_ascent(model, layer_idx,channel_idx,LR = 0.05,steps = 150, tv_weight=1e-2, generator= None):
    # freeze the weights as we only ant the gradient of output ith respect to input not the weights
    # but when we finish we restore the original state of the model
    model.eval()
    orig_requires_grad = [p.requires_grad for p in model.parameters()]
    device = list(model.parameters())[0].device
    
    activation_out = {}
    def hook(module, input, output):
        activation_out['activation'] = output
    handler = model.layers[layer_idx].register_forward_hook(hook)
    
    for param in model.parameters():
        param.requires_grad = False
        
    # 1. Start from low-variance gray noise instead of loud std=1 noise
    pattern = (torch.randn((1,3,32,32), generator= generator, device = device) *0.005).requires_grad_(True)
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
            objective.backward()
            with torch.no_grad():
                # normalize the grad to so the step size still robust 
                grad = pattern.grad / (pattern.grad.std() + 1e-8)
                pattern += LR * grad
                # add some regularization to prevent the input from exploding to infinity
                pattern *= 0.99
                
    finally:
        handler.remove()
        for param, grad_state in zip(model.parameters(), orig_requires_grad):
            param.requires_grad = grad_state
    return pattern

#-------------------------------------------------------------------------------------------------
def sanity_check(data,tr_engine):
    losses  = []
    for i in range(len(data['images'])):
        loss, logits = tr_engine.step(data['images'] , data['classes'])
        losses.append(loss)
    plt.plot(losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('senity check')
    plt.show()
#---------------------------------------------------------------------------------------------------
def grad_cam(model, layer_idx, img_ , cls_idx ):
    model.eval()
    activation = {}
    target_size = tuple(img_.shape[-2:])
    def hook(module,input,output):
        activation['target'] = output
        activation['target'].retain_grad()
    handler = model.layers[layer_idx].register_forward_hook(hook)
    output = model(img_)[0,cls_idx]
    activation['target'].grad = None
    output.backward()
    handler.remove()
    # we multiply each channel by its mean so that each one contribute according to its importance
    # in grad cam we want to visualize the the importance of the feature with respect to specific class
    # so we only care about the featues that increase the class score so we ignore negative gradients
    Relu = ReLU()
    weights = activation['target'].grad.mean(dim=(2,3), keepdim = True) #shape(1,C,1,1)
    heat_map = weights * activation['target']
    heat_map = Relu(heat_map.sum(dim = 1)) #shape(1,W,H)
    heat_map_resized = torch.nn.functional.interpolate(heat_map.unsqueeze(1), size = target_size, mode = 'bilinear', align_corners = False).squeeze()
    
    # normalize the heat map to ensure values within the alid range for an image
    max = torch.amax(heat_map_resized, dim = (0,1), keepdim = True)
    min = torch.amin(heat_map_resized, dim = (0,1), keepdim = True)
    heat_map_resized -=  min
    heat_map_resized /= ((max - min) +1e-8)
    heat_map_resized = heat_map_resized.cpu().detach().numpy()
    img_result = img_.squeeze(0).permute(1,2,0).cpu().detach().numpy()
    # plot the heatmap over image
    plt.imshow(img_result)
    plt.imshow(heat_map_resized , cmap = 'jet' , alpha = 0.3)
    plt.axis('off')
    plt.colorbar()
    plt.show()
    
