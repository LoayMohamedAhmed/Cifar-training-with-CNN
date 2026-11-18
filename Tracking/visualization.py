import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

def visualize_image(image_tensor , title):
    # to seprate the tensor from the computational graph, move it into cpu and finally remove the channel 0
    image_tensor = image_tensor.detach().cpu().squeeze(0) 
    # normalize the input iamge as the gradient ascent doesn't know what's the valid image looks like
    # it may result in negative values.
    image_tensor -= image_tensor.min()
    image_tensor /= image_tensor.max()
    image_tensor = image_tensor.clamp(0,1) # to ensure the range is between 0 and 1
    image_tensor = image_tensor.permute(1,2,0)
    plt.imshow(image_tensor)
    plt.title(title)

#-------------------------------------------------------------------------------------------
def display_layer_out(layer_idx , model , image):
    
    # construct hook to capture the target layer output
    out = {}
    def hook(module,input, output):
        out['target'] = output
    target_layer = model.layers[layer_idx]
    hook_handel = target_layer.register_forward_hook(hook)
    with torch.no_grad():
        model(image)
    images = out['target'][0].detach().cpu()
    
    # delet hook
    del target_layer._forward_hooks[hook_handel]
    
    # normalize the images to be within correct range
    mins = torch.amin(images, dim = (-1,-2), keepdim=True)
    maxs = torch.amax(images, dim = (-1,-2), keepdim=True)
    images = (images - mins)/(maxs-mins + 1e-8) # dim (C,H,W)
    images = images.unsqueeze(1) #to add a new dim which is the channel dim, new dim (C,1,H,W)
    # make_grid expect the image to have 4 dimentions so we add extra dimention at channel 1
    # grid will replicate the single channel we add to conver it to RGB
    grid = make_grid(images, nrow = 16) 
    
    plt.figure(figsize=(16,8))
    plt.imshow(grid[0], cmap= 'viridis')
    plt.axis("off")
    plt.title(f'{model.module_names[id(target_layer)]}\n'
             f"activation shape {tuple(out['target'].shape)}"
             )
    plt.show()
