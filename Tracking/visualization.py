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
