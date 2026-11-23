import torch
import numpy as np
import copy
from pathlib import Path
import pandas as pd
from Src.layers import BatchNorm2d, Dropout
from torch.nn import functional as f
from Src.Module import module

class Sequential(module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        #self.state = 1
        self.module_names = {}
        # method inherited from module to enable gradient for parameters 
        self.enable_grad()
        self.layer_names()
        best_param = copy.deepcopy(self.parameters())
        # assign layers inside sequential as children of it
        for i, layer in enumerate(self.layers):
            self.add_module(str(i) , layer)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self,param):
        if isinstance(param,(str,Path)):
            # map location tells pytorch to load into the CPU even if the data saved from GPU
            param = torch.load(param, map_location = 'cpu')
        current_param = self.state_dict()
        with torch.no_grad():
            for key , tensor in param.items():
                current_param[key].copy_(tensor)

    def layer_names(self):
        layers = {}
        for layer in self.layers:
            name = layer.__class__.__name__
            if name in layers.keys():
                layers[name]+=1
            else:
                layers[name] = 1
            self.module_names[id(layer)] = name +' '+ str(layers[name])
                
    def train(self):
        self.state = 1
        
        
    def __call__(self,X):
        self.out = X
        for layer in self.layers:
            if not self.state and isinstance(layer, (Dropout, BatchNorm2d )):
               layer.state = 0
            self.out = layer(self.out)
        return self.out  
        
    def state_dict(self):
        # this function assign each layer weight with a layer index+layer weight
        # 0.W , 0.0.W, 0.0.b for hierachical structure 
        param = {}
        def param_discover(module,prefix = ""):
            if isinstance(module , Sequential):
                for i , layer in enumerate(module.layers):
                    param_discover(layer,f"{prefix}{i}.")
            else:
                for k, val in module.state_dict().items():
                    param[f"{prefix}{k}"] = val 
                    
        param_discover(self)          
        return param

    def parameters(self):
        # this function assign each layer weight with a layer index+layer weight
        # 0.W , 0.0.W, 0.0.b for hierachical structure 
        params = []
        def param_discover(module):
            if isinstance(module , Sequential):
                for i , layer in enumerate(module.layers):
                    param_discover(layer)
            else:
                for param in module.parameters():
                    params.append(param) 
                    
        param_discover(self)          
        return params
    
    def summary(self,input_size):
        total_param = 0
        summary = pd.DataFrame({'Layer_type':[],'output shape':[],'#parameters':[]})
        X = torch.rand((1,*input_size))
        self.eval()
        for layer in self.layers:
            with torch.no_grad():
                if isinstance(layer,BatchNorm2d):
                    layer.state = 0
                X = layer(X)
            no_param = int(np.sum([p.numel() for p in layer.parameters()]))
            layer_type = self.module_names[id(layer)]
            dim = ' , '.join([str(X.shape[i]) for i in range(1,len(X.shape))])
            out_shape = '(nan , ' + f'{dim}' +' )'
            summary.loc[len(summary)] = [ layer_type ,out_shape,no_param]
            total_param+= no_param
        self.train()
        display(summary)
        print( f"total number of parameters: {total_param}")

    def total_loss(self, data):
        total_loss = 0
        total_images = 0
        with torch.no_grad():
            for images,classes in data:
                logits = self.__call__(images)
                loss = f.cross_entropy(logits,classes,reduction='sum')
                total_loss+=loss.item()
                total_images += classes.shape[0]
            total_loss/= total_images
        return total_loss

    def EarlyStopping(self, patience_per):
        best_val_loss = np.inf
        stop = False
        remaining_period = patience_per
        def early_stopping(current_val_loss):
            # in nested function if i don't set nonlocal identifire berfore the nested variable
            # it will treat them as local ones
            nonlocal best_val_loss, stop, remaining_period
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                self.best_param = copy.deepcopy(self.parameters())
                remaining_period = patience_per
            else:
                remaining_period -= 1
                if remaining_period == 0:
                    stop = True
            return stop, best_val_loss
        return early_stopping
