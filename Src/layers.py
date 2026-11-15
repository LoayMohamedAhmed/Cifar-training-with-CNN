import numpy as np
import cv2
import torch
import torch.nn.functional as f
import pandas as pd
import copy
g = torch.Generator().manual_seed(2882001)

class Linear:
    def __init__(self, in_dim , out_dim , bias = True):
        self.W = torch.randn((in_dim , out_dim), generator = g)*(2/in_dim)**0.5
        self.b = torch.zeros(out_dim) if bias else None
        self._hook_id = 0
        self._forward_hooks ={}

    def __call__(self,X):
        self.out = X @ self.W
        if self.b is not None:
            self.out+= self.b

        for hook_id , hook_fn in self._forward_hooks.items():
            hook_result = hook_fn(self, X, self.out)
            if hook_result is not None:
                self.out = hook_result
        return self.out
    
    def parameters(self):
        return {'W':self.W} | ({'b':self.b} if self.b is not None else {})

    def register_forward_hook(self, hook_fn):
        handle_id = self._hook_id
        self._forward_hooks[self._hook_id] = hook_fn
        self._hook_id += 1
        return handle_id

#--------------------------------------------------------------------------------------------

class ReLU:
    def __init__(self):
        # construct a dic that hold all hooks applied to that layer
        self._forward_hooks = {}
        self._hook_id = 0
        
    def __call__(self,Z):
        #self.out = torch.maximum(torch.tensor(0.0),Z)
        self.out = torch.maximum(0.001*Z,Z)
        # loop over each hook and execute it
        for hook_id , hook_fn in self._forward_hooks.items():
            hook_result = hook_fn(self, Z, self.out)
            # overwrite the output if the hook was used to alter the output
            if hook_result is not None:
                self.out = hook_result
        return self.out
        
    def parameters(self):
        return {}
        
    # this function responsible for receiving hooks and assign them
    # to that layer and return a handle if you determine todelete 
    # the hook later
    def register_forward_hook(self,hook_fn):
        handle_id = self._hook_id
        self._forward_hooks[self._hook_id] =hook_fn
        self._hook_id+=1
        return handle_id

#-------------------------------------------------------------------
class Flatten:
    def __call__(self,X):
        self.out = X.view(X.shape[0],-1)
        return self.out
    def parameters(self):
        return []

#--------------------------------------------------------------------
class Softmax:
    def __init__(self):
        # construct a dic that hold all hooks applied to that layer
        self._forward_hooks = {}
        self._hook_id = 0
        
    def __call__(self,logits):
        self.out = f.softmax(logits,dim=1)
        for hook_id , hook_fn in self._forward_hooks.items():
            hook_result = hook_fn(self, X, self.out)
            if hook_result is not None:
                self.out = hook_result
        return self.out
        
    def parameters(self):
        return {}

    def register_forward_hook(self, hook_fn):
        handle_id = self._hook_id
        self._forward_hooks[self._hook_id] = hook_fn
        self._hook_id += 1
        return handle_id
#------------------------------------------------------------------
class Dropout:
    def __init__(self,dr_rate):
        self.rate = dr_rate
        self.state = 1
        self._forward_hooks = {}
        self._hook_id = 0

    def __call__(self , X):
        if self.state:
            self.mask = torch.rand_like(X)
            self.out = X * self.mask/(1-self.rate)
        else:
            self.out = X
            self.state = 1

        for hook_id , hook_fn in self._forward_hooks.items():
            hook_result = hook_fn(self, X, self.out)
            if hook_result is not None:
                self.out = hook_result
        return self.out
    
    def parameters(self):
        return {}

    def register_forward_hook(self, hook_fn):
        handle_id = self._hook_id
        self._forward_hooks[self._hook_id] = hook_fn
        self._hook_id += 1
        return handle_id

#------------------------------------------------------------------------------
class BatchNorm2d:
    def __init__(self,channel_num,eps =1e-8,momentum = 0.999):
        self.state = 1
        self.B = momentum
        self.gama = torch.ones((1,channel_num,1,1))
        self.beta = torch.zeros((1,channel_num,1,1))
        self.eps = eps
        self.RAM = 0  # running average mean
        self.RAS = 0  # running average standared deviation
        self._forward_hooks = {}
        self._hook_id = 0
        
    def __call__(self,X):
            m = X.shape[0]
            #dim = [c for c in range(len(X.shape))]
            if self.state:
                self.mean = X.mean(dim = (0,2,3), keepdim = True)
                self.std = X.std(dim = (0,2,3), keepdim = True)
                with torch.no_grad():
                    self.RAM = self.B * self.RAM + (1-self.B) * self.mean
                    self.RAS = self.B * self.RAS + (1-self.B) * self.std
            else:
                self.mean = self.RAM
                self.std = self.RAS
                self.state = 1
            self.normalized = (X-self.mean)/(self.std + self.eps)
            self.out = self.normalized * self.gama + self.beta
            for hook_id , hook_fn in self._forward_hooks.items():
                hook_result = hook_fn(self, X, self.out)
                if hook_result is not None:
                    self.out = hook_result
            return self.out

    def parameters(self):
        return {'gama':self.gama , 'beta':self.beta}

    def register_forward_hook(self, hook_fn):
        handle_id = self._hook_id
        self._forward_hooks[self._hook_id] = hook_fn
        self._hook_id += 1
        return handle_id

#---------------------------------------------------------------------------------
class Sequential:
    def __init__(self, *layers):
        self.layers = lsit(layers)
        self.state = 1
        self.module_names = {}
        for p in self.parameters().values():
            p.requires_grad = True
        self.layer_names()
        best_param = copy.deepcopy(self.parameters())

    def load(self,param):
        if isinstance(param,(str,Path)):
            # map location tells pytorch to load into the CPU even if the data saved from GPU
            param = torch.load(param, map_location = 'cpu')
        current_param = self.parameters()
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
        
    def eval(self):
        self.state = 0
        
    def __call__(self,X):
        self.out = X
        for layer in self.layers:
            if not self.state and isinstance(layer, (Dropout, BatchNorm2d )):
               layer.state = 0
            self.out = layer(self.out)
        return self.out  
        
    def parameters(self):
        # this function assign each layer weight with a layer index+layer weight
        # 0.W , 00.W, 00.b for hierachical structure 
        param = {}
        def param_discover(module,prefix = ""):
            if isinstance(module , Sequential):
                for i , layer in enumerate(module.layers):
                    param_discover(layer,f"{prefix}{i}.")
            else:
                for k, val in module.parameters().items():
                    param[f"{prefix}{k}"] = val 
        param_discover(self)          
        return param
    
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

    def EarlyStopping(self, patience_per,val_data):
        best_val_loss = np.inf
        stop = False
        remaining_period = patience_per
        def early_stopping():
            # in nested function if i don't set nonlocal identifire berfore the nested variable
            # it will treat them as local ones
            nonlocal best_val_loss, stop, remaining_period
            current_val_loss = self.total_loss(val_data)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                self.best_param = copy.deepcopy(self.parameters())
                remaining_period = patience_per
            else:
                remaining_period -= 1
                if remaining_period == 0:
                    stop = True
            return stop, current_val_loss, best_val_loss
        return early_stopping
