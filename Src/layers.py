import numpy as np
import cv2
import torch
import torch.nn.functional as f
import pandas as pd
import copy
from pathlib import Path

class Handler:
    def __init__(self, hooks, hook_id):
        self.hooks = hooks
        self.hook_id = hook_id
        
    def remove(self):
        del self.hooks[self.hook_id]

class Linear:
    def __init__(self, in_dim , out_dim , bias = True, generator = None):
        self.W = torch.randn((in_dim , out_dim), generator = generator)*(2/in_dim)**0.5
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
    
    def state_dict(self):
        return {'weight':self.W} | ({'bias':self.b} if self.b is not None else {})

    def parameters(self):
        return [self.W] + ([self.b] if self.b is not None else [])

    def register_forward_hook(self, hook_fn):
        handler = Handler(self._forward_hooks, self._hook_id)
        self._forward_hooks[self._hook_id] = hook_fn
        self._hook_id += 1
        return handler

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
        return []

    def state_dict(self):
        return {}
        
    # this function responsible for receiving hooks and assign them
    # to that layer and return a handle if you determine todelete 
    # the hook later
    def register_forward_hook(self,hook_fn):
        handler = Handler(self._forward_hooks, self._hook_id)
        self._forward_hooks[self._hook_id] =hook_fn
        self._hook_id+=1
        return handler

#-------------------------------------------------------------------
class Flatten:
    def __call__(self,X):
        self.out = X.view(X.shape[0],-1)
        return self.out
    def parameters(self):
        return []

    def state_dict(self):
        return {}

#--------------------------------------------------------------------
class Softmax:      
    def __call__(self,logits):
        self.out = f.softmax(logits,dim=1)
        return self.out
        
    def parameters(self):
        return []

    def state_dict(self):
        return {}
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
        return []

    def state_dict(self):
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
        self.RAM = torch.zeros((1,channel_num,1,1))  # running average mean
        self.RAS = torch.zeros((1,channel_num,1,1))  # running average standared deviation
        self._forward_hooks = {}
        self._hook_id = 0
        
    def __call__(self,X):
            m = X.shape[0]
            #dim = [c for c in range(len(X.shape))]
            if self.state:
                self.mean = X.mean(dim = (0,2,3), keepdim = True)
                self.std = X.std(dim = (0,2,3), keepdim = True)
                with torch.no_grad():
                    self.RAM.copy_(self.B * self.RAM + (1-self.B) * self.mean)
                    self.RAS.copy_(self.B * self.RAS + (1-self.B) * self.std)
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

    def state_dict(self):
        return {'weight':self.gama , 'bias':self.beta , 'running_mean' :self.RAM ,'running_var' :self.RAS}

    def parameters(self):
        return [self.gama, self.beta]

    def register_forward_hook(self, hook_fn):
        handle_id = self._hook_id
        self._forward_hooks[self._hook_id] = hook_fn
        self._hook_id += 1
        return handle_id

