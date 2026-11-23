import numpy as np
import cv2
import torch
import torch.nn.functional as f
import pandas as pd
import copy
from Src.layers import Handler


class Conv2d:
    def __init__(self, num_filters, filter_size, in_channels ,padding = 0, stride = 1, bias = True, generator = None):
        if isinstance(filter_size, int):# kernel width and hight
            self.Kh = self.Kw = filter_size
        else:
            self.Kh, self.Kw = filter_size 
        self.Kd = in_channels #kernel depth
        self.pad, self.str = padding, stride
        self.fan_in = self.Kw * self.Kh * self.Kd
        self.W = torch.randn((num_filters , self.fan_in), generator= generator)*(2/self.fan_in)**0.5 #(f,C*h*W)
        self.b = torch.zeros((num_filters,1)) if bias else None
        self._forward_hooks = {}
        self._hook_id = 0
        
    
    def __call__(self,X):
        GEMM,out_H,out_W = self.im2col(X)
        out_flat = self.W @ GEMM
        if self.b is not None:
            out_flat += self.b
            
        out = out_flat.reshape(self.W.shape[0],X.shape[0],out_H,out_W).permute(1,0,2,3) # shape(N,cout,H,W)
        for hook_id , hook_fn in self._forward_hooks.items():
            hook_result = hook_fn(self, X, out)
            # overwrite the output if the hook was used to alter the output
            if hook_result is not None:
                out = hook_result
        return out

    def register_forward_hook(self,hook_fn):
        handler = Handler(self._forward_hooks, self._hook_id)
        self._forward_hooks[self._hook_id] =hook_fn
        self._hook_id+=1
        return handler

    def state_dict(self):
        return {'Weight':self.W} | ({'bias':self.b} if self.b is not None else {})

    def parameters(self):
        return [self.W] + ([self.b] if self.b is not None else [])

    def im2col(self,X):
        pad = (self.pad,)*4
        X_pad = f.pad(X, pad, value = 0)
        N,C,H,W = X_pad.shape
        out_H = (H - self.Kh)//self.str +1
        out_W = (W - self.Kw)//self.str +1
        Device = X.device
        # construct the offset of the kernel locally
        # for 3*3 offset is i = [0,0,0, 1,1,1, 2,2,2] j = [0,1,2, 0,1,2, 0,1,2]
        #repeat_interleave is the same as repeat in numpy
        i_offset = torch.arange(self.Kh, device= Device).repeat_interleave(self.Kw) # shape of (patch size)
        j_offset = torch.arange(self.Kw, device= Device).repeat(self.Kh) # repeat is the same as tile in numpy

        # generate the starting point for each patch
        i_start = torch.arange(out_H, device= Device).repeat_interleave(out_W) * self.str # shape of (out_H * out_W) = number of patches
        j_start = torch.arange(out_W, device= Device).repeat(out_H) * self.str

        # generate the indices for each patch
        i_indices = i_start.reshape(-1,1) + i_offset.reshape(1,-1) # (1,number of patches,patch size)
        j_indices = j_start.reshape(-1,1) + j_offset.reshape(1,-1)

        # extract the patches from the data
        # S is the patch size, p is the number of patches, C number of channels, N is the batch size
        col = X_pad[:,:,i_indices,j_indices] # shape (N,C,P,S)
        col = col.permute(1,3,0,2) # shape(C,S,N,P)
        # construct GEMM "general matrix multiplication"
        # Using .contiguous() is sometimes required after .permute() before .reshape()
        # to ensure that the data stored in conteguous blocks
        col =  col.contiguous().reshape(C * self.Kw * self.Kh, -1) #to have shape (C*S,N*P)
        return col,out_H, out_W

# ---------------------------------------------------------------------------------------------------------
class Pooling2d:
    def __init__(self, kernel_size ,pool_type, stride = None, padding = 0):
        if isinstance(kernel_size, int):
            self.Kh = self.Kw = kernel_size
        else:
            self.Kh, self.Kw = kernel_size
        self.ptype = pool_type
        self.str = stride if stride is not None else kernel_size
        self.pad = padding

    def state_dict(self):
        return {}

    def parameters(self):
        return {}

    def __call__(self,X):
        if self.ptype == 'GAP':
            return X.mean(dim = (-2,-1))
        N,C,H,W = X.shape
        X_pad = f.pad(X,(self.pad,)*4,value = 0)
        out_H = (H + 2*self.pad - self.Kh)//self.str + 1
        out_W = (W + 2*self.pad - self.Kw)//self.str + 1
        device = X.device
        i_offset = torch.arange(self.Kh, device= device).repeat_interleave(self.Kw)
        j_offset = torch.arange(self.Kw, device= device).repeat(self.Kh)

        i_start = torch.arange(out_H, device= device).repeat_interleave(out_W) *self.str
        j_start = torch.arange(out_W, device= device).repeat(out_H) * self.str

        i_indices = i_start.reshape(-1,1) + i_offset.reshape(1,-1)
        j_indices = j_start.reshape(-1,1) + j_offset.reshape(1,-1)

        col = X_pad[:, : , i_indices,j_indices] # shape(N,C,P,S)
        if self.ptype == 'MAX':
            out = (col.max(dim =-1))[0].reshape(N,C,out_H,out_W)
        elif self.ptype == 'AVG':
            out = (col.mean(dim = -1)).reshape(N,C,out_H,out_W)
        return out
