import torch
import numpy as np

class module:
  def __init__(self):
    self.train_state = True
    self._modules = {}

  def train(self,mode = True):
    # traverse recursively over the modules in the tree to set their state
    def module_set_state(module):
      for child_module in module._modules.values():
        module_set_state(child_module)
      module.train_state = mode
    module_set_state(self)

  def __setattr__(self , name , value):
    if isinstance(value , module):
      self.add_modules(name,value)
    # you should call the parent method otherwise the attributes will not be saved
    object.__setattr__(self, name, value)

  def add_module(self, name, value):
    self._modules[name] = value
    
  def eval(self):
    self.train(False)

  def enable_grad(self):
    for p in self.parameters():
      p.requires_grad = True
