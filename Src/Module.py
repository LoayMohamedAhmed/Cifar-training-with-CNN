import torch
import numpy as np

class module:
  def __init__(self):
    self.train_state = True

  def train(self,mode = True):
    self.train_state = mode
    
  def eval(self):
    self.train(False)

  def enable_grad(self):
    for p in self.parameters():
            p.requires_grad = True
