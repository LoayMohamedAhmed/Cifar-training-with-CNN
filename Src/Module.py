import torch
import numby

class module:
  def __init__(self):
    self.train_state = True

  def train(self,mode = True):
    self.train_state = mode
    
  def eval(self):
    self.train(False)
