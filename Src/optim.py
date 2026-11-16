import torch
import math
import numpy as np
from torch.nn import functional as f
from Src.layers import Softmax
class Adams:
    def __init__(params, B1=0.9 , B2 = 0.999, LR = 1e-3  ,epsilon = 1e-7 ):
        self.LR , self.B1 , self.B2, self.eps =  LR, B1, B2 , epsilon
        self.state = {}
        if isinstance(params, list):
            self.groups = param
            self.construct_group_params()
            
        else:
            self.groups = [{'params': params }]
            self.construct_group_params()


    def construct_group_params(self):
        for group in self.groups:
            group['lr'] = self.LR if 'lr' not in group else group['lr']
            group['betas'] = (self.B1,Self.B2) if 'betas' not in group else group['betas']
            group['eps'] = self.eps if 'eps' not in group else group['eps']
            for p in group['params']:
                self.state[id(p)] = {
                    'step':1,
                    'mt': torch.zeros_like(p),
                    'st': torch.zeros_like(p)
                }

    def __call__(self,epoch):
         with torch.no_grad():
            for group in self.groups:
                (B1 ,B2), lr, eps = group['betas'], group['lr'], group['eps']
                for p in group['params']:
                    # to not update the frozen paramters
                    if p.grad is None:
                        continue
                    t = copy.copy(self.state[p]['step'])
                    self.state[p]['step'] += 1
                    mt = self.state[p]['mt']
                    st = self.state[p]['st']
                        
                    # don't use mt= ..... because pytorch creat-
                    # -new tensor of the rsult and make mt point at them use .copy instead-
                    # that will compute the values and set a copy of them inside each mt and st tensor inplace
                    mt.copy_( B1 * mt + (1 - B1) * p.grad)
                    st.copy_( B2 * st + (1 - B2) * p.grad.pow(2))
                    
                    # apply correction bias to both mt and st because you intialize them with 0 so they
                    #   will force the network to take artificially tiny baby step.
                    #      This is called being "biased towards zero."
                    # the correction scale up the momentum at intial epochs and by the time t grows
                    #  its effect disappear entirly because the B**t decreases hardly so (1-B**t) will
                    #                be almost 1
                    mt_hat = mt/(1 - B1**t)
                    st_hat = st/(1 - B2**t)
                    
                    update_step = lr * mt_hat/(torch.sqrt(st_hat)+ eps)
                    p.sub_(update_step)

#----------------------------------------------------------------------------------------------------------
class LR_schedule:
    def __init__(self, max_lr,min_lr,decay_rate, step_size, algo_type, warm_up = 5, epochs = None ):
        self.warmup_lr = np.linspace(min_lr,max_lr,warm_up)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.decay_rate, self.step_size = decay_rate, step_size
        self.algo_type, self.warm_up, self.epochs = algo_type, warm_up, epochs
        
    def warmup(self,epoch):
        return self.warmup_lr[epoch]

    def __call__(self, epoch):
        if self.algo_type == 'step decay':
            return self.step_decay(epoch)
        elif self.algo_type == 'exponentail':
            return self.exponential(epoch)
        elif self.algo_type == 'cosin':
            return self.cosin_annealing(epoch)
        
    def step_decay(self, epoch):
        if epoch < self.warm_up:
          return self.warmup(epoch)  
        out = self.max_lr * self.decay_rate**math.floor((epoch-self.warm_up)/self.step_size)
        return out

    def exponential(self, epoch):
        if epoch < self.warm_up:
          return self.warmup(epoch) 
        out = self.max_lr * math.exp(-self.decay_rate * (epoch-self.warm_up))
        return out
        
    def cosin_annealing(self, epoch):
        if epoch < self.warm_up:
          return self.warmup(epoch) 
        out = self.min_lr + 0.5* (self.max_lr - self.min_lr) * (1+math.cos(math.pi*(epoch-self.warm_up)/(self.epochs-self.warm_up)))
        return out

def cross_entropy_Loss(logits , classes , weights , lambda_l1 = None,lambda_l2 = None ):
    lasso_penality = 0 if lambda_l1 is None else lambda_l1 * sum(torch.abs(W).sum() for W in weights)
    cost_W = f.cross_entropy(logits,classes)
    ridge_penality = 0 if lambda_l2 is None else lambda_l2 * sum((W**2).sum() for W in weights.values())
    loss = cost_W + ridge_penality + lasso_penality 
    return loss , cost_W

def total_loss_accur(model , data):
    total_loss = 0
    total_images = 0
    accuracy = 0
    SM = Softmax()
    with torch.no_grad():
        for images,classes in data:
             logits = model(images)
             pred = torch.argmax(SM(logits),dim = 1)
             accuracy += (pred == classes).float().sum()
             loss = f.cross_entropy(logits,classes,reduction='sum')
             total_loss+=loss.item()
             total_images += classes.shape[0]
        accuracy/= (total_images)
        total_loss/= total_images
    return total_loss, accuracy*100
