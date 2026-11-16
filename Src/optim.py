import torch
import math
import numpy as np
from torch.nn import functional as f
class Adams:
    def __init__(self, param, B1=0.9 , B2 = 0.999, LR = 1e-3  ,epsilon = 1e-7,Schedual = None):
        self.Mt = [torch.zeros_like(p) for p in param.values()]
        self.St = [torch.zeros_like(p) for p in param.values()]
        self.step = 0
        self.param = param
        self.LR , self.B1 , self.B2, self.eps =  LR, B1, B2 , epsilon
        self.schad = Schedual
        # compute the ratio between the weight update step size and the current weight
        self.Update_to_Weight_Ratio = [[] for p in param.values()]
       

    def __call__(self,epoch):
        self.step +=1
        t = self.step
        idx = 0
        with torch.no_grad():
            for mt , st , p in zip(self.Mt,self.St,self.param.values()):
                if p.grad is None:
                    continue
                # don't use mt= ..... because pytorch create 
                #    new tensor of the rsult and make mt point at them
                mt.copy_( self.B1*mt +(1-self.B1) * p.grad)
                st.copy_(self.B2 * st  + (1-self.B2) * p.grad.pow(2))
                # apply correction bias to both mt and st because you intialize them with 0 so they
                #   will force the network to take artificially tiny baby step.
                #      This is called being "biased towards zero."
                mt_hat = mt/(1-self.B1**t)
                st_hat = st/(1-self.B2**t)
                # the correction scale up the momentum at intial epochs and by the time t grows
                #  its effect disappear entirly because the B**t decreases hardly so (1-B**t) will
                #                be almost 1
                update_step = self.LR * mt_hat/(torch.sqrt(st_hat)+self.eps)
                self.Update_to_Weight_Ratio[idx].append(((-update_step).std()/p.std()).log10().item())
                p.sub_(update_step)
                idx+=1
        self.LR = self.LR if not self.schad else self.schad(epoch)

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
