import torch
import math
import numpy as np
from torch.nn import functional as f
from Src.layers import Softmax
from Src.engine import Train_engine
from matplotlib import pyplot as plt
from tqdm import tqdm

class Adams_W:
    def __init__(self, params, B1=0.9 , B2 = 0.999, LR = 1e-3  ,epsilon = 1e-7, weight_decay =0 ):
        self.LR , self.B1 , self.B2, self.eps =  LR, B1, B2 , epsilon
        self.weight_decay = weight_decay
        self.state = {}
        # compute the ratio between the weight update step size and the current weight
        self.Update_to_Weight_Ratio = []
        if isinstance(params, list):
            self.groups = params
            
        else:
            self.groups = [{'params': params }]
        self.construct_group_params()
       


    def construct_group_params(self):
        for group in self.groups:
            group['lr'] = group.get('lr', self.LR)
            group['betas'] = group.get('betas', (self.B1, self.B2))
            group['eps'] = group.get('eps', self.eps)
            group['weight_decay'] = group.get('weight_decay', self.weight_decay)
            for p in group['params']:
                self.Update_to_Weight_Ratio.append([])
                self.state[id(p)] = {
                    'step':1,
                    'mt': torch.zeros_like(p),
                    'st': torch.zeros_like(p)
                }

    def __call__(self):
         with torch.no_grad():
            idx = 0
            for group in self.groups:
                (B1 ,B2), lr, eps = group['betas'], group['lr'], group['eps']
                weight_decay = group['weight_decay']
                for p in group['params']:
                    # to not update the frozen paramters
                    if p.grad is None:
                        continue
                    t = self.state[id(p)]['step']
                    self.state[id(p)]['step'] += 1
                    mt = self.state[id(p)]['mt']
                    st = self.state[id(p)]['st']
                        
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

                    update_step = lr * (mt_hat/(torch.sqrt(st_hat)+ eps) + p * weight_decay)
                    self.Update_to_Weight_Ratio[idx].append(((-update_step).std()/p.std()).log10().item())
                    p.sub_(update_step)
                    idx +=1

#----------------------------------------------------------------------------------------------------------
class LR_schedule:
    def __init__(self, optimizer, algo_type, epochs, decay_rate = None, step_size = None, warm_up = 5):
        self.optimizer = optimizer
        self.warmup_lr = {}
        self.decay_rate, self.step_size = decay_rate, step_size
        self.algo_type, self.warm_up, self.epochs = algo_type, warm_up, epochs
        self.base_lr = [group['lr'] for group in self.optimizer.groups]
        
        for i, group in enumerate(self.optimizer.groups):
            max_lr = self.base_lr[i]
            min_lr = max_lr/10 if algo_type != 'cosin' else 1e-6
            self.warmup_lr[i] = np.linspace(min_lr ,max_lr, warm_up)
        
    def warmup(self,epoch):
        return self.warmup_lr[epoch]

    def step(self, epoch):
        sched_ = self.sched()
        for i, group in enumerate(self.optimizer.groups):
            max_lr = self.base_lr[i]
            min_lr = max_lr/10 if self.algo_type != 'cosin' else 1e-6
            if epoch < self.warm_up:
                new_lr = self.warmup_lr[i][epoch]
            else:
                new_lr =  sched_(max_lr ,epoch, min_lr)
            group['lr'] = new_lr

    def set_lr(self,lr):
        for group in self.optimizer.groups:
            group['lr'] =  lr
        
    def sched(self): 
        def step_decay(max_lr,epoch , min_lr):
            out = max_lr * self.decay_rate**math.floor((epoch-self.warm_up)/self.step_size)
            return out
    
        def exponential(max_lr , epoch , min_lr):
            out = max_lr * math.exp(-self.decay_rate * (epoch-self.warm_up))
            return out
            
        def cosin_annealing(max_lr , epoch , min_lr):
            out = min_lr + 0.5* (max_lr - min_lr) * (1+math.cos(math.pi*(epoch-self.warm_up)/(self.epochs-self.warm_up)))
            return out

        if self.algo_type == 'step decay':
            return step_decay
        elif self.algo_type == 'exponential':
            return exponential
        elif self.algo_type == 'cosin':
            return cosin_annealing
        else:
            raise ValueError(f"Unknown algo_type: {self.algo_type}")
            
#-----------------------------------------------------------------------------------------
class cross_entropy_Loss:
    def __init__(self , weights , lambda_l1 = None,lambda_l2 = None ):
        self.weights = weights
        self.l1 = lambda_l1 
        self.l2 = lambda_l2
        self.lasso = True if self.l1 is not None else False
        self.ridge = True if self.l2 is not None else False
        
    def compute(self, logits , classes):
        cost_W = f.cross_entropy(logits,classes)
        lasso_penality = 0 if not self.lasso else self.l1 * sum(torch.abs(W).sum() for W in self.weights)
        ridge_penality = 0 if not self.ridge else self.l2 * sum((W**2).sum() for W in self.weights)
        loss = cost_W + ridge_penality + lasso_penality 
        return loss

#-------------------------------------------------------------------------------------------

def total_loss_accur(model , data):
    total_loss = 0.0
    total_images = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for images,classes in data:
             logits = model(images)
             loss = f.cross_entropy(logits,classes,reduction='sum')
             pred = torch.argmax(logits,dim = 1)
             correct += (pred == classes).sum().item()
             total_loss+=loss.item()
             total_images += classes.shape[0]
        accuracy = 100 * correct / total_images
        avg_loss = total_loss / total_images
    return avg_loss, accuracy

#------------------------------------------------------------------------------------------

def max_lr_finder(LRs , data_lr , tr_engine, sched):
    losses = []
    for i, (images, classes) in enumerate(tqdm( data_lr , desc ="lr finder iterations")):
        sched.set_lr(10**LRs[i])
        # tr_engine return loss as float not a tensor so no need to use .item()
        loss, logits = tr_engine.step(images, classes)
        losses.append(loss)
    losses = np.array(losses)
    plt.plot(10**LRs, np.log(losses))
    plt.xlabel('learning rate')
    plt.ylabel('losses')
    plt.title('learning rate VS loss')
    plt.show()
    return 10**LRs[losses.argmin()]
