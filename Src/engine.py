class Train_engine:
    def __init__(self,model, optim , sched,loss):
        self.model = model
        self.parameters = self.model.parameters()
        self.optim = optim
        self.sched = sched
        self.loss = loss

    def step(self, X, ground_truth):
        # forward pass
        self.model.train()
        logits = model(X)
        loss = self.loss.compute(logits , ground_truth)
            
        # backward pass
        for p in self.parameters.values():
            p.grad = None
        loss.backward()
            
        # update step 
        self.sched.step(epoch)
        self.optim()
        return loss.item(),logits
