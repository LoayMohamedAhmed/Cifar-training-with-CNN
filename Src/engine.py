class Train_engine:
    def __init__(self,model, optim , loss):
        self.model = model
        self.parameters = self.model.parameters()
        self.optim = optim
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
        self.optim()
        return loss.item() , logits
