import torch
g = torch.Generator().manual_seed(2882001)
class Linear:
    def __init__(self, in_dim , out_dim , bias = True):
        self.W = torch.randn((in_dim , out_dim), generator = g)*(2/in_dim)**0.5
        self.b = torch.zeros(out_dim) if bias else None

    def __call__(self,X):
        self.out = X @ self.W
        if self.b is not None:
            self.out+= self.b
        return self.out
    
    def parameters(self):
        return [self.W] + ([self.b] if self.b is not None else [])
