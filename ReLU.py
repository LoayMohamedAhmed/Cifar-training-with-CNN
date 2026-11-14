import torch

class ReLU:
    def __init__(self):
        # construct a dic that hold all hooks applied to that layer
        self._forward_hooks = {}
        self._hook_id = 0
        
    def __call__(self,Z):
        #self.out = torch.maximum(torch.tensor(0.0),Z)
        self.out = torch.maximum(0.001*Z,Z)
        # loop over each hook and execute it
        for hook_id , hook_fn in self._forward_hooks.items():
            hook_result = hook_fn(self, Z, self.out)
            # overwrite the output if the hook was used to alter the output
            if hook_result is not None:
                self.out = hook_result
        return self.out
        
    def parameters(self):
        return []
        
    # this function responsible for receiving hooks and assign them
    # to that layer and return a handle if you determine todelete 
    # the hook later
    def register_forward_hook(self,hook_fn):
        handle_id = self._hook_id
        self._forward_hooks[self._hook_id] =hook_fn
        self._hook_id+=1
        return handle_id
