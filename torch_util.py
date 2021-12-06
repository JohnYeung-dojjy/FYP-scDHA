import torch

def check_grad_nan(parameters):
        
    for p in parameters:
        if torch.isnan(p).any():
            return True
    
    return False
        
    