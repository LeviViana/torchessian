import torch
from copy import deepcopy


def hessian_matmul(model, loss_function, v, batch):
    r = 0.001
    model_delta = deepcopy(model)
    begin = end = 0
    for p in model_delta.parameters():
        end = begin + p.data.numel()
        p_flat = p.data.view(-1)        
        p_flat += r * v[begin:end]
        begin = end
    
    x, y = batch
    E = loss_function(model(x), y)
    E_delta = loss_function(model_delta(x), y)
    
    E.backward()
    E_delta.backward()
    
    grad_w = torch.cat(list(p.grad.view(1, -1) for p in model.parameters()), 1)
    grad_w_delta = torch.cat(list(p.grad.view(1, -1) for p in model_delta.parameters()), 1)
    
    grad_w.squeeze_()
    grad_w_delta.squeeze_()
    
    return (grad_w_delta - grad_w) / r


def lanczos():
    return