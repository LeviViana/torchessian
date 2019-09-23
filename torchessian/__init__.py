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
    
    model.zero_grad()
    
    return (grad_w_delta - grad_w) / r


def lanczos(model, loss_function, batch, m):
    global H
    n = sum(p.data.numel() for p in model.parameters())
    v = torch.ones(n)
    v /= torch.norm(v)
    w = hessian_matmul(model, loss_function, v, batch)
    alpha = []
    alpha.append(w.dot(v))
    w -= alpha[0] * v
    
    V = [v]
    beta = []
    
    for i in range(1, m):
        b = torch.norm(w)
        beta.append(b)
        if b > 0:
            v = w / b
        else:
            done = False
            k = 0
            while not done:
                k += 1
                v = torch.rand(n)
                
                for v_ in V:
                    v -= v.dot(v_) * v_
                
                done = torch.norm(n) > 0
                if k > n * 10:
                    raise Exception("Can't find orthogonal vector")
                            
        for v_ in V:
            v -= v.dot(v_) * v_
        
        v /= torch.norm(v)
               
        V.append(v)
        w = hessian_matmul(model, loss_function, v, batch)
        alpha.append(w.dot(v))
        w = w - alpha[-1] * V[-1] - beta[-1] * V[-2]

    T = torch.diag(torch.Tensor(alpha))
    for i in range(m - 1):
        T[i, i + 1] = beta[i] 
        T[i + 1, i] = beta[i]

    V = torch.cat(list(v.unsqueeze(0) for v in V), 0)
    return T, V