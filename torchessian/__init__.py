import torch
from copy import deepcopy
import math

SIG = 0.001
EPS = 0.001

def hessian_matmul(model, loss_function, v, batch):
    global EPS
    
    model.zero_grad()
    model_delta = deepcopy(model)
    begin = end = 0
    for p in model_delta.parameters():
        end = begin + p.data.numel()
        p_flat = p.data.view(-1)        
        p_flat += EPS * v[begin:end]
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
    
    return (grad_w_delta - grad_w) / EPS


def lanczos(model, loss_function, batch, m):
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


def gauss_quadrature(model, loss_function, batch, m):
    T, _ = lanczos(model, loss_function, batch, m)
    D, U = torch.eig(T, eigenvectors=True)
    L = D[:, 0] # All eingenvalues are real anyway
    W = torch.Tensor(list(U[0, i] ** 2 for i in range(m)))
    return L, W


def f(l, t, sig):
    return torch.exp(-(t - l) ** 2 / (2 * sig)) / (sig * math.sqrt(2 * math.pi))


def F(x, L, W, m):
    global SIG
    result = torch.zeros_like(x)
    for i in range(m):
        result += f(L[i], x, SIG) * W[i]
    
    return result


def spectrum(model, loss_function, batch, m, x_min=-100, x_max=100):
    L, W = gauss_quadrature(model, loss_function, batch, m)
    support = torch.linspace(x_min, x_max, 1000)
    density = F(support, L, W, m)
    return support, density
