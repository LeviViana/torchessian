import torch
from . import hessian_matmul

def lanczos(model, loss_function, dataloader, m, buffer=2):
    n = sum(p.data.numel() for p in model.parameters() if p.requires_grad)

    assert n >= m
    assert buffer >= 2

    v = torch.ones(n)
    v /= torch.norm(v)
    
    w = torch.zeros_like(v)
    
    print("[LANCZOS iter 0] batch")
    k = len(dataloader)
    for i, batch in enumerate(dataloader):
        device = model.fc.weight.data.device
        v_ = v.to(device)
        w = w.to(device) 
        x, y = batch
        x, y = x.to(device), y.to(device)
        batch = x, y
        w += hessian_matmul(model, loss_function, v_, batch) / k
    
    v = v.to(w.device)
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
                v = torch.rand(n).to(w.device)

                for v_ in V:
                    v_ = v_.to(v.device)
                    v -= v.dot(v_) * v_

                done = torch.norm(v) > 0
                if k > 2 and not done: # This shouldn't happen even twice
                    raise Exception("Can't find orthogonal vector")

        # Full re-orthogonalization
        for v_ in V:
            v_ = v_.to(v.device)
            v -= v.dot(v_) * v_

        v /= torch.norm(v)
        V.append(v)

        # Saving GPU memory
        if len(V) > buffer:
            V[-buffer - 1] = V[-buffer - 1].cpu()

        w = torch.zeros_like(v)
        print("[LANCZOS iter %d] batch" % i)
        for j, batch in enumerate(dataloader):
            device = model.fc.weight.data.device
            v_ = v.to(device)
            w = w.to(device) 
            x, y = batch
            x, y = x.to(device), y.to(device)
            batch = x, y
            w += hessian_matmul(model, loss_function, v_, batch) / k
            
        alpha.append(w.dot(v))
        w = w - alpha[-1] * V[-1] - beta[-1] * V[-2]

    T = torch.diag(torch.Tensor(alpha))
    for i in range(m - 1):
        T[i, i + 1] = beta[i]
        T[i + 1, i] = beta[i]

    V = torch.cat(tuple(v.cpu().unsqueeze(0) for v in V), 0)
    return T, V


def gauss_quadrature(model, loss_function, dataloader, m, buffer=2):
    T, _ = lanczos(model, loss_function, dataloader, m, buffer)
    D, U = torch.eig(T, eigenvectors=True)
    L = D[:, 0] # All eingenvalues are real
    W = torch.Tensor(list(U[0, i] ** 2 for i in range(m)))
    return L, W
