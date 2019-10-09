import torch
import math

SIG = 0.0032 # SIG ** 2 == 1e-5

def hessian_matmul(model, loss_function, v, batch):

    model.zero_grad()

    x, y = batch
    E = loss_function(model(x), y)
    v.requires_grad = False

    grad_result = torch.autograd.grad(
                E,
                (p for p in model.parameters() if p.requires_grad),
                create_graph=True
    )
    grad_result = torch.cat(tuple(p.view(1, -1) for p in grad_result), 1)
    grad_result.backward(v.view(1, -1))

    result = torch.cat(
        tuple(p.grad.view(1, -1) for p in model.parameters() if p.requires_grad),
        1
    )

    return result.squeeze(0)


def lanczos(model, loss_function, batch, m, buffer=2):
    """
        This in an implementation of the Lanczos Algorithm as stated in
        https://en.wikipedia.org/wiki/Lanczos_algorithm.
        Inputs:
            model: a torch.nn.Module with some parameters having requires_grad
                set True.
            loss_function: the loss the neural net is optimizing, and for which
                the hessian spectrum will be calculated.
            batch: a tuple (inputs, targets) representing the dataset for which
                the hessian spectrum will be calculated.
            m: the number of iterations of the Lanczos Algorithm.
            buffer: the number of base vectors you wish to keep in GPU memory,
                if ever you are using GPU. Set buffer to 2 if you are having
                OOM errors.
        Outputs:
            T: the tridiagonal matrix of the Lanczos Algorithm.
            V : the orthonormal basis of the Lanczos Algorithm.
    """
    n = sum(p.data.numel() for p in model.parameters() if p.requires_grad)

    assert n >= m
    assert buffer >= 2

    v = torch.ones(n).to(batch[0].device)
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

        w = hessian_matmul(model, loss_function, v, batch)
        alpha.append(w.dot(v))
        w = w - alpha[-1] * V[-1] - beta[-1] * V[-2]

    T = torch.diag(torch.Tensor(alpha))
    for i in range(m - 1):
        T[i, i + 1] = beta[i]
        T[i + 1, i] = beta[i]

    V = torch.cat(tuple(v.cpu().unsqueeze(0) for v in V), 0)
    return T, V


def gauss_quadrature(model, loss_function, batch, m, buffer=2):
    T, _ = lanczos(model, loss_function, batch, m, buffer)
    D, U = torch.eig(T, eigenvectors=True)
    L = D[:, 0] # All eingenvalues are real
    W = torch.Tensor(list(U[0, i] ** 2 for i in range(m)))
    return L, W


def gaussian_density(x, mean, sig):
    return torch.exp(-(x - mean) ** 2 / (2 * sig)) / (sig * math.sqrt(2 * math.pi))


def F(x, L, W, m):
    global SIG
    result = torch.zeros_like(x)
    for i in range(m):
        result += gaussian_density(x, L[i], SIG) * W[i]

    return result


def spectrum(model, loss_function, batch, m, x_min=-50, x_max=50, buffer=2):
    L, W = gauss_quadrature(model, loss_function, batch, m, buffer)
    support = torch.linspace(x_min, x_max, 10000)
    density = F(support, L, W, m)
    return support, density
