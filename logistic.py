import numpy as np

def logistic(prod, y):
    """
    prod: Nx1
    y: Nx1
    """
    obj = np.log(1 + np.exp(-y * prod))
    return np.mean(obj)

def logistic_grad(prod, xl, y):
    """
    prod: Nx1
    xl: Nxdl
    y: Nx1
    grad: dlx1
    """
    e = np.exp(-y * prod)
    coef = -y * e / (1 + e)
    grad = xl.transpose().dot(coef) / len(coef)
    return grad

def logistic_acc(prod, y):
    """
    prod: Nx1
    y: nx1
    """
    pred = np.logical_not(np.logical_xor(prod > 0, y > 0))
    return np.mean(pred)

def logistic_grad_diff(prod1, prod2, xl, y):
    e = np.exp(-y * prod1)
    coef1 = -y * e / (1 + e)
    e = np.exp(-y * prod2)
    coef2 = -y * e / (1 + e)
    grad = xl.transpose().dot(coef1 - coef2) / len(coef1)
    return grad
