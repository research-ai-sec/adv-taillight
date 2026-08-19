import numpy as np
import torch
from enum import Enum


class OptimType(Enum):
    adam = 'adam'
    dpr = 'dpr'
    aiadam = 'aiadam'
    mim = 'mim'
    nifgsm = 'nifgsm'
    fgsm = 'fgsm'
    apaa = 'apaa'
    mim_tanh = 'mim_tanh'


class MIFGSMOpt:

    def __init__(self, size, lr, dtype=torch.float32, decay=1.0):
        self.momentum = torch.zeros(size, dtype=dtype)
        self.decay = decay
        self.lr = lr
        self.type = OptimType.mim

    def reset(self):
        self.momentum = torch.zeros_like(self.momentum)

    def update(self, grad):
        grad = grad.detach().cpu()
        epsilon = 1e-10

        if torch.isnan(grad).any() or torch.isinf(grad).any():
            print("grad contains NaN or Inf!")
            return None


        if len(grad.shape) == 5:
            grad_mean = torch.mean(torch.abs(grad), dim=(2, 3, 4), keepdim=True)
        elif len(grad.shape) == 4:
            grad_mean = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        elif len(grad.shape) == 2:
            grad_mean = torch.mean(torch.abs(grad), dim=1, keepdim=True)
        elif len(grad.shape) == 3:
            grad_mean = torch.mean(torch.abs(grad), dim=(1, 2), keepdim=True)

        grad_normalized = grad / (
            grad_mean + epsilon
        )


        delta = self.decay * self.momentum + grad_normalized
        self.momentum = delta

        if torch.isnan(delta).any() or torch.isinf(delta).any():
            print("delta contains NaN or Inf!")
            return None


        delta = self.lr * delta.sign()

        return delta


class AdamOpt:

    def __init__(self, size, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, dtype=np.float32):
        self.exp_avg = torch.zeros(size, dtype=dtype)
        self.exp_avg_sq = torch.zeros(size, dtype=dtype)
        self.beta1 = torch.tensor(beta1, dtype=dtype)
        self.beta2 = torch.tensor(beta2, dtype=dtype)
        self.eps = torch.tensor(eps, dtype=dtype)
        self.lr = torch.tensor(lr, dtype=dtype)
        self.step = torch.tensor(0, dtype=dtype)
        self.type = OptimType.dpr


    def update(self, grad):

        self.step += 1

        bias_correction1 = 1 - self.beta1 ** self.step
        bias_correction2 = 1 - self.beta2 ** self.step

        self.exp_avg = self.beta1 * self.exp_avg + (1 - self.beta1) * grad
        self.exp_avg_sq = self.beta2 * self.exp_avg_sq + (1 - self.beta2) * (grad ** 2)
        denom = (np.sqrt(self.exp_avg_sq) / np.sqrt(bias_correction2)) + self.eps

        step_size = self.lr / bias_correction1

        return (step_size / denom * self.exp_avg)