"""Trigger the autograd error"""
import torch
from torch import nn, autograd

class identity(autograd.Function):
    @staticmethod
    def forward(ctx, data):
        # correct
        #ctx.save_for_backward(data)

        # correct
        #ctx.save_for_backward(data + 1)

        # RuntimeError: No grad accumulator for a saved leaf!
        ctx.save_for_backward(data.view((1, -1)))
        return data

    @staticmethod
    def backward(ctx, data):
        print(ctx.saved_tensors)
        return data


a = torch.ones((10,)).requires_grad_()
b = identity.apply(a).sum()
b.backward()
