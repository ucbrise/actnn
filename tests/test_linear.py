import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from actnn import config, QLinear

torch.manual_seed(0)

# config.activation_compression_bits = [8]
config.compress_activation = False
model = nn.Linear(100, 10).cuda()
qmodel = QLinear(100, 10).cuda()
with torch.no_grad():
    qmodel.weight.copy_(model.weight)
    qmodel.bias.copy_(model.bias)

x = torch.randn(128, 64, 100).cuda()


def get_grad(model):
    model.weight.grad = None
    model.bias.grad = None
    y = model(x)
    loss = torch.square(y).sum()
    loss.backward()


get_grad(model)
get_grad(qmodel)

print((model.weight.grad - qmodel.weight.grad).norm(), model.weight.grad.norm())
print((model.bias.grad - qmodel.bias.grad).norm(), model.bias.grad.norm())
