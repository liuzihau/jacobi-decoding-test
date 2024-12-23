import torch
from torch import nn
x = nn.Parameter(torch.randn(896))
print(x.unsqueeze(0).unsqueeze(0).repeat(1, 10, 1).shape)