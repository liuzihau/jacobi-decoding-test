import torch
from torch import nn
# x = nn.Parameter(torch.randn(896))
# print(x.unsqueeze(0).unsqueeze(0).repeat(1, 10, 1).shape)


output = torch.randn(4, 10, 256)
target = torch.randint(0, 255, (4, 10))
_, pred = output.topk(3, 2, True, True)
print(pred.shape)
print(target.shape)
correct = pred.eq(target.view(4, 10, -1).expand_as(pred))
print(correct.float().sum(0).sum(-1).shape)
