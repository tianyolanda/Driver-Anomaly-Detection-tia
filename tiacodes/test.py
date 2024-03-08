import torch
x = torch.ones([1, 2, 1, 3, 3])
y = torch.ones([1, 2, 1, 1, 1]) * 3
print('x',x)
print('y',y)
print('x*y',x*y)