import torch.nn as nn
import torch.nn.functional as F
import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z)
print(out)
out.backward()
