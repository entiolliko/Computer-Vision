import torch
import math

x = torch.asarray([1.0,1.0,1.0])
X = torch.asarray([[1.0,2.0,3.0], [1.0,2.0,3.0], [1.0,2.0,3.0], [1.0,2.0,3.0]])

x_ = x.expand(list(X.shape)[0], 3)
print(torch.sum((x_ - X)**2, dim=1))
print(torch.cdist(x_, X))