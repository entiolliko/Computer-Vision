import torch
import numpy as np

x = np.asarray([1,2,3,4,5])
cond = x < 3
print(np.argwhere(cond == True).shape[0])
