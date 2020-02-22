import torch
import torch.nn as nn
import numpy as np
import copy

def clones(module: nn.Module, 
           N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) 
                          for _ in range(N)])

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

if __name__ == "__main__":
    x = nn.Linear(3, 3)
    xs = clones(x, 5)
    print(xs)