import numpy as np
import torch

class Metrics():
    def __init__(self):
        self.reset()

class Accuracy(Metrics):
    def __init__(self):
        super().__init__()

    def reset(self):        
        self.scores = []
    
    def calculate(self, output, target):
        _, pred = torch.max(output, dim=1)
        return (torch.sum(pred == target).float() / target.size(0)).item()
    
    def update(self, x):
        self.scores.append(x)
    
    def value(self):
        return np.mean(self.scores)
        
    def summary(self):
        return self.scores