
import numpy as np
import torch
import torchvision

def execute_filename(filename):
    filename = filename.split('-')[1]
    return filename

def rescale(img):
    return (img - torch.min(img))/(torch.max(img) - torch.min(img))

class NormMaxMin():
    def __call__(self, x):
        return (x.float() - torch.min(x)) / (torch.max(x) - torch.min(x))
