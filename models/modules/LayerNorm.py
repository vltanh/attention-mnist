import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b

if __name__ == "__main__":
    features = 3
    eps = 1e-8
    batch_size = 8

    g = LayerNorm(features, eps)
    inp = torch.randn(size=(batch_size, features))
    out = g(inp)
    print(inp.shape)
    print(out.shape)