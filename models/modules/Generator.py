import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)
    
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

if __name__ == "__main__":
    import torch

    d_model = 3
    vocab = 2
    batch_size = 8

    g = Generator(d_model, vocab)
    inp = torch.randn(size=(batch_size, d_model))
    out = g(inp)
    print(inp.shape)
    print(out.shape)