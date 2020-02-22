import torch.nn as nn
import numpy as np

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
    
    def forward(self, x):
        return self.lut(x) * np.sqrt(self.d_model)