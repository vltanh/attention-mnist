import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.modules.utils import clones

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [ l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
              for l, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = self.attention(query, key, value, mask=mask,
                                      dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None, dropout=None):
        scores = self.get_score(query, key)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def get_score(self, query, key):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / np.sqrt(d_k)
        return scores
        