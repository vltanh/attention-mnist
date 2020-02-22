import copy

import torch.nn as nn

from models.modules.MultiHeadedAttention import MultiHeadedAttention
from models.modules.PositionalEncoding import PositionalEncoding
from models.modules.PositionwiseFeedForward import PositionwiseFeedForward
from models.modules.EncoderDecoder import EncoderDecoder
from models.modules.Encoder import Encoder, EncoderLayer
from models.modules.Decoder import Decoder, DecoderLayer
from models.modules.Embeddings import Embeddings
from models.modules.Generator import Generator

class AtNet(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, N=1,
                 d_model=512, d_ff=2048, h=8, dropout=0.1):
        super().__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(
                Embeddings(d_model, src_vocab), 
                c(position)
            ),
            nn.Sequential(
                Embeddings(d_model, tgt_vocab),
                c(position)
            ),
            Generator(d_model, tgt_vocab)
        )

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.model(src, tgt, src_mask, tgt_mask)