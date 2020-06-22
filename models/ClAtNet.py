import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.MultiHeadedAttention import MultiHeadedAttention
from models.modules.Encoder import Encoder, EncoderLayer
from models.modules.PositionwiseFeedForward import PositionwiseFeedForward

class ClAtNet(nn.Module):
    def __init__(self, in_channels, size,
                 nfeatures, nclasses, nheads, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, nfeatures, kernel_size=3, padding=1),
            nn.ReLU()
        )
        attn = MultiHeadedAttention(h=nheads, d_model=nfeatures, dropout=dropout)
        ff = PositionwiseFeedForward(nfeatures, d_ff=2*nfeatures, dropout=dropout)
        self.attn = Encoder(EncoderLayer(nfeatures, attn, ff, dropout), 1)
        self.cls = nn.Linear(size[0] * size[1], nclasses)
        self.nfeatures = nfeatures

        # self.init_(self.conv)
        self.init_(self.attn)
        # self.init_(self.cls)

    def init_(self, module):
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), self.nfeatures, -1).permute(0, 2, 1)
        x = self.attn(x, None)
        x = torch.mean(x, dim=-1)
        # x = x.reshape(-1, 28 * 28)
        x = self.cls(x)
        return x

if __name__ == "__main__":
    dev = torch.device('cpu')
    net = ToyModel(128, 2).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    for iter_id in range(100):
        inps = torch.rand(4, 3, 100, 100).to(dev)
        lbls = torch.randint(low = 0, high = 2, size = (4, 100, 100)).to(dev)

        outs = net(inps)
        loss = criterion(outs, lbls)
        loss.backward()
        optimizer.step()
        
        print(outs.shape)
        print(iter_id, loss.item())