import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.ClAtNet import ClAtNet
from datasets.mnist import MNISTDataset

import os

test_dataset = MNISTDataset('data/MNIST/mnist_test.csv')

nimgs = 10
nsteps = 784
nheads = 4

dev = torch.device('cuda:0')
net = ClAtNet(nfeatures=128, nclasses=10, nheads=nheads, dropout=0.0).to(dev)
net.load_state_dict(torch.load('weights/7.pth')['model_state_dict'])
net.eval()

for img_id, (img, lbl) in enumerate(test_dataset):
    if img_id == nimgs: break

    print(f'Working on {img_id}...')
    os.system(f'mkdir -p vis/{img_id:05d}')
    
    x = net.conv(img.unsqueeze(0).to(dev))
    x = x.reshape(-1, net.nfeatures, 28 * 28).permute(0, 2, 1)
    x, attn = net.attn(x, x, x)
    x = x.reshape(-1, net.nfeatures * 28 * 28)
    x = net.cls(x)

    for step_idx in range(nsteps):
        tmp = torch.cat(3 * [img]).permute(1, 2, 0)
        tmp[step_idx // 28, step_idx % 28] = torch.Tensor([1, 0, 0])

        plt.figure(figsize=(10 * nheads, 10))
        for head_idx in range(nheads):
            plt.subplot(1, nheads, head_idx + 1)
            plt.imshow(tmp, origin='upper')
            # plt.subplot(1, nheads, head_idx + 1)
            plt.imshow(attn.detach().cpu().squeeze(0)[head_idx][step_idx].reshape(28, 28), alpha=0.5, origin='upper')
            plt.xticks([])
            plt.yticks([])
            plt.title(f'GT: {lbl.item()}, Pred: {torch.max(x, dim=1)[1].item()}')
        # plt.suptitle(step_idx)

        plt.tight_layout()
        # plt.show()
        plt.savefig(f'vis/{img_id:05d}/{step_idx:03d}')
        plt.close()

# c = 0
# tbar = tqdm(enumerate(test_dataset))
# for img_id, (img, lbl) in tbar:
#     out, _ = net(img.unsqueeze(0).to(dev))
#     out = out.cpu()
#     _, pred = torch.max(out, dim=1)
#     c += (lbl == pred).cpu().item()
#     tbar.set_description(f'Correct: {c}/{img_id+1}')