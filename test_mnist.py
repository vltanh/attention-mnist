import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.ClAtNet import ClAtNet
from datasets.mnist import MNISTDataset

import os

test_dataset = MNISTDataset('data/MNIST/mnist_test.csv')

nimgs = 1000
nsteps = 784
nheads = 1

dev = torch.device('cuda:0')
net = ClAtNet(nfeatures=64, nclasses=10, nheads=nheads, dropout=0.0).to(dev)
net.load_state_dict(torch.load('runs/2020-02-25_20:17:54/best_loss.pth')['model_state_dict'])
net.eval()

def print_attn(m, i, o):
    attn = m.attn

    # plt.imshow(attn.detach().cpu().squeeze())
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    
    for step_idx in range(nsteps):
        tmp = torch.cat(3 * [img]).permute(1, 2, 0)
        tmp[step_idx // 28, step_idx % 28] = torch.Tensor([1, 0, 0])

        plt.figure(figsize=(10 * nheads, 10))
        for head_idx in range(nheads):
            plt.subplot(1, nheads, head_idx + 1)
            plt.imshow(tmp, origin='upper')
            # plt.subplot(1, nheads, head_idx + 1)
            plt.imshow(attn.detach().cpu().squeeze(0)[head_idx][step_idx].reshape(28, 28), alpha=0.8, origin='upper')
            plt.xticks([])
            plt.yticks([])
            # plt.title(f'GT: {lbl.item()}, Pred: {torch.max(x, dim=1)[1].item()}')
        # plt.suptitle(step_idx)

        plt.tight_layout()
        # plt.show()
        plt.savefig(f'vis/{img_id:05d}/{step_idx:03d}')
        plt.close()
net.attn.layers[0].self_attn.register_forward_hook(print_attn)

for img_id, (img, lbl) in enumerate(test_dataset):
    if img_id == nimgs: break

    print(f'Working on {img_id}...')
    os.system(f'mkdir -p vis/{img_id:05d}')

    x = net(img.unsqueeze(0).to(dev))
    print(f'[{img_id}] Pred: {torch.max(x, dim=1)[1].item()}, Truth:{lbl.item()}')

# c = 0
# tbar = tqdm(enumerate(test_dataset))
# for img_id, (img, lbl) in tbar:
#     out, _ = net(img.unsqueeze(0).to(dev))
#     out = out.cpu()
#     _, pred = torch.max(out, dim=1)
#     c += (lbl == pred).cpu().item()
#     tbar.set_description(f'Correct: {c}/{img_id+1}')