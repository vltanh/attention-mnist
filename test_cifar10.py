import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.ClAtNet import ClAtNet
from datasets.cifar10 import CIFAR10Dataset

import os

SUBMISSION = True

nbatches = 32

test_dataset = CIFAR10Dataset('data/CIFAR10/test')#, 
                            #   'data/CIFAR10/trainLabels.csv')
test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=nbatches)

nimgs = len(test_dataset)
nsteps = 32 * 32
nheads = 2

dev = torch.device('cuda:0')
net = ClAtNet(in_channels=3, size=(32, 32),
              nfeatures=64, nclasses=10, nheads=nheads, dropout=0.0).to(dev)
net.load_state_dict(torch.load('runs/2020-02-26_07:43:34/best_loss.pth')['model_state_dict'])
net.eval()

def print_attn(m, i, o):
    attn = m.attn

    # plt.imshow(attn.detach().cpu().squeeze())
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    
    for img_idx in range(nbatches):
        for step_idx in range(nsteps):
            tmp = img[img_idx].clone().permute(1, 2, 0)
            tmp[step_idx // 32, step_idx % 32] = torch.Tensor([1, 0, 0])

            # plt.figure(figsize=(10 * nheads, 10))
            for head_idx in range(nheads):
                plt.subplot(1, nheads, head_idx + 1)
                plt.imshow(tmp, origin='upper')
                # plt.subplot(1, nheads, head_idx + 1)
                plt.imshow(attn[img_idx].detach().cpu()[head_idx][step_idx].reshape(32, 32), alpha=0.5, origin='upper')
                plt.xticks([])
                plt.yticks([])
                # plt.title(f'GT: {lbl.item()}, Pred: {torch.max(x, dim=1)[1].item()}')
            # plt.suptitle(step_idx)

            plt.tight_layout()
            # plt.show()
            plt.savefig(f'vis/{img_id:05d}/{step_idx:03d}')
            plt.close()
if not SUBMISSION:
    net.attn.layers[0].self_attn.register_forward_hook(print_attn)

if SUBMISSION:
    f = open('sub.csv', 'w')
    f.write('id,label\r\n')
for img_id, (img, _id) in enumerate(test_dataloader):
    if img_id == nimgs: break

    print(f'Working on {img_id}...')
    # os.system(f'mkdir -p vis/{img_id:05d}')

    x = net(img.to(dev)).detach().cpu()
    # print(f'[{img_id}] Pred: {test_dataset.cid2name[torch.max(x, dim=1)[1].item()]}, \
    #                    Truth:{test_dataset.cid2name[lbl.item()]}')
    # print(f'[{_id}] Pred: {test_dataset.cid2name[torch.max(x, dim=1)[1].item()]}')

    if SUBMISSION:
        for __id, _x in zip(_id, x):
            f.write(f'{__id},{test_dataset.cid2name[torch.max(_x, dim=-1)[1].item()]}\r\n')
if SUBMISSION:
    f.close()

# c = 0
# tbar = tqdm(enumerate(test_dataset))
# for img_id, (img, lbl) in tbar:
#     out, _ = net(img.unsqueeze(0).to(dev))
#     out = out.cpu()
#     _, pred = torch.max(out, dim=1)
#     c += (lbl == pred).cpu().item()
#     tbar.set_description(f'Correct: {c}/{img_id+1}')