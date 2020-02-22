import argparse 
import json 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils import data
from tqdm import tqdm
from torchnet import meter

from datasets.mnist import MNISTDataset
from models.ClAtNet import ClAtNet
from models.toymodel import ToyModel
from workers.trainer import Trainer
from metrics.accuracy import Accuracy

def train(config):
    assert config is not None, "Do not have config file!"

    device = torch.device('cuda:{}'.format(config['gpus']) 
                          if config.get('gpus', False) 
                             and torch.cuda.is_available() 
                          else 'cpu')

    # 1: Load datasets
    train_dataset = MNISTDataset('data/MNIST/mnist_train.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=32)

    val_dataset = MNISTDataset('data/MNIST/mnist_test.csv')
    val_dataloader = DataLoader(val_dataset, batch_size=1)

    # 2: Define network
    net = ClAtNet(nfeatures=128, nclasses=10, nheads=4, dropout=0.1).to(device)
    print(net)
    # 3: Define loss
    criterion = nn.CrossEntropyLoss()
    # 4: Define Optimizer
    optimizer = torch.optim.Adam(net.parameters())
    # 5: Define metrics
    metric = Accuracy()

    # 6: Create trainer
    trainer = Trainer(device = device,
                      config = config,
                      net = net,
                      criterion = criterion,
                      optimier = optimizer,
                      metric = metric)
    # 7: Start to train
    trainer.train(train_dataloader=train_dataloader, 
                  val_dataloader=val_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--cfg', default='config.json')

    args = parser.parse_args()
    config_path = args.cfg
    config = json.load(open(config_path, 'r'))
    config['gpus'] = args.gpus

    train(config)