import argparse 
import yaml
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils import data
from tqdm import tqdm
from torchnet import meter

from datasets.mnist import MNISTDataset
from datasets.cifar10 import CIFAR10Dataset
from models.ClAtNet import ClAtNet
from models.toymodel import ToyModel
from workers.trainer import Trainer
from metrics.accuracy import Accuracy
from utils.random import set_seed

def train(config):
    assert config is not None, "Do not have config file!"

    device = torch.device('cuda:{}'.format(config['gpus']) 
                          if config.get('gpus', None) is not None 
                             and torch.cuda.is_available() 
                          else 'cpu')

    # 1: Load datasets
    set_seed()

    '''CIFAR10'''
    dataset = CIFAR10Dataset('data/CIFAR10/train', 
                             'data/CIFAR10/trainLabels.csv')
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, 
                                                               [len(dataset) - len(dataset) // 5, 
                                                                len(dataset) // 5])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    '''MNIST'''
    # train_dataset = MNISTDataset('data/MNIST/mnist_train.csv')
    # train_dataloader = DataLoader(train_dataset, shuffle=True, 
    #                               num_workers=4, batch_size=32)

    # val_dataset = MNISTDataset('data/MNIST/mnist_test.csv')
    # val_dataloader = DataLoader(val_dataset, batch_size=1,
    #                             num_workers=4)

    # 2: Define network
    set_seed()
    net = ClAtNet(in_channels=3, size=(32, 32),
                  nfeatures=64, nclasses=10, nheads=2, dropout=0.0).to(device)
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
    parser.add_argument('--config')
    parser.add_argument('--gpus', default=None)

    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config['gpus'] = args.gpus

    train(config)