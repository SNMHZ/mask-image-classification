import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import pandas as pd

from model import MyModel
from dataset import TrainDataset_01
from train import train_one_epoch


EPOCH = 10
lr = 0.001
device = torch.device('cuda')

TRAIN_DIR = '/opt/ml/input/data/train'
train_info = pd.read_csv(os.path.join(TRAIN_DIR, 'train_info_01.csv'))

image_paths = train_info['path'].values
labels = train_info['label'].values

def main():
    model = MyModel()
    optimizer = Adam(model.parameters(), lr=lr)
    train_dataset = TrainDataset_01(image_paths, labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(EPOCH):
        train_one_epoch(model, optimizer, train_loader, device)


if __name__ == "__main__":
    main()