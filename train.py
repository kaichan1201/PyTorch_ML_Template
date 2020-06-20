import torch
from tqdm import tqdm

def train(model, train_loader, criterion, optimizer):
    model.train()
    for idx, (data, gt) in tqdm(enumerate(train_loader)):
        out = model(data)

def validate(model, val_loader, criterion, optimizer):
    pass

def infer(model, test_loader):
    pass