import argparse
import torch

from torch.utils.data import DataLoader

from dataset import CustomData
from model import BaseModel
from train import train, validate, infer


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--val_epoch', type=int, default=10)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    # prepare GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # prepare dataloader
    train_loader = DataLoader(dataset=CustomData('train'),
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)

    val_loader = DataLoader(dataset=CustomData('val'),
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)

    test_loader = DataLoader(dataset=CustomData('test'),
                             batch_size=1,
                             num_workers=args.num_workers,
                             shuffle=False)
    
    # prepare model
    network = BaseModel().to(device)

    # training
    # prepare optimizer

    # prepare criterion(s)

    # start training
    for e in range(1, args.epoch + 1):
        # train
        if e % args.val_epoch == 0:
            # validation
            pass
