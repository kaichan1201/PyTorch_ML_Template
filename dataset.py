import torch
import torchvision
from torch.utils.data import Dataset

class CustomData(Dataset):
    def __init__(self, mode, dir_path):
        self.mode = mode
        self.dir_path = dir_path

        # initialize corresponding dataset & transforms
        if mode == 'train':
            pass
        elif mode == 'val':
            pass
        elif mode == 'test':
            pass
        else:
            raise NotImplementedError

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
