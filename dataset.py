import torch
import torchvision
from torch.utils.data import Dataset

class CustomData(Dataset):
    def __init__(self, mode, dir_path):
        self.mode = mode
        self.dir_path = dir_path

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
