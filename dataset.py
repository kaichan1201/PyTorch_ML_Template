import torch
import torchvision
from torch.utils.data import Dataset

class CustomData(Dataset):
    def __init__(self, mode):
        self.mode = mode

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
