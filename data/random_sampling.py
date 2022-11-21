import torch
from torch.utils.data import Dataset

class RandomNoise(Dataset):
    def __init__(self, h, w, dim, length=500):
        self.h = h
        self.w = w
        self.dim = dim
        self.length = length
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.randn(self.dim, self.h, self.w)