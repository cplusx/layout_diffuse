import torch
from torch.utils.data import Dataset

class Dummy(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 100

    def __getitem__(self, index):
        return torch.zeros([3, 256, 256]), torch.tensor([8191]*90).view(-1)