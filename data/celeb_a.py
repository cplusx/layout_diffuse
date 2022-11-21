import os
import numpy as np
import cv2
import torch
import albumentations as A
from torchvision.datasets.vision import VisionDataset

def get_train_transform(image_size):
    train_transform = A.Compose([
        A.PadIfNeeded(min_height=image_size, min_width=image_size),
        A.Resize(width=image_size, height=image_size),
    ])
    return train_transform

def get_test_transform(image_size):
    test_transform = A.Compose([
        A.PadIfNeeded(min_height=image_size, min_width=image_size),
        A.Resize(width=image_size, height=image_size),
    ])
    return test_transform


class CelebA(VisionDataset):
    def __init__(
        self, root, split='train', data_len=-1,
        transform=None, target_transform=None,
    ):
        '''
        root=/home/ubuntu/disk2/data/face/CelebA
        '''
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert split in ['train', 'val', 'test'], f'got {split}'
        self.split = split
        self.img_dir = os.path.join(root, 'img_align_celeba')
        if split == 'train':
            target_idx = '0'
        elif split == 'val':
            target_idx = '1'
        else:
            target_idx = '2'

        self.keys = []
        with open(os.path.join(root, 'list_eval_partition.txt')) as IN:
            for l in IN:
                i, j = l.strip().split(' ')
                if j == target_idx:
                    self.keys.append(i)
        if data_len > 0:
            self.keys = self.keys[:data_len]

    def _load_image(self, image_name):
        image_path = os.path.join(
            self.img_dir, 
            f'{image_name}'
        )
        image = cv2.imread(image_path)[...,::-1]
        return image
            
    def _flip(self, image):
        if self.split == 'train' and np.random.rand() < 0.5:
            image = torch.flip(image, dims=[1])
        return image

    def __getitem__(self, index):
        this_key = self.keys[index]
        image = self._load_image(this_key)

        transformed = self.transform(image=image)
        image = torch.from_numpy(transformed['image'])

        image = self._flip(image)
        
        image = (image).to(torch.float) / 255.
        # h, w, dim -> dim, h, w
        image = image.permute(2, 0, 1)

        image = (image - 0.5) * 2

        return image

    def __len__(self):
        """Return the number of images."""
        return len(self.keys)