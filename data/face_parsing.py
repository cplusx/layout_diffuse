import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import albumentations as A
import math
from PIL import Image
from torchvision.datasets.vision import VisionDataset

celebAMask_label_list = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
celebAMask_labels = {i: v for i, v in enumerate(celebAMask_label_list)}
flip_mapping = torch.tensor([-1] * len(list(celebAMask_labels.keys())), dtype=torch.long) 
for i, x in enumerate(celebAMask_labels.keys()):
    flip_mapping[x] = i
flip_mapping[4] = 5; flip_mapping[5] = 4
flip_mapping[6] = 7; flip_mapping[7] = 6
flip_mapping[8] = 9; flip_mapping[9] = 8

class MaskMeshConverter(torch.nn.Module):
    '''
    convert a segmentation mask to multiple channels using mesh grid
    '''
    def __init__(self, labels, mesh_dim=3):
        super().__init__()
        self.labels = labels
        num_grid_each_dim = math.ceil(len(labels)**(1/mesh_dim))
        mesh_d = torch.meshgrid(
            *[torch.linspace(0,1,num_grid_each_dim)]*mesh_dim
        )
        mesh_d = [i.reshape(-1) for i in mesh_d]
        self.mesh = torch.stack(mesh_d, dim=-1)
        self.mesh_embedding = torch.nn.Embedding(len(self.mesh), mesh_dim)
        self.mesh_embedding.weight.data = self.mesh

        # maps index in mask to index in mesh
        assert torch.tensor(labels).min() >= 0
        index_map = torch.tensor([-1] * (torch.tensor(labels).max() + 1), dtype=torch.int)
        reverse_index_map = torch.tensor([-1] * (torch.tensor(labels).max() + 1), dtype=torch.int)
        for i, x in enumerate(labels):
            index_map[x] = i
            reverse_index_map[i] = x
        self.register_buffer('index_map', index_map)
        self.register_buffer('reverse_index_map', reverse_index_map)
        
    def index_mask_to_nd_mesh(self, mask):
        mesh_idx_mask = self.index_map[mask]
        embedding = self.mesh_embedding(mesh_idx_mask).detach()
        return embedding

    def nd_mesh_to_index_mask(self, mesh):
        mesh_size = mesh.size()
        mesh = mesh.view(mesh_size[0], -1, mesh_size[-1]) # bs, hxw, mesh_dim
        mesh_dist_to_embedding = torch.cdist(
            mesh, 
            self.mesh_embedding.weight.data[None].expand(
                mesh_size[0], -1, -1
            ),
            p=1
        )
        mesh_nn = torch.argmin(
            mesh_dist_to_embedding[:, :, :len(self.labels)],
            dim=-1, keepdim=True
        ).view(*mesh_size[:-1]) # bs, h, w
        return self.reverse_index_map[mesh_nn]


    def __call__(self, mask):
        return self.index_mask_to_nd_mesh(mask)

class MaskOnehotConverter(torch.nn.Module):
    def __init__(self, labels):
        '''Two step mapping to handle incontinuous index case (e.g. 255 for ignore)'''
        super().__init__()
        self.num_classes = len(labels)
        # maps index in mask to index in one-hot
        assert torch.tensor(labels).min() >= 0
        index_map = torch.tensor([-1] * (torch.tensor(labels).max() + 1), dtype=torch.int)
        reverse_index_map = torch.tensor([-1] * (torch.tensor(labels).max() + 1), dtype=torch.int)
        for i, x in enumerate(labels):
            index_map[x] = i
            reverse_index_map[i] = x
        self.register_buffer('index_map', index_map)
        self.register_buffer('reverse_index_map', reverse_index_map)

    def index_mask_to_one_hot(self, mask):
        continous_idx_mask = self.index_map[mask].to(torch.long)
        one_hot_tensor = F.one_hot(continous_idx_mask, num_classes=self.num_classes)
        return one_hot_tensor.to(torch.float)

    def one_hot_to_index_mask(self, one_hot_tensor):
        # one_hot_tensor: b, dim, h, w
        continous_idx_mask = torch.argmax(one_hot_tensor, dim=1).to(torch.long)
        mask = self.reverse_index_map[continous_idx_mask]
        return mask

    def __call__(self, mask):
        return self.index_mask_to_one_hot(mask)

def get_train_transform(image_size):
    train_transform = A.Compose([
        A.PadIfNeeded(min_height=image_size, min_width=image_size),
        A.Resize(width=image_size, height=image_size, interpolation=cv2.INTER_AREA),
        # A.HorizontalFlip(p=0.5), # disable it since we need to modify left and right index for eyes, eyebrows and ears. Move this function to dataset.
        # A.RandomBrightnessContrast(p=0.2), # maybe not good for face generation?
    ])
    return train_transform

def get_test_transform(image_size):
    test_transform = A.Compose([
        A.PadIfNeeded(min_height=image_size, min_width=image_size),
        A.Resize(width=image_size, height=image_size, interpolation=cv2.INTER_AREA),
    ])
    return test_transform

class CelebAMaskHQ(VisionDataset):
    def __init__(
        self, root, split='train', data_len=-1,
        transform=None, target_transform=None,
        dual_transforms=None,
    ):
        '''
        root=/home/ubuntu/disk2/data/face/CelebAMask-HQ
        Remember to preprocess dataset with https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_mask.py
        '''
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert split in ['train', 'val'], f'got {split}'
        self.split = split
        self.img_dir = os.path.join(root, 'CelebA-HQ-img')
        self.mask_dir = os.path.join(root, 'CelebAMaskHQ-mask')
        with open(os.path.join(root, f'{split}.txt')) as IN:
            self.keys = [i.strip() for i in IN]
        if data_len > 0:
            self.keys = self.keys[:data_len]
        self.dual_transforms = dual_transforms

    def _load_image(self, image_name):
        image_path = os.path.join(
            self.img_dir, 
            f'{image_name}.jpg'
        )
        image = cv2.imread(image_path)[...,::-1]
        return image
            
    def _load_mask(self, image_name):
        image_path = os.path.join(
            self.mask_dir, 
            f'{image_name}.png'
        )
        image = np.array(Image.open(image_path))
        return image

    def _flip(self, image, mask):
        if self.split == 'train' and np.random.rand() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
            mask = flip_mapping[mask.to(torch.long)]
        return image, mask

    def _process_mask(self, mask):
        mask = torch.unsqueeze(mask, dim=-1)
        return mask.to(torch.float32) / 255.

    def __getitem__(self, index):
        this_key = self.keys[index]
        image = self._load_image(this_key)
        image = (image).astype(np.float32) / 255.
        mask = self._load_mask(this_key)

        transformed = self.dual_transforms(image=image, masks=[mask])
        image = torch.from_numpy(transformed['image'])
        mask = torch.from_numpy(transformed['masks'][0])

        # flip during training with correct mask index
        image, mask = self._flip(image, mask)
        
        mask = self._process_mask(mask)
        # h, w, dim -> dim, h, w
        image = image.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)

        image = (image - 0.5) * 2
        mask = (mask - 0.5) * 2

        ret = {}
        ret['image'] = image # return original image and mask for visualization
        ret['seg_mask'] = mask
        return ret

    def __len__(self):
        """Return the number of images."""
        return len(self.keys)

# class CelebAMaskHQPartial(CelebAMaskHQ):
#     def __init__(self, root, split='train', data_len=-1, transform=None, target_transform=None, dual_transforms=None, down_resolutions=[1,2,4,8,16,32]):
#         super().__init__(root, split, data_len, transform, target_transform, dual_transforms, convert_mask2mesh=False, mesh_dim=3, convert_mask2onehot=False, condition='image')
#         self.down_resolutions = down_resolutions

#     def _process_mask(self, mask):
#         return mask.to(torch.long)

#     def __getitem__(self, index):
#         this_key = self.keys[index]
#         raw_image = self._load_image(this_key)
#         raw_mask = self._load_mask(this_key)

#         transformed = self.dual_transforms(image=raw_image, masks=[raw_mask])
#         image = torch.from_numpy(transformed['image'])
#         mask = torch.from_numpy(transformed['masks'][0])

#         # flip during training with correct mask index
#         image, mask = self._flip(image, mask)
        
#         image = (image).to(torch.float) / 255.
#         mask = self._process_mask(mask)
#         # h, w, dim -> dim, h, w
#         image = image.permute(2, 0, 1)

#         image = (image - 0.5) * 2

#         ret = {}
#         ret['image'] = image # dim, h, w
#         ret['seg_mask'] = mask # h, w
#         ret['seg_mask_down'] = {}
#         for res in self.down_resolutions:
#             h, w = mask.shape
#             assert h % res == 0 and w % res == 0
#             this_mask = mask[res//2::res, res//2::res] # equal to downsample with nearest neighbour
#             ret['seg_mask_down'][f'{res}x'] = this_mask
#         return ret
