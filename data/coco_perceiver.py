import os
import numpy as np
import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms

def get_whole_image_transforms(image_size):
    transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform

def get_sub_image_transforms(image_size):
    transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform

class CocoDetectionSubImages(CocoDetection):
    def __init__(
        self, 
        root: str, 
        annFile: str, 
        transform = None, 
        target_transform = None, 
        transforms = None,
        sub_image_transform = None,
    ) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.sub_image_transform = sub_image_transform
        
    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        h, w, _ = np.array(image).shape
        sub_images = []
        bboxes = []
        for items in target:
            bbox = items['bbox']
            x_min, y_min, width, height = bbox
            bbox = [x_min/w, y_min/h, width/w, height/h]
            this_sub_image = image.crop((x_min, y_min, x_min+width, y_min+height)) # left, top, right, bottom
            sub_images.append(this_sub_image)
            bboxes.append(bbox)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        sub_images = list(map(self.sub_image_transform, sub_images))
        if len(sub_images) == 0:
            pass
        else:
            sub_images = torch.stack(sub_images, dim=0)[:30]
            bboxes = torch.tensor(bboxes)[:30]
        
        return image, sub_images, bboxes

def get_coco_perceiver_dataset(root="/home/ubuntu/disk2/data/COCO", image_size=128, sub_image_size=128):
    train_set = CocoDetectionSubImages(
        root=os.path.join(root, 'train2017'), 
        annFile=os.path.join(root, 'annotations', 'instances_train2017.json'),
        transform=get_whole_image_transforms(image_size),
        sub_image_transform=get_sub_image_transforms(sub_image_size)
    )
    val_set = CocoDetectionSubImages(
        root=os.path.join(root, 'val2017'), 
        annFile=os.path.join(root, 'annotations', 'instances_val2017.json'),
        transform=get_whole_image_transforms(image_size),
        sub_image_transform=get_sub_image_transforms(sub_image_size)
    )
    return train_set, val_set