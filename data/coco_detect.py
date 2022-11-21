import os
import numpy as np
import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms
import albumentations as A
import cv2

def get_coco_id_mapping(
    instance_path="/home/ubuntu/disk2/data/COCO/annotations/instances_val2017.json",
    stuff_path="/home/ubuntu/disk2/data/COCO/annotations/stuff_val2017.json", 
    subset_index=None
):
    import json
    def load_one_file(file_path):
        with open(file_path, 'r') as IN:
            data = json.load(IN)
        id_mapping = {}
        for item in data['categories']:
            item_id = item['id']
            item_name = item['name']
            id_mapping[item_id] = item_name
        if subset_index is not None:
            id_mapping = {id_mapping[i] for i in subset_index}
        return id_mapping
    instance_mapping = load_one_file(instance_path)
    stuff_mapping = load_one_file(stuff_path)

    instance_mapping.update(stuff_mapping)
    return instance_mapping

# def get_transforms(image_size):
#     transform = transforms.Compose([
#                 transforms.Resize((image_size, image_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     return transform
# see https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/ for more details for transforming bounding boxes
def get_train_transforms(image_size):
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.5, 1), ratio=(0.95, 1.05))
    ])
    return train_transform
# , bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.3, label_fields=['class_labels']))

def get_test_transforms(image_size):
    test_transform = A.Compose([
        A.SmallestMaxSize(max_size=image_size),
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(1, 1), ratio=(1, 1))
    ])
    return test_transform


def get_train_transforms_bbox(image_size):
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=image_size),
        A.HorizontalFlip(p=0.5),
        # A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.5, 1), ratio=(0.95, 1.05))
        # A.RandomSizedBBoxSafeCrop(height=image_size, width=image_size)
        A.RandomSizedCrop(min_max_height=[image_size//2, image_size], height=image_size, width=image_size)
    ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.3)) # , label_fields=['class_labels']
    return train_transform

def get_test_transforms_bbox(image_size):
    test_transform = A.Compose([
        A.SmallestMaxSize(max_size=image_size),
        # A.RandomResizedCrop(height=image_size, width=image_size, scale=(1, 1), ratio=(1, 1))
        # A.RandomSizedBBoxSafeCrop(height=image_size, width=image_size)
        A.RandomSizedCrop(min_max_height=[image_size, image_size], height=image_size, width=image_size)
    ], bbox_params=A.BboxParams(format='coco', min_area=128, min_visibility=0.3))
    return test_transform

class CocoImageOnly(CocoDetection):
    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)

        image = np.array(image).astype(np.float32) / 255.
        transformed = self.transform(image=image)

        image = torch.from_numpy(transformed['image'])
        image = image.permute(2, 0, 1)
        image = (image - 0.5) * 2

        return image

class CocoDetectionBboxOnly(CocoDetection):
    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        bboxes = []
        image = np.array(image).astype(np.float32) / 255.
        for items in target:
            category_id = items['category_id'] - 1 # so that category_id starts from 0
            bbox = items['bbox']
            x_min, y_min, width, height = bbox
            if width <= 0:
                width = 1
            if height <= 0:
                height = 1
            bbox = [x_min, y_min, width, height, category_id]
            bboxes.append(bbox)

        transformed = self.transform(image=image, bboxes=bboxes)
        image = transformed['image']
        bboxes = transformed['bboxes']

        h, w, _ = image.shape
        image = torch.from_numpy(transformed['image'])
        image = image.permute(2, 0, 1)
        image = (image - 0.5) * 2

        bbox_anno = []
        for bbox in bboxes:
            x_min, y_min, width, height, category_id = bbox
            bbox = [x_min/w, y_min/h, width/w, height/h, category_id]
            bbox_anno.append(bbox)

        return image, torch.tensor(bbox_anno).view(len(bbox_anno), 5)

def get_coco_dataset(root="/home/ubuntu/disk2/data/COCO", image_size=256):
    train_set = CocoImageOnly(
        root=os.path.join(root, 'train2017'), 
        annFile=os.path.join(root, 'annotations', 'instances_train2017.json'),
        transform=get_train_transforms(image_size)
    )
    val_set = CocoImageOnly(
        root=os.path.join(root, 'val2017'), 
        annFile=os.path.join(root, 'annotations', 'instances_val2017.json'),
        transform=get_test_transforms(image_size)
    )
    return train_set, val_set


def get_coco_detect_dataset(root="/home/ubuntu/disk2/data/COCO", image_size=128):
    train_set = CocoDetectionBboxOnly(
        root=os.path.join(root, 'train2017'), 
        annFile=os.path.join(root, 'annotations', 'instances_train2017.json'),
        transform=get_train_transforms_bbox(image_size)
    )
    val_set = CocoDetectionBboxOnly(
        root=os.path.join(root, 'val2017'), 
        annFile=os.path.join(root, 'annotations', 'instances_val2017.json'),
        transform=get_test_transforms_bbox(image_size),
    )
    return train_set, val_set