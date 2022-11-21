import os
import numpy as np
import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms
import albumentations as A
import cv2

VEHICLE_IDS = list(range(2, 10))
ANIMAL_IDS = list(range(16, 26))
def get_coco_subset_id_mapping(subset='animal'):
    from .coco_detect import get_coco_id_mapping
    if subset == 'animal':
        return get_coco_id_mapping(ANIMAL_IDS)
    elif subset == 'vehicle':
        return get_coco_id_mapping(VEHICLE_IDS)
    else:
        raise RuntimeError(f'Got {subset}')
# see https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/ for more details for transforming bounding boxes

def get_train_transforms_bbox(image_size):
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomSizedCrop(min_max_height=[image_size//2, image_size], height=image_size, width=image_size)
    ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.3)) # , label_fields=['class_labels']
    return train_transform

def get_test_transforms_bbox(image_size):
    test_transform = A.Compose([
        A.SmallestMaxSize(max_size=image_size),
        A.Resize(image_size, image_size)
        # A.RandomSizedCrop(min_max_height=[image_size, image_size], height=image_size, width=image_size)
    ], bbox_params=A.BboxParams(format='coco', min_area=128, min_visibility=0.3))
    return test_transform


class CocoDetectionBboxOnly(CocoDetection):
    '''
    Note, the subset_ids starts from 1 (as in coco), the returned ids starts from 0 (for training)
    '''
    def __init__(self, subset_ids=[], min_area=0.2, subset_id_file=None, **kwargs):
        super().__init__(**kwargs)
        self.min_area = min_area
        self.subset_ids = subset_ids
        if os.path.exists(subset_id_file):
            with open(subset_id_file, 'r') as IN:
                self.ids = [int(l.strip()) for l in IN]
        else:
            self.filter_ids(subset_ids)
            self.save_id_to_text_file(subset_id_file)

    def save_id_to_text_file(self, subset_id_file):
        with open(subset_id_file, 'w') as OUT:
            OUT.write('\n'.join(list(map(str, self.ids))))

    def filter_ids(self, subset_ids):
        from tqdm import tqdm
        num_old_ids = len(self.ids)
        new_ids = []
        for coco_id in tqdm(self.ids):
            image = np.array(self._load_image(coco_id))
            h, w = image.shape[:2]
            image_area = h * w
            target = self._load_target(coco_id)
            for items in target:
                category_id = items['category_id']
                item_area = items['area']
                if (category_id in subset_ids) and (item_area > (image_area * self.min_area)):
                    new_ids.append(coco_id)
                    break
        self.ids = new_ids
        print(f'INFO: filtered {len(self.ids)} from {num_old_ids} images')

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        bboxes = []
        image = np.array(image).astype(np.float32) / 255.
        h, w = image.shape[:2]
        image_area = h * w
        for items in target:
            category_id = items['category_id']
            item_area = items['area']
            if (category_id in self.subset_ids) and (item_area > (image_area * self.min_area)):
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


def get_coco_subset_detect_dataset(root="/home/ubuntu/disk2/data/COCO", image_size=128, subset_type='animal', min_area=0.2):
    subset_id_file_train = os.path.join(root, f'train_{subset_type}_ids_min_area_{min_area}.txt')
    subset_id_file_test = os.path.join(root, f'test_{subset_type}_ids_min_area_{min_area}.txt')
    if subset_type == 'animal':
        subset_ids = ANIMAL_IDS
    elif subset_type == 'vehicle':
        subset_ids = VEHICLE_IDS

    train_set = CocoDetectionBboxOnly(
        root=os.path.join(root, 'train2017'), 
        annFile=os.path.join(root, 'annotations', 'instances_train2017.json'),
        subset_ids=subset_ids,
        min_area=min_area,
        subset_id_file=subset_id_file_train,
        transform=get_train_transforms_bbox(image_size)
    )
    val_set = CocoDetectionBboxOnly(
        root=os.path.join(root, 'val2017'), 
        annFile=os.path.join(root, 'annotations', 'instances_val2017.json'),
        subset_ids=subset_ids,
        min_area=min_area,
        subset_id_file=subset_id_file_test,
        transform=get_test_transforms_bbox(image_size)
    )
    return train_set, val_set