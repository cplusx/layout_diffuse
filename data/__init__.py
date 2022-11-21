from model_utils import default

def get_dataset(**kwargs):
    dataset = kwargs['dataset']
    if dataset in ['celeb_mask', 'celeb_mask_partial']:
        from .face_parsing import CelebAMaskHQ, CelebAMaskHQPartial, get_train_transform, get_test_transform
        root = kwargs['root']
        image_size = kwargs['image_size']
        train_set = CelebAMaskHQ(
            root,
            dual_transforms=get_train_transform(image_size), 
            **kwargs['train_args']
        )
        val_set = CelebAMaskHQ(
            root,
            dual_transforms=get_test_transform(image_size),
            **kwargs['val_args']
        )
    elif dataset == 'coco_stuff_layout':
        from .coco_w_stuff import get_cocostuff_dataset
        root = kwargs['root']
        image_size = kwargs['image_size']
        train_set, val_set = get_cocostuff_dataset(
            root, image_size
        )
    elif dataset == 'coco_stuff_layout_caption':
        from .coco_w_stuff import get_cocostuff_caption_dataset
        root = kwargs['root']
        image_size = kwargs['image_size']
        train_set, val_set = get_cocostuff_caption_dataset(
            root, image_size, **kwargs['dataset_args']
        )
    elif dataset == 'coco_stuff_layout_caption_label':
        from .coco_w_stuff import get_cocostuff_caption_label_dataset
        root = kwargs['root']
        image_size = kwargs['image_size']
        train_set, val_set = get_cocostuff_caption_label_dataset(
            root, image_size, **kwargs['dataset_args']
        )
    elif dataset == 'vg_layout_label':
        from .vg import get_vg_caption_dataset
        root = kwargs['root']
        image_size = kwargs['image_size']
        train_set, val_set = get_vg_caption_dataset(
            root, image_size
        )
    else:
        raise NotImplementedError(f'got {dataset}')

    return train_set, val_set