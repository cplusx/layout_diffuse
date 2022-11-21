from model_utils import default

def get_dataset(**kwargs):
    dataset = kwargs['dataset']
    if dataset == 'dummy':
        from .dummy import Dummy
        train_set = Dummy()
        val_set = Dummy()
    elif dataset in ['celeb_mask', 'celeb_mask_partial']:
        from .face_parsing import CelebAMaskHQ, CelebAMaskHQPartial, get_train_transform, get_test_transform
        root = kwargs['root']
        image_size = kwargs['image_size']
        if dataset == 'celeb_mask':
            convert_mask2mesh = kwargs.get('convert_mask2mesh') or False
            convert_mask2onehot = kwargs.get('convert_mask2onehot') or False
            condition = kwargs.get('condition') or 'image'
            train_set = CelebAMaskHQ(
                root,
                dual_transforms=get_train_transform(image_size), 
                convert_mask2mesh=convert_mask2mesh,
                convert_mask2onehot=convert_mask2onehot,
                condition=condition,
                **kwargs['train_args']
            )
            val_set = CelebAMaskHQ(
                root,
                dual_transforms=get_test_transform(image_size),
                convert_mask2mesh=convert_mask2mesh,
                convert_mask2onehot=convert_mask2onehot,
                condition=condition,
                **kwargs['val_args']
            )
        elif dataset == 'celeb_mask_partial':
            down_resolutions = kwargs.get('down_resolutions') or [1,2,4,8,16,32]
            train_set = CelebAMaskHQPartial(
                root,
                dual_transforms=get_train_transform(image_size), 
                down_resolutions=down_resolutions,
                **kwargs['train_args']
            )
            val_set = CelebAMaskHQPartial(
                root,
                dual_transforms=get_test_transform(image_size),
                down_resolutions=down_resolutions,
                **kwargs['val_args']
            )
    elif dataset == 'coco':
        from .coco_detect import get_coco_dataset
        root = kwargs['root']
        image_size = kwargs['image_size']
        train_set, val_set = get_coco_dataset(
            root, image_size
        )
    elif dataset == 'coco_layout':
        from .coco_detect import get_coco_detect_dataset
        root = kwargs['root']
        image_size = kwargs['image_size']
        train_set, val_set = get_coco_detect_dataset(
            root, image_size
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
    elif dataset == 'coco_layout_subset':
        from .coco_detect_subset import get_coco_subset_detect_dataset
        root = kwargs['root']
        image_size = kwargs['image_size']
        subset_type = default(kwargs.get('subset_type'), 'animal')
        min_area = default(kwargs.get('min_area'), 0.2)
        train_set, val_set = get_coco_subset_detect_dataset(
            root, image_size, subset_type=subset_type, min_area=min_area
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