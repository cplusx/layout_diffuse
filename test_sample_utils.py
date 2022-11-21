from torch.utils.data import DataLoader
from train_sample_utils import get_models, get_DDPM
from data import get_dataset
from data.random_sampling import RandomNoise
from model_utils import default, get_obj_from_str
from callbacks.sampling_save_fig import BasicImageSavingCallback

def get_test_dataset(args):
    sampling_args = args['sampling_args']
    sampling_w_noise = default(sampling_args.get('sampling_w_noise'), False)
    if sampling_w_noise:
        test_dataset = RandomNoise(
            sampling_args['image_size'],
            sampling_args['image_size'],
            sampling_args['in_channel'],
            sampling_args['num_samples']
        )
    else:
        from data import get_dataset
        args['data']['val_args']['data_len'] = sampling_args['num_samples']
        _, test_dataset = get_dataset(**args['data'])
    test_loader = DataLoader(test_dataset, batch_size=args['data']['batch_size'], num_workers=4, shuffle=False)
    return test_dataset, test_loader

def get_test_callbacks(args, expt_path):
    sampling_args = args['sampling_args']
    callbacks = []
    callbacks_obj = sampling_args.get('callbacks')
    for target in callbacks_obj:
        callbacks.append(
            get_obj_from_str(target)(expt_path)
        )
    return callbacks
        # if args['condition']:
        #     from callbacks.celeb_mask.sampling_save_fig import CelebMaskConditionalImageSavingCallback
        #     callbacks.append(
        #         CelebMaskConditionalImageSavingCallback(expt_path)
        #     )
        # else:
        #     from callbacks.celeb_mask.sampling_save_fig import CelebMaskUnconditionalImageSavingCallback
        #     callbacks.append(
        #         CelebMaskUnconditionalImageSavingCallback(expt_path)
        #     )
        # [BasicImageSavingCallback(expt_path)]
    # elif expt_name in [
    #     'celeb_mask_palette_cond_image', 
    #     'celeb_mask_palette_cond_mask', 
    # ]:
    #     from callbacks.palette.sampling_save_fig import CelebMaskPaletteImageSavingCallback
    #     callbacks.append(
    #         CelebMaskPaletteImageSavingCallback(expt_path, condition=args['data']['condition'])
    #     )

    # elif expt_name in ['celeb_mask_joint_mask_embedding']:
    #     # ========== later move to config, also add iterative refinement to image save callbacks =========
    #     ddpm_model.condition = args['condition'] # set ddpm
    #     if args['condition']:
    #         from data import get_dataset
    #         args['data']['val_args']['data_len'] = -1 # use all images
    #         _, test_dataset = get_dataset(**args['data'])
    #     else:
    #         test_dataset = RandomNoise(
    #             args['data']['image_size'], 
    #             args['data']['image_size'], 
    #             denoise_args['in_channel'],
    #             args['num_samples']
    #         )