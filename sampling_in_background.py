import os
import torch
import argparse
import json
from train_sample_utils import get_models, get_DDPM
from data.coco_w_stuff import get_coco_id_mapping
import numpy as np
import cv2
import time
from test_sample_utils import sample_one_image
coco_id_to_name = get_coco_id_mapping()
coco_name_to_id = {v: int(k) for k, v in coco_id_to_name.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, 
        default='config/train.json')
    parser.add_argument(
        '-e', '--epoch', type=int, 
        default=None, help='which epoch to evaluate, if None, will use the latest')
    parser.add_argument(
        '--openai_api_key', type=str,
        default=None, help='openai api key for generating text prompt')

    ''' parser configs '''
    args_raw = parser.parse_args()
    with open(args_raw.config, 'r') as IN:
        args = json.load(IN)
    args.update(vars(args_raw))
    expt_name = args['expt_name']
    expt_dir = args['expt_dir']
    expt_path = os.path.join(expt_dir, expt_name)
    os.makedirs(expt_path, exist_ok=True)

    '''1. create denoising model'''
    denoise_args = args['denoising_model']['model_args']
    models = get_models(args)

    diffusion_configs = args['diffusion']
    ddpm_model = get_DDPM(
        diffusion_configs=diffusion_configs,
        log_args=args,
        **models
    )

    '''4. load checkpoint'''
    print('INFO: loading checkpoint')
    if args['epoch'] is None:
        ckpt_to_use = 'latest.ckpt'
    else:
        ckpt_to_use = f'epoch={args["epoch"]:04d}.ckpt'
    ckpt_path = os.path.join(expt_path, ckpt_to_use)
    print(ckpt_path)
    if os.path.exists(ckpt_path):
        print(f'INFO: Found checkpoint {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
        ddpm_model.load_state_dict(ckpt)
    else:
        ckpt_path = None
        raise RuntimeError('Cannot do inference without pretrained checkpoint')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ddpm_model = ddpm_model.to(device)
    ddpm_model.text_fn = ddpm_model.text_fn.to(device)
    ddpm_model.text_fn.device = device
    ddpm_model.denoise_fn = ddpm_model.denoise_fn.to(device)
    ddpm_model.vqvae_fn = ddpm_model.vqvae_fn.to(device)

    while True:
        # read file in the folder. If there is a file, sample the image and save it to the folder "flask_images_sampled" and remove the file from the folder "flask_images_to_sample"

        from glob import glob
        files_to_sample = glob('interactive_plotting/tmp/*.txt')
        for f in files_to_sample:
            print('INFO: processing file', f)
            image, image_with_bbox, canvas_with_bbox = sample_one_image(f, ddpm_model, device, class_name_to_id=coco_name_to_id, class_id_to_name=coco_id_to_name, api_key=args['openai_api_key'])
            # save the image
            cat_image = np.concatenate([image, image_with_bbox, canvas_with_bbox], axis=1)
            cv2.imwrite(f.replace('.txt', '.jpg'), (cat_image[..., ::-1] * 255).astype(np.uint8))
            # remove the file
            os.remove(f)

        time.sleep(1)