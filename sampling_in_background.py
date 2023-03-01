import os
import torch
import argparse
import json
from train_sample_utils import get_models, get_DDPM
from callbacks.coco_layout.sampling_save_fig import ColorMapping, plot_bbox_without_overlap, plot_bounding_box
from data.coco_w_stuff import get_coco_id_mapping
import numpy as np
import cv2
import time
coco_id_mapping = get_coco_id_mapping()
coco_name_to_id = {v: int(k) for k, v in coco_id_mapping.items()}
def postprocess_image(batched_x, batched_bbox):
    x = batched_x[0]
    bbox = batched_bbox[0]
    x = x.permute(1, 2, 0).detach().cpu().numpy().clip(-1, 1)
    x = (x + 1) / 2
    image_with_bbox = overlap_image_with_bbox(x, bbox)
    canvas_with_bbox = overlap_image_with_bbox(np.ones_like(x), bbox)
    return x, image_with_bbox, canvas_with_bbox
        

def overlap_image_with_bbox(image, bbox):
    label_color_mapper = ColorMapping(id_class_mapping=coco_id_mapping)
    image_with_bbox = plot_bbox_without_overlap(
        image.copy(),
        bbox,
        label_color_mapper
    ) if len(bbox) <= 10 else None
    if image_with_bbox is not None:
        return image_with_bbox
    return plot_bounding_box(
        image.copy(), 
        bbox,
        label_color_mapper
    )

def generate_completion(caption, api_key):
    import openai
    # check if api_key is valid
    def validate_api_key(api_key):
        import re
        regex = "^sk-[a-zA-Z0-9]{48}$" # regex pattern for OpenAI API key
        if not isinstance(api_key, str):
            return None
        if not re.match(regex, api_key):
            return None
        return api_key
    openai.api_key = validate_api_key(api_key)
    if openai.api_key is None:
        print('WARNING: invalid OpenAI API key, using default caption')
        return caption
    prompt = 'Describe an scene with following words: ' + caption + '. Use the above words to generate a prompt for drawing with a diffusion model. Use less than 120 words and include all given words. The final image should looks nice and be related to the given words.'
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].text.strip()

def concatenate_class_labels_to_caption(objects, api_key=None):
    caption = ''
    for i in objects:
        caption += coco_id_mapping[i[4]+1] + ', '
    caption = caption.rstrip(', ')
    if api_key is not None:
        caption = generate_completion(caption)
        print('INFO: using openai text completion and the generated caption is: \n', caption)
    return caption

def sample_one_image(file_path, ddpm_model, device, api_key=None):
    # the format of text file is: x, y, w, h, class_id
    with open(file_path, 'r') as IN:
        objects = [i.strip().split(',') for i in IN]
    for i in objects:
        i[0] = float(i[0])
        i[1] = float(i[1])
        i[2] = float(i[2])
        i[3] = float(i[3])
        i[4] = int(coco_name_to_id[i[4].strip()]) - 1
    batch = []
    batch.append(torch.randn(1, 3, 512, 512).to(device))
    batch.append(torch.from_numpy(np.array(objects)).to(device).unsqueeze(0))
    batch.append((concatenate_class_labels_to_caption(objects, api_key), ))
    res = ddpm_model.test_step(batch, 0) # we pass a batch but only text and layout is used when sampling
    sampled_images = res['sampling']['model_output']
    return postprocess_image(sampled_images, batch[1])

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
        # read file in the folder "flask_images_to_sample". If there is a file, sample the image and save it to the folder "flask_images_sampled" and remove the file from the folder "flask_images_to_sample"

        # check folder flask_images_to_sample/*.txt
        from glob import glob
        files_to_sample = glob('../UI_plotting/tmp/*.txt')
        for f in files_to_sample:
            print('INFO: professing file', f)
            image, image_with_bbox, canvas_with_bbox = sample_one_image(f, ddpm_model, device, )
            # save the image
            cat_image = np.concatenate([image, image_with_bbox, canvas_with_bbox], axis=1)
            cv2.imwrite(f.replace('.txt', '.jpg'), (cat_image[..., ::-1] * 255).astype(np.uint8))
            # remove the file
            os.remove(f)

        time.sleep(1)