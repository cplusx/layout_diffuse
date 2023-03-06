import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from data.random_sampling import RandomNoise
from model_utils import default, get_obj_from_str
from callbacks.coco_layout.sampling_save_fig import ColorMapping, plot_bbox_without_overlap, plot_bounding_box
import cv2

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

def postprocess_image(batched_x, batched_bbox, class_id_to_name, image_callback=lambda x: x):
    x = batched_x[0]
    bbox = batched_bbox[0]
    x = x.permute(1, 2, 0).detach().cpu().numpy().clip(-1, 1)
    x = (x + 1) / 2
    x = image_callback(x)
    image_with_bbox = overlap_image_with_bbox(x, bbox, class_id_to_name)
    canvas_with_bbox = overlap_image_with_bbox(np.ones_like(x), bbox, class_id_to_name)
    return x, image_with_bbox, canvas_with_bbox
        
def overlap_image_with_bbox(image, bbox, class_id_to_name):
    label_color_mapper = ColorMapping(id_class_mapping=class_id_to_name)
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

def generate_completion(caption, api_key, additional_caption=''):
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
    prompt = f'Describe a {additional_caption} scene with following objects: ' + caption + '. Use the above words to generate a prompt for drawing with a diffusion model. Use at least 30 words and at most 80 words and include all given objects. The final image should looks nice and be related to the given words and tags.'
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{
            "role": "user", 
            "content": prompt
        }]
    )

    return response.choices[0].message.content.strip() + additional_caption

def concatenate_class_labels_to_caption(objects, class_id_to_name, api_key=None, additional_caption=''):
    # if want to add additional description for styles, add it to additonal_caption
    caption = ''
    for i in objects:
        caption += class_id_to_name[i[4]+1] + ', '
    caption = caption.rstrip(', ')
    if api_key is not None:
        caption = generate_completion(caption, api_key=api_key, additional_caption=additional_caption)
        print('INFO: using openai text completion and the generated caption is: \n', caption)
    else:
        print('INFO: using default caption: \n', caption)
    return caption

def sample_one_image(bbox_path, ddpm_model, device, class_name_to_id, class_id_to_name, api_key=None, image_size=(512, 512), additional_caption=''):
    # the format of text file is: x, y, w, h, class_id
    with open(bbox_path, 'r') as IN:
        raw_objects = [i.strip().split(',') for i in IN]
    objects = []
    for i in raw_objects:
        i[0] = float(i[0])
        i[1] = float(i[1])
        i[2] = float(i[2])
        i[3] = float(i[3])
        class_name = i[4].strip()
        if class_name in class_name_to_id:
            # remove objects that are not in coco, these objects have class id but not appear in coco
            i[4] = int(class_name_to_id[class_name]) - 1
            objects.append(i)
    if len(objects) == 0:
        return None, None, None
    batch = []
    image_resizer = ImageResizer()
    new_h, new_w = image_resizer.get_proper_size(image_size)
    batch.append(torch.randn(1, 3, new_h, new_w).to(device))
    batch.append(torch.from_numpy(np.array(objects)).to(device).unsqueeze(0))
    batch.append((
        concatenate_class_labels_to_caption(objects, class_id_to_name, api_key, additional_caption), 
    ))
    res = ddpm_model.test_step(batch, 0) # we pass a batch but only text and layout is used when sampling
    sampled_images = res['sampling']['model_output']
    return postprocess_image(sampled_images, batch[1], class_id_to_name, image_callback=lambda x: image_resizer.to_original_size(x))


class ImageResizer:
    def __init__(self):
        self.original_size = None

    def to_proper_size(self, img):
        # Get the new height and width that can be divided by 64
        new_h, new_w = self.get_proper_size(img.shape[:2])

        # Resize the image using OpenCV's resize function
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return resized

    def to_original_size(self, img):
        # Resize the image to original size using OpenCV's resize function
        resized = cv2.resize(img, (self.original_size[1], self.original_size[0]), interpolation=cv2.INTER_AREA)

        return resized

    def get_proper_size(self, size):
        self.original_size = size
        # Calculate the new height and width that can be divided by 64
        if size[0] % 64 == 0:
            new_h = size[0]
        else:
            new_h = size[0] + (64 - size[0] % 64)

        if size[1] % 64 == 0:
            new_w = size[1]
        else:
            new_w = size[1] + (64 - size[1] % 64)

        return new_h, new_w

def parse_test_args():
    import argparse
    import json
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
    parser.add_argument(
        '--model_path', type=str,
        default=None, help='model path for generating layout diffuse, if not provided, will use the latest.ckpt')
    parser.add_argument(
        '--additional_caption', type=str,
        default='', help='additional caption for the generated image')

    ''' parser configs '''
    args_raw = parser.parse_args()
    with open(args_raw.config, 'r') as IN:
        args = json.load(IN)
    args.update(vars(args_raw))
    return args

def load_test_models(args):
    from train_utils import get_models, get_DDPM
    models = get_models(args)

    diffusion_configs = args['diffusion']
    ddpm_model = get_DDPM(
        diffusion_configs=diffusion_configs,
        log_args=args,
        **models
    )
    return ddpm_model

def load_model_weights(ddpm_model, args):
    print('INFO: loading checkpoint')
    if args['model_path'] is not None:
        ckpt_path = args['model_path']
    else:
        expt_path = os.path.join(args['expt_dir'], args['expt_name'])
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