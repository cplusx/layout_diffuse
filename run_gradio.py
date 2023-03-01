import gradio as gr
import os
import torch
import json
from train_sample_utils import get_models, get_DDPM
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from callbacks.coco_layout.sampling_save_fig import ColorMapping, plot_bbox_without_overlap, plot_bounding_box
from data.coco_w_stuff import get_coco_id_mapping
import numpy as np

def get_sampled_image_and_bbox(batched_x, batched_bbox):
    sampled_images = []
    bbox_images = []
    white_canvas_images = []
    for x, bbox in zip(batched_x, batched_bbox):
        x = x.permute(1, 2, 0).detach().cpu().numpy().clip(-1, 1)
        x = (x + 1) / 2
        sampled_images.append(x)
        bbox_image = overlap_image_with_bbox(x, bbox)
        if bbox_image is not None:
            bbox_images.append(bbox_image)
        white_canvas_images.append(overlap_image_with_bbox(np.ones_like(x), bbox))
    return (sampled_images, bbox_images, white_canvas_images)

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

coco_id_mapping = get_coco_id_mapping()
args_raw = {
    'config': 'configs/cocostuff_SD2_1.json',
    'epoch': 9,
    'nnode': 1
}

with open(args_raw['config'], 'r') as IN:
    args = json.load(IN)
args.update(args_raw)
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
from test_sample_utils import get_test_dataset
test_dataset, test_loader = get_test_dataset(args)

'''4. load checkpoint'''
print('INFO: loading checkpoint')
if args['epoch'] is None:
    ckpt_to_use = 'latest.ckpt'
else:
    ckpt_to_use = f'epoch={args["epoch"]:04d}.ckpt'
ckpt_path = os.path.join(expt_path, ckpt_to_use)
if os.path.exists(ckpt_path):
    print(f'INFO: Found checkpoint {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
    ddpm_model.load_state_dict(ckpt)
else:
    ckpt_path = None
    raise RuntimeError('Cannot do inference without pretrained checkpoint')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ddpm_model = ddpm_model.to(device)


def sample_images(index, preview):
    index = int(index)
    assert integer_validator(index)
    if not preview:
        batch = list(test_dataset[index])
        batch[0] = batch[0].to(device).unsqueeze(0)
        batch[1] = batch[1].to(device).unsqueeze(0)
        res = ddpm_model.test_step(batch, 0)
        sampled_images = res['sampling']['model_output']
        images = get_sampled_image_and_bbox(sampled_images, batch[1].cpu())
        img = images[0][0]
        bbox_img = images[1][0]
        white_canvas_img = images[2][0]
        result_img = np.concatenate((img, bbox_img, white_canvas_img), axis=1)
        return result_img
    else:
        batch = list(test_dataset[index])
        raw_image = batch[0].to(device).unsqueeze(0)
        batch[1] = batch[1].to(device).unsqueeze(0)
        images = get_sampled_image_and_bbox(raw_image, batch[1].cpu())
        white_canvas_img = images[2][0]
        return white_canvas_img

def integer_validator(x):
    try:
        if x < 0:
            raise ValueError("Input must be an integer greater than 0.")
        return True
    except:
        raise ValueError("Input must be an integer greater than 0.")


output_component = gr.outputs.Image(type="numpy")
gr.Interface(
    sample_images, 
    inputs=[
        gr.inputs.Textbox(default="1", label="Enter an index greater than 0"),
        gr.inputs.Checkbox(default=False, label="Preview image")
    ],
    outputs=output_component, 
    capture_session=True, 
    title="LayoutDiffuse", 
    description="Generate sample images using the DDPM model",
    allow_flagging=False,
    live=False
).launch(share=True)