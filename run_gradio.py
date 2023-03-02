import argparse
from datetime import datetime
import gradio as gr
import os
import torch
import json
from train_sample_utils import get_models, get_DDPM
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
from data.coco_w_stuff import get_coco_id_mapping
import numpy as np
from test_sample_utils import sample_one_image

coco_id_to_name = get_coco_id_mapping()
coco_name_to_id = {v: int(k) for k, v in coco_id_to_name.items()}

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
if os.path.exists(ckpt_path):
    print(f'INFO: Found checkpoint {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
    ddpm_model.load_state_dict(ckpt)
else:
    ckpt_path = None
    raise RuntimeError('Cannot do inference without pretrained checkpoint')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ddpm_model = ddpm_model.to(device)

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def obtain_bbox_from_yolo(image):
    H, W = image.shape[:2]
    results = yolo_model(image)
    # convert results to [x, y, w, h, object_name]
    xyxy_conf_cls = results.xyxy[0].detach().cpu().numpy()
    bboxes = []
    for x1, y1, x2, y2, conf, cls_idx in xyxy_conf_cls:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_name = yolo_model.names[int(cls_idx)]
        if conf >= 0.5:
            bboxes.append([x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H, cls_name])
    return bboxes

def save_bboxes(bboxes, save_dir):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = str(hash(str(current_time)))[1:10]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{file_name}.txt')
    with open(save_path, 'w') as OUT:
        for bbox in bboxes:
            OUT.write(','.join([str(x) for x in bbox]))
            OUT.write('\n')
    return save_path

def sample_images(ref_image):
    bboxes = obtain_bbox_from_yolo(ref_image)
    bbox_path = save_bboxes(bboxes, 'tmp')
    image, image_with_bbox, canvas_with_bbox = sample_one_image(
        bbox_path, 
        ddpm_model, 
        device, 
        coco_name_to_id, coco_id_to_name, 
        api_key=None, 
        image_size=ref_image.shape[:2]
    )
    # os.remove(bbox_path)
    if image is None:
        # Return a placeholder image and a message
        placeholder = np.zeros((ref_image.shape[0], ref_image.shape[1], 3), dtype=np.uint8)
        message = "No object found in the image"
        return message, placeholder, placeholder, placeholder
    else:
        return "", image, image_with_bbox, canvas_with_bbox

# Define the Gradio interface with a message component
input_image = gr.inputs.Image()
output_images = [gr.outputs.Image(type='numpy') for i in range(3)]
message = gr.outputs.Textbox(label="Information", type="text")
interface = gr.Interface(
    fn=sample_images,
    inputs=input_image,
    outputs=[message] + output_images,
    capture_session=True, 
    title="LayoutDiffuse", 
    description="Drop a reference image to generate a new image with the same layout",
    allow_flagging=False,
    live=False
)

interface.launch(share=True)
