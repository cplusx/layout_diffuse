import os
import torch
from data.coco_w_stuff import get_coco_id_mapping
import numpy as np
import cv2
import time
from test_sample_utils import sample_one_image, parse_test_args, load_test_models, load_model_weights
coco_id_to_name = get_coco_id_mapping()
coco_name_to_id = {v: int(k) for k, v in coco_id_to_name.items()}

if __name__ == '__main__':
    args = parse_test_args()
    ddpm_model = load_test_models(args)
    load_model_weights(ddpm_model=ddpm_model, args=args)

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