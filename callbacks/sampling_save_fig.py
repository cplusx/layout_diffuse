import os
import torch
import torchvision
import cv2
import numpy as np
from pytorch_lightning.callbacks import Callback

from .utils import unnorm, clip_image

def format_dtype_and_shape(x):
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3 and x.shape[0] == 3:
            x = x.permute(1, 2, 0)
        if len(x.shape) == 4 and x.shape[1] == 3:
            x = x.permute(0, 2, 3, 1)
        x = x.detach().cpu().numpy()
    return x

def save_figure(image, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if image.min() < 0:
        image = clip_image(unnorm(image))
    image = format_dtype_and_shape(image)
    image = (image * 255).astype(np.uint8)
    cv2.imwrite(save_path, image[..., ::-1])

def save_sampling_history(image, save_path):
    if image.min() < 0:
        image = clip_image(unnorm(image))
    grid_img = torchvision.utils.make_grid(image, nrow=4)
    save_figure(grid_img, save_path)

class BasicImageSavingCallback(Callback):
    def __init__(self, expt_path, start_idx=0):
        self.expt_path = expt_path
        self.current_idx = start_idx
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        rank = pl_module.global_rank
        current_epoch = pl_module.current_epoch
        y_0_hat = outputs['sampling']['model_output']
        y_t_hist = outputs['sampling']['model_history_output']
        for image, hist_image in zip(y_0_hat, y_t_hist):
            save_figure(
                image,
                save_path=os.path.join(
                    self.expt_path, 
                    f'sampling_at_{current_epoch:05d}', 
                    f'{rank:02d}_{self.current_idx:05d}.png')
            )
            save_sampling_history(
                hist_image,
                save_path=os.path.join(
                    self.expt_path, 
                    f'sampling_hist_at_{current_epoch:05d}', 
                    f'{rank:02d}_{self.current_idx:05d}.png')
            )
            self.current_idx += 1
