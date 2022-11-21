import os
import torch
import torchvision
import cv2
import numpy as np
from pytorch_lightning.callbacks import Callback
from data.face_parsing import MaskMeshConverter, celebAMask_labels
from ..sampling_save_fig import format_dtype_and_shape, save_figure, save_sampling_history

def save_mask_index(image, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image = format_dtype_and_shape(image)
    image = image.astype(np.uint8)
    cv2.imwrite(save_path, image)

class CelebMaskImageSavingCallback(Callback):
    def __init__(self, expt_path, condition, start_idx=0):
        self.expt_path = expt_path
        assert condition in ['image', 'mask']
        self.condition = condition
        self.current_idx = start_idx
        self.mask_cvt = MaskMeshConverter(
            labels = list(celebAMask_labels.keys()),
            mesh_dim=3
        ) # convert colorized mask back to index

    def save_y_0_hat(self, y_0_hat, y_cond, prefix, rank, current_epoch, current_idx):
        for y_0, cond in zip(y_0_hat, y_cond):
            if self.condition == 'mask':
                mask = cond
                image = y_0
            else:
                mask = y_0
                image = cond
            save_figure(
                image,
                save_path=os.path.join(
                    self.expt_path, 
                    f'{prefix}_at_{current_epoch:05d}_image', 
                    f'{rank:02d}_{current_idx:05d}.png')
            )
            save_figure(
                mask,
                save_path=os.path.join(
                    self.expt_path, 
                    f'{prefix}_at_{current_epoch:05d}_mask', 
                    f'{rank:02d}_{current_idx:05d}.png')
            )
            save_mask_index(
                self.mask_cvt.nd_mesh_to_index_mask(
                    mask.permute(1, 2, 0)[None].detach().cpu()
                ).to(torch.uint8)[0],  # this function requires input to have (b, h, w, dim). out: (b, h, w)
                save_path=os.path.join(
                    self.expt_path, 
                    f'{prefix}_at_{current_epoch:05d}_mask_index', 
                    f'{rank:02d}_{current_idx:05d}.png')
            )
            current_idx += 1

    def save_y_t_hist(self, y_t_hist, prefix, rank, current_epoch, current_idx):
        for hist_image_or_mask in y_t_hist:
            save_sampling_history(
                hist_image_or_mask,
                save_path=os.path.join(
                    self.expt_path, 
                    f'{prefix}_at_{current_epoch:05d}_image', 
                    f'{rank:02d}_{current_idx:05d}.png')
            )
            current_idx += 1

class CelebMaskPaletteImageSavingCallback(CelebMaskImageSavingCallback):

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        rank = pl_module.global_rank
        current_epoch = pl_module.current_epoch
        y_0_hat = outputs['sampling']['model_output']
        y_cond = outputs['condition']
        y_t_hist = outputs['sampling']['model_history_output']
        self.save_y_0_hat(
            y_0_hat, y_cond,
            prefix=f'cond_{self.condition}_sampling',
            rank=rank, current_epoch=current_epoch,
            current_idx = self.current_idx
        )
        self.save_y_t_hist(
            y_t_hist,
            prefix=f'cond_{self.condition}_sampling_hist',
            rank=rank, current_epoch=current_epoch,
            current_idx = self.current_idx
        )

        self.current_idx += len(y_0_hat)
