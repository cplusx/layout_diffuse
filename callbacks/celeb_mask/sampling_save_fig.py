import os
import torch
import torchvision
import cv2
import numpy as np
from pytorch_lightning.callbacks import Callback
from data.face_parsing import MaskMeshConverter, celebAMask_labels
from ..sampling_save_fig import format_dtype_and_shape, save_figure, save_sampling_history

def format_image(x):
    x = x.cpu()
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    x = x.permute(1,2,0).detach().numpy()
    return x

def save_mask_index(image, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image = format_dtype_and_shape(image)
    image = image.astype(np.uint8)
    cv2.imwrite(save_path, image)

def save_raw_image_tensor(x, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, image=x)

# TODO, change colorizer to the gt color of celeb mask
class MaskColorizer():
    def __init__(self):
        self.mask_cvt = MaskMeshConverter(
            labels = list(celebAMask_labels.keys()),
            mesh_dim=3
        )
    def __call__(self, x):
        assert len(x.shape) == 3 or len(x.shape) == 4, f'mask should have shape 3 or 4, got {x.shape}'
        # input: 1, h, w or time, 1, h, w
        x = x.squeeze(-3) # (h, w) or (time, h, w)
        x = self.mask_cvt(x) # h, w, 3 or time, h, w, 3
        if len(x.shape) == 3:
            return x.permute(2, 0, 1) # 3, h, w
        elif len(x.shape) == 4:
            return x.permute(0, 3, 1, 2) # time, 3, h, w
        else:
            raise RuntimeError(f'Unknown dim, mask shape is {x.shape}')

class CelebMaskImageSavingCallback(Callback):
    def __init__(self, expt_path, start_idx=0):
        self.expt_path = expt_path
        self.current_idx = start_idx
        self.mask_colorizer = MaskColorizer()
        self.repeat_idx = -1

    def save_y_0_hat(self, image, mask, prefix, rank, current_epoch, current_idx, num_gpus=1):
            save_figure(
                image,
                save_path=os.path.join(
                    self.expt_path, 
                    f'epoch_{current_epoch:05d}',
                    f'image', 
                    f'{rank+num_gpus*current_idx:04d}_{self.repeat_idx:02d}.png')
            )
            if mask is not None:
                save_figure(
                    self.mask_colorizer(mask[None]),
                    save_path=os.path.join(
                        self.expt_path, 
                        f'epoch_{current_epoch:05d}',
                        f'mask', 
                        f'{rank+num_gpus*current_idx:04d}.png')
                )
                save_mask_index(
                    mask,
                    save_path=os.path.join(
                        self.expt_path, 
                        f'epoch_{current_epoch:05d}',
                        f'mask_index', 
                        f'{rank+num_gpus*current_idx:04d}.png')
                )

            save_raw_image_tensor(
                format_image(image),
                save_path=os.path.join(
                    self.expt_path, 
                    f'epoch_{current_epoch:05d}',
                    'raw_tensor', 
                    f'{rank+num_gpus*current_idx:04d}_{self.repeat_idx:02d}') # will add .npz automatically
            )

    def save_y_t_hist(self, y_t_hist, prefix, rank, current_epoch, current_idx):
        for hist_image in y_t_hist:
            save_sampling_history(
                hist_image,
                save_path=os.path.join(
                    self.expt_path, 
                    f'{prefix}_at_{current_epoch:05d}_image', 
                    f'{rank:02d}_{current_idx:05d}.png')
            )
            current_idx += 1

class CelebMaskPartialAttnImageSavingCallback(CelebMaskImageSavingCallback):

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            self.repeat_idx += 1
            self.current_idx = 0
        rank = pl_module.global_rank
        current_epoch = pl_module.current_epoch

        y_0_hat = outputs['sampling']['model_output']
        y_t_hist = outputs['sampling']['model_history_output']
        masks = batch['seg_mask']
        for image, mask in zip(y_0_hat, masks):
            self.save_y_0_hat(
                image, mask,
                prefix='sampling',
                rank=rank, current_epoch=current_epoch,
                current_idx = self.current_idx,
                num_gpus=trainer.num_devices
            )
        # self.save_y_t_hist(
        #     y_t_hist,
        #     prefix='sampling_hist',
        #     rank=rank, current_epoch=current_epoch,
        #     current_idx = self.current_idx
        # )

            self.current_idx += 1

class CelebMaskBaselineImageSavingCallback(CelebMaskImageSavingCallback):

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            self.repeat_idx += 1
            self.current_idx = 0
        rank = pl_module.global_rank
        current_epoch = pl_module.current_epoch

        y_0_hat = outputs['sampling']['model_output']
        y_t_hist = outputs['sampling']['model_history_output']
        masks = batch['seg_mask']
        for image, mask in zip(y_0_hat, masks):
            self.save_y_0_hat(
                image, None,
                prefix='sampling',
                rank=rank, current_epoch=current_epoch,
                current_idx = self.current_idx,
                num_gpus=trainer.num_devices
            )
        # self.save_y_t_hist(
        #     y_t_hist,
        #     prefix='sampling_hist',
        #     rank=rank, current_epoch=current_epoch,
        #     current_idx = self.current_idx
        # )

            self.current_idx += 1