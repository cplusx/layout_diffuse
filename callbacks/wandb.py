import wandb
import torch
import torchvision
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from .utils import unnorm, clip_image

class WandBImageLogger(Callback):
    def __init__(
        self, 
        wandb_logger: WandbLogger=None,
        max_num_images: int=16,
    ) -> None:
        super().__init__()
        self.wandb_logger = wandb_logger
        self.max_num_images = max_num_images

    # TODO move following two functions to utils.py
    def tensor2numpy(self, x):
        x = x.float() # handle bf16
        '''convert 4D (b, dim, h, w) pytorch tensor to numpy (b, h, w, dim)
        or convert 3D (dim, h, w) pytorch tensor to numpy (h, w, dim)'''
        if len(x.shape) == 4:
            return x.permute(0, 2, 3, 1).detach().cpu().numpy()
        else:
            return x.permute(1, 2, 0).detach().cpu().numpy()

    def tensor2image(self, x):
        x = x.float() # handle bf16
        '''convert 4D (b, dim, h, w) pytorch tensor to wandb Image class'''
        grid_img = torchvision.utils.make_grid(
            x, nrow=4
        ).permute(1, 2, 0).detach().cpu().numpy()
        img = wandb.Image(
            grid_img
        )
        return img

    def on_train_batch_end(
        self, trainer, pl_module, 
        outputs, batch, batch_idx
    ):
        # record images in first batch
        if isinstance(outputs, list):
            print(outputs)
            raise
        if batch_idx == 0:
            raw_image = self.tensor2image(clip_image(unnorm(
                outputs['raw_image'][:self.max_num_images]
            )))
            model_input = self.tensor2image(clip_image(unnorm(
                outputs['model_input'][:self.max_num_images]
            )))
            model_output = self.tensor2image(clip_image(unnorm(
                outputs['model_output'][:self.max_num_images]
            )))
            y_0_hat = self.tensor2image(clip_image(unnorm(
                outputs['y_0_hat'][:self.max_num_images]
            )))
            self.wandb_logger.experiment.log({
                'train/raw_image': raw_image,
                'train/model_input': model_input,
                'train/model_output': model_output,
                'train/y_0_hat': y_0_hat
            })

    def on_validation_batch_end(
        self, trainer, pl_module, 
        outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""
        if batch_idx == 0:
            y_0 = self.tensor2image(clip_image(unnorm(
                outputs['y_0_image'][:self.max_num_images]
            )))
            self.wandb_logger.experiment.log({
                f'validation/raw_image': y_0,
            })
            outputs.pop('y_0_image')
            '''outputs has result of restoration and sampling'''
            for output_type, this_outputs in outputs.items():
                y_0_hat = self.tensor2image(clip_image(unnorm(
                    this_outputs['model_output'][:self.max_num_images]
                )))
                y_t_hist = unnorm(
                    this_outputs['model_history_output'][:self.max_num_images]
                ) # bs, time step, im_dim, im_h, im_w

                self.wandb_logger.experiment.log({
                    f'validation/{output_type}': y_0_hat,
                })

                '''log predictions as a Table'''
                columns = [f't={t}' for t in reversed(range(y_t_hist.shape[1]))]
                data = []
                for this_y_t_hist in y_t_hist:
                    this_Images = [
                        wandb.Image(self.tensor2numpy(i)) for i in this_y_t_hist
                    ]
                    data.append(this_Images)
                self.wandb_logger.log_table(
                    key=f'validation_table/{output_type}', 
                    columns=columns, data=data
                )

class WandBVAEImageLogger(Callback):
    def __init__(
        self, 
        wandb_logger: WandbLogger=None,
        max_num_images: int=16,
    ) -> None:
        super().__init__()
        self.wandb_logger = wandb_logger
        self.max_num_images = max_num_images

    def tensor2image(self, x):
        '''convert 4D (b, dim, h, w) pytorch tensor to wandb Image class'''
        grid_img = torchvision.utils.make_grid(
            x, nrow=4
        ).permute(1, 2, 0).detach().cpu().numpy()
        img = wandb.Image(
            grid_img
        )
        return img

    def on_train_batch_end(
        self, trainer, pl_module, 
        outputs, batch, batch_idx
    ):
        if batch_idx == 0:
            raw_image = self.tensor2image(clip_image(unnorm(
                outputs['raw_image'][:self.max_num_images]
            )))
            model_output = self.tensor2image(clip_image(unnorm(
                outputs['model_output'][:self.max_num_images]
            )))
            self.wandb_logger.experiment.log({
                'train/raw_image': raw_image,
                'train/model_output': model_output,
            })

    def on_validation_batch_end(
        self, trainer, pl_module, 
        outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""
        if batch_idx == 0:
            x = self.tensor2image(clip_image(unnorm(
                outputs['raw_image'][:self.max_num_images]
            )))
            self.wandb_logger.experiment.log({
                f'validation/raw_image': x,
            })
            o = self.tensor2image(clip_image(unnorm(
                outputs['model_output'][:self.max_num_images]
            )))
            self.wandb_logger.experiment.log({
                f'validation/model_output': o,
            })