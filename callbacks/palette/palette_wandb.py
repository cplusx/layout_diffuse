import wandb
from ..wandb import WandBImageLogger, clip_image, unnorm
from ..utils import to_mask_if_dim_gt_3, colorize_mask

class PaletteWandBImageLogger(WandBImageLogger):
    def on_train_batch_end(
        self, trainer, pl_module, 
        outputs, batch, batch_idx
    ):
        if batch_idx == 0:
            # if raw_image is one-hot mask, it needs to be map to 1 dim for visualization
            raw_image = self.tensor2image(to_mask_if_dim_gt_3(clip_image(unnorm(
                outputs['raw_image'][:self.max_num_images]
            ))))
            model_input = self.tensor2image(to_mask_if_dim_gt_3(clip_image(unnorm(
                outputs['model_input'][:self.max_num_images]
            ))))
            model_output = self.tensor2image(to_mask_if_dim_gt_3(clip_image(unnorm(
                outputs['model_output'][:self.max_num_images]
            ))))
            y_0_hat = self.tensor2image(to_mask_if_dim_gt_3(clip_image(unnorm(
                outputs['y_0_hat'][:self.max_num_images]
            ))))
            self.wandb_logger.experiment.log({
                'train/raw_image': raw_image,
                'train/model_input': model_input,
                'train/model_output': model_output,
                'train/y_0_hat': y_0_hat
            })

            y_cond = to_mask_if_dim_gt_3(unnorm(outputs['condition']))
            if y_cond.shape[1] == 1:
                '''if condition is mask, convert it to colored mask'''
                y_cond = colorize_mask(y_cond)
            y_cond = self.tensor2image(
                y_cond[:self.max_num_images]
            )
            self.wandb_logger.experiment.log({
                'train/y_cond': y_cond
            })

    def on_validation_batch_end(
        self, trainer, pl_module, 
        outputs, batch, batch_idx, 
        dataloader_idx
    ):
        if batch_idx == 0:
            y_cond = outputs.pop('condition')
            y_cond = to_mask_if_dim_gt_3(unnorm(y_cond[:self.max_num_images]))
            if y_cond.shape[1] == 1:
                '''if condition is mask, convert it to colored mask'''
                y_cond = colorize_mask(y_cond)
            y_cond = self.tensor2image(y_cond)
            self.wandb_logger.experiment.log({
                'validation/y_cond': y_cond
            })

            for output_type, this_outputs in outputs.items():
                y_0_hat = to_mask_if_dim_gt_3(unnorm(
                    this_outputs['model_output'][:self.max_num_images]
                ))
                y_t_hist = to_mask_if_dim_gt_3(unnorm(
                    this_outputs['model_history_output'][:self.max_num_images]
                ), dim=2) # bs, time step, im_dim, im_h, im_w

                '''log images with `WandbLogger.log_image`
                must convert to a list'''
                self.wandb_logger.log_image(
                    key=f'validation/{output_type}', 
                    images=[i for i in self.tensor2numpy(y_0_hat)],
                    caption=[f'im_{i}' for i in range(len(y_0_hat))]
                )

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
