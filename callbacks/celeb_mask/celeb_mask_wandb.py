import wandb
from ..wandb import WandBImageLogger, clip_image, unnorm
from ..utils import to_mask_if_dim_gt_3
# colorize mask is included in the to_mask... func

class CelebMaskWandBImageLogger(WandBImageLogger):
    def on_train_batch_end(
        self, trainer, pl_module, 
        outputs, batch, batch_idx
    ):
        if batch_idx == 0:
            # if raw_image is one-hot mask, it needs to be map to 1 dim for visualization
            raw_image = self.tensor2image(clip_image(unnorm(
                outputs['raw_image'][:self.max_num_images]
            )))
            raw_mask = self.tensor2image(to_mask_if_dim_gt_3(clip_image(unnorm(
                outputs['raw_mask'][:self.max_num_images]
            ))))

            model_input = clip_image(unnorm(
                outputs['model_input'][:self.max_num_images]
            ))
            model_input_image = self.tensor2image(model_input[:, :3])
            model_input_mask = self.tensor2image(to_mask_if_dim_gt_3(
                model_input[:, 3:]
            ))

            model_output = clip_image(unnorm(
                outputs['model_output'][:self.max_num_images]
            ))
            model_output_image = self.tensor2image(model_output[:, :3])
            model_output_mask = self.tensor2image(to_mask_if_dim_gt_3(
                model_output[:, 3:]
            ))

            y_0_hat = clip_image(unnorm(
                outputs['y_0_hat'][:self.max_num_images]
            ))
            y_0_hat_image = self.tensor2image(y_0_hat[:, :3])
            y_0_hat_mask = self.tensor2image(to_mask_if_dim_gt_3(
                y_0_hat[:, 3:]
            ))

            self.wandb_logger.experiment.log({
                'train/raw_image': raw_image,
                'train/raw_mask': raw_mask,
                'train/model_input_image': model_input_image,
                'train/model_input_mask': model_input_mask,
                'train/model_output_image': model_output_image,
                'train/model_output_mask': model_output_mask,
                'train/y_0_hat_image': y_0_hat_image,
                'train/y_0_hat_mask': y_0_hat_mask
            })


    def on_validation_batch_end(
        self, trainer, pl_module, 
        outputs, batch, batch_idx, 
        dataloader_idx
    ):
        if batch_idx == 0:
            y_0_image = outputs.pop('y_0_image')
            y_0_mask = outputs.pop('y_0_mask')

            y_0_image = self.tensor2image(clip_image(unnorm(
                y_0_image[:self.max_num_images]
            )))
            y_0_mask = self.tensor2image(to_mask_if_dim_gt_3(clip_image(unnorm(
                y_0_mask[:self.max_num_images]
            ))))
            self.wandb_logger.experiment.log({
                'validation/y_0_image': y_0_image,
                'validation/y_0_mask': y_0_mask
            })

            for output_type, this_outputs in outputs.items():
                y_0_hat = clip_image(unnorm(
                    this_outputs['model_output'][:self.max_num_images]
                ))
                y_0_hat_image = self.tensor2image(y_0_hat[:, :3])
                y_0_hat_mask = self.tensor2image(to_mask_if_dim_gt_3(y_0_hat[:, 3:]))
                self.wandb_logger.experiment.log({
                    f'validation/{output_type}_image': y_0_hat_image,
                    f'validation/{output_type}_mask': y_0_hat_mask
                })


                y_t_hist = clip_image(unnorm(
                    this_outputs['model_history_output'][:self.max_num_images]
                )) # bs, time step, im_dim, im_h, im_w
                y_t_hist_image = y_t_hist[:, :, :3]
                y_t_hist_mask = to_mask_if_dim_gt_3(
                    y_t_hist[:, :, 3:], dim=2
                )

                # '''log images with `WandbLogger.log_image`
                # must convert to a list'''
                # self.wandb_logger.log_image(
                #     key=f'validation/{output_type}_image', 
                #     images=[i for i in self.tensor2numpy(y_0_hat_image)],
                #     caption=[f'im_{i}' for i in range(len(y_0_hat_image))]
                # )
                # self.wandb_logger.log_image(
                #     key=f'validation/{output_type}_mask', 
                #     images=[i for i in self.tensor2numpy(y_0_hat_mask)],
                #     caption=[f'im_{i}' for i in range(len(y_0_hat_mask))]
                # )

                '''log predictions as a Table'''
                columns = [f't={t}' for t in reversed(range(y_t_hist_image.shape[1]))]
                data = []
                for this_y_t_hist in y_t_hist_image:
                    this_Images = [
                        wandb.Image(self.tensor2numpy(i)) for i in this_y_t_hist
                    ]
                    data.append(this_Images)
                self.wandb_logger.log_table(
                    key=f'validation_table/{output_type}_image', 
                    columns=columns, data=data
                )

                columns = [f't={t}' for t in reversed(range(y_t_hist_mask.shape[1]))]
                data = []
                for this_y_t_hist in y_t_hist_mask:
                    this_Images = [
                        wandb.Image(self.tensor2numpy(i)) for i in this_y_t_hist
                    ]
                    data.append(this_Images)
                self.wandb_logger.log_table(
                    key=f'validation_table/{output_type}_mask', 
                    columns=columns, data=data
                )

class CelebMaskEmbeddingWandBImageLogger(WandBImageLogger):
    def on_train_batch_end(
        self, trainer, pl_module, 
        outputs, batch, batch_idx
    ):
        if batch_idx == 0:
            # if raw_image is one-hot mask, it needs to be map to 1 dim for visualization
            raw_image = self.tensor2image(clip_image(unnorm(
                outputs['raw_image'][:self.max_num_images]
            )))
            raw_mask = self.tensor2image(to_mask_if_dim_gt_3(
                outputs['raw_mask'][:self.max_num_images]
            ))

            model_input = outputs['model_input'][:self.max_num_images]
            model_input_image = self.tensor2image(clip_image(unnorm(model_input[:, :3])))
            model_input_mask = self.tensor2image(to_mask_if_dim_gt_3(
                model_input[:, 3:]
            )) # b, 3, h, w

            model_output = outputs['model_output'][:self.max_num_images]
            model_output_image = self.tensor2image(clip_image(unnorm(model_output[:, :3])))
            model_output_mask = self.tensor2image(to_mask_if_dim_gt_3(
                model_output[:, 3:]
            )) # b, 3, h, w

            y_0_hat = outputs['y_0_hat'][:self.max_num_images]
            y_0_hat_image = self.tensor2image(clip_image(unnorm(y_0_hat[:, :3])))
            y_0_hat_mask = self.tensor2image(to_mask_if_dim_gt_3(
                y_0_hat[:, 3:]
            )) # b, num classes, h, w

            self.wandb_logger.experiment.log({
                'train/raw_image': raw_image,
                'train/raw_mask': raw_mask,
                'train/model_input_image': model_input_image,
                'train/model_input_mask': model_input_mask,
                'train/model_output_image': model_output_image,
                'train/model_output_mask': model_output_mask,
                'train/y_0_hat_image': y_0_hat_image,
                'train/y_0_hat_mask': y_0_hat_mask
            })


    def on_validation_batch_end(
        self, trainer, pl_module, 
        outputs, batch, batch_idx, 
        dataloader_idx
    ):
        if batch_idx == 0:
            y_0_image = outputs.pop('y_0_image')
            y_0_mask = outputs.pop('y_0_mask')

            y_0_image = self.tensor2image(clip_image(unnorm(
                y_0_image[:self.max_num_images]
            )))
            y_0_mask = self.tensor2image(to_mask_if_dim_gt_3(
                y_0_mask[:self.max_num_images]
            ))
            self.wandb_logger.experiment.log({
                'validation/y_0_image': y_0_image,
                'validation/y_0_mask': y_0_mask
            })

            for output_type, this_outputs in outputs.items():
                y_0_hat = this_outputs['model_output'][:self.max_num_images]
                y_0_hat_image = clip_image(unnorm(y_0_hat[:, :3]))
                y_0_hat_mask = to_mask_if_dim_gt_3(y_0_hat[:, 3:]) # b, num classes, h, w

                y_t_hist = this_outputs['model_history_output'][:self.max_num_images]
                y_t_hist_image = clip_image(unnorm(y_t_hist[:, :, :3]))
                y_t_hist_mask = to_mask_if_dim_gt_3(
                    y_t_hist[:, :, 3:], dim=2
                ) # b, t, num classes, h, w

                '''log images with `WandbLogger.log_image`
                must convert to a list'''
                self.wandb_logger.log_image(
                    key=f'validation/{output_type}_image', 
                    images=[i for i in self.tensor2numpy(y_0_hat_image)],
                    caption=[f'im_{i}' for i in range(len(y_0_hat_image))]
                )
                self.wandb_logger.log_image(
                    key=f'validation/{output_type}_mask', 
                    images=[i for i in self.tensor2numpy(y_0_hat_mask)],
                    caption=[f'im_{i}' for i in range(len(y_0_hat_mask))]
                )

                '''log predictions as a Table'''
                columns = [f't={t}' for t in reversed(range(y_t_hist_image.shape[1]))]
                data = []
                for this_y_t_hist in y_t_hist_image:
                    this_Images = [
                        wandb.Image(self.tensor2numpy(i)) for i in this_y_t_hist
                    ]
                    data.append(this_Images)
                self.wandb_logger.log_table(
                    key=f'validation_table/{output_type}_image', 
                    columns=columns, data=data
                )

                columns = [f't={t}' for t in reversed(range(y_t_hist_mask.shape[1]))]
                data = []
                for this_y_t_hist in y_t_hist_mask:
                    this_Images = [
                        wandb.Image(self.tensor2numpy(i)) for i in this_y_t_hist
                    ]
                    data.append(this_Images)
                self.wandb_logger.log_table(
                    key=f'validation_table/{output_type}_mask', 
                    columns=columns, data=data
                )