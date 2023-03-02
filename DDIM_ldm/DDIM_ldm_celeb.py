import os
import torch
from .DDIM_ldm import DDIM_LDM_VQVAETraining

class DDIM_LDM_pretrained_celeb(DDIM_LDM_VQVAETraining):
    def process_batch(self, batch, mode='train'):
        return super().process_batch(batch['image'], mode)

class DDIM_LDM_LayoutDiffuse_celeb_mask(DDIM_LDM_pretrained_celeb):
    def __init__(self, *args, freeze_pretrained_weights=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_pretrained_weights = freeze_pretrained_weights

    def process_batch(self, batch, mode='train'):
        y_t, target, t, x_0, model_kwargs = super().process_batch(batch, mode)
        model_kwargs.update({'context': {
            'layout': batch['seg_mask']
        }})
        return y_t, target, t, x_0, model_kwargs

    def initialize_unet(self, unet_init_weights):
        if unet_init_weights is not None:
            if os.path.exists(unet_init_weights):
                print(f'INFO: initialize denoising UNet from {unet_init_weights}, NOTE: without partial attention layers')
                model_sd = torch.load(unet_init_weights)
                self_model_sd = self.denoise_fn.state_dict()
                self_model_params = list(self.denoise_fn.named_parameters())
                self_model_k = list(map(lambda x: x[0], self_model_params))
                self.params_not_pretrained = []
                k_idx = 0
                for model_layer_idx, (model_k, model_v) in enumerate(model_sd.items()):
                    while (self_model_params[k_idx][1].shape != model_v.shape) or (model_k.split('.')[0:2] != self_model_k[k_idx].split('.')[0:2]):
                        self.params_not_pretrained.append(self_model_params[k_idx][1])
                        k_idx += 1
                    self_model_sd[self_model_k[k_idx]] = model_v
                    k_idx += 1
                self.denoise_fn.load_state_dict(self_model_sd)
            else:
                print(f'WARNING: cannot find pretrained weights {unet_init_weights}, initialize from scratch')

    def training_step(self, batch, batch_idx):
        self.clip_denoised = False # during training do not clip to -1 to 1 to prevent grad detached
        y_t, y_target, t, raw_image, model_kwargs = self.process_batch(batch, mode='train')
        pred = self.denoise_fn(y_t, t, **model_kwargs)
        loss, loss_simple, loss_vlb = self.get_loss(pred, y_target, t)
        with torch.no_grad():
            if self.training_target == 'noise':
                y_0_hat = self.predict_start_from_noise(
                    y_t, t=t, 
                    noise=pred.detach()
                )
            else:
                y_0_hat = pred.detach()

        self.log(f'train_loss', loss)
        self.log(f'train_loss_simple', loss_simple)
        self.log(f'train_loss_vlb', loss_vlb)
        if self.learn_logvar:
            self.log(f'logvar', self.logvar.data.mean())
        return {
            'loss': loss,
            'raw_image': raw_image,
            'model_input': y_t,
            'model_output': pred,
            'y_0_hat': self.decode_latent_to_image(y_0_hat)
        }

    def validation_step(self, batch, batch_idx):
        y_t, _, _, y_0_image, model_kwargs = self.process_batch(batch, mode='val')
        restored = self.sampling(noise=y_t, model_kwargs=model_kwargs)
        sampled = self.sampling(noise=torch.randn_like(y_t), model_kwargs=model_kwargs)
        return {
            'y_0_image': y_0_image,
            'restore': restored,
            'sampling': sampled
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        y_t, _, _, y_0_image, model_kwargs = self.process_batch(batch, mode='val')
        sampled = self.sampling(noise=torch.randn_like(y_t), model_kwargs=model_kwargs)
        return {
            'sampling': sampled
        }

    def configure_optimizers(self):
        if self.freeze_pretrained_weights:
            assert hasattr(self, 'params_not_pretrained')
            print('INFO: pretrained weights are not trainable')
            params = self.params_not_pretrained
        else:
            params = list(self.denoise_fn.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        optimizer = torch.optim.Adam(params, **self.optim_args)
        return optimizer