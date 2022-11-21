from tkinter import Y
import torch
from .DDIM_ldm import DDIM_LDM_VQVAETraining
from model_utils import default

class DDIM_LDM_pretrained_celeb(DDIM_LDM_VQVAETraining):
    def process_batch(self, batch, mode='train'):
        return super().process_batch(batch['image'], mode)

class DDIM_LDM_celeb_baseline(DDIM_LDM_VQVAETraining):
    def __init__(self, *args, freeze_pretrained_weights=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_pretrained_weights = freeze_pretrained_weights

    def initialize_unet(self, unet_init_weights):
        if unet_init_weights is not None:
            model_sd = torch.load(unet_init_weights)
            self_model_sd = self.denoise_fn.state_dict()
            self_model_params = dict(self.denoise_fn.named_parameters())
            self.params_not_pretrained = []
            for self_k, model_k in zip(self_model_sd, model_sd):
                assert self_k == model_k, f'get {self_k} and {model_k}'
                if self_model_sd[self_k].shape == model_sd[model_k].shape:
                    self_model_sd[self_k] = model_sd[model_k]
                else:
                    self.params_not_pretrained.append(self_model_params[self_k])
            self.denoise_fn.load_state_dict(self_model_sd)

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

    def process_batch(self, batch, mode='train'):
        y_t, target, t, x_0, model_kwargs = super().process_batch(batch['image'], mode)
        seg_mask = batch['seg_mask'][:, :, 2::4, 2::4]
        y_t = torch.cat([y_t, seg_mask], dim=1)
        return y_t, target, t, x_0, model_kwargs

    def training_step(self, batch, batch_idx):
        x, y, t, raw_image, model_kwargs = self.process_batch(batch, mode='train')
        pred = self.denoise_fn(x, t, **model_kwargs)
        loss, loss_simple, loss_vlb = self.get_loss(pred, y, t)
        with torch.no_grad():
            if self.training_target == 'noise':
                y_0_hat = self.predict_start_from_noise(
                    x[:, :3], t=t, 
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
            'model_input': x[:, :3],
            'model_output': pred,
            'y_0_hat': self.decode_latent_to_image(y_0_hat)
        }

    def validation_step(self, batch, batch_idx):
        y_t, _, _, y_0_image, model_kwargs = self.process_batch(batch, mode='val')
        dim = y_t.shape[1]
        assert dim == 6, RuntimeError(f'dim != 6, get dim = {dim}')
        y_t_img = y_t[:, :3]
        y_cond = y_t[:, 3:]
        restored = self.sampling(noise=y_t_img, y_cond=y_cond)
        sampled = self.sampling(noise=torch.randn_like(y_t_img), y_cond=y_cond)
        return {
            'y_0_image': y_0_image,
            'restore': restored,
            'sampling': sampled
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        y_t, _, _, y_0_image, model_kwargs = self.process_batch(batch, mode='val')
        dim = y_t.shape[1]
        assert dim == 6, RuntimeError(f'dim != 6, get dim = {dim}')
        y_t_img = y_t[:, :3]
        y_cond = y_t[:, 3:]
        sampled = self.sampling(noise=torch.randn_like(y_t_img), y_cond=y_cond)
        return {
            'sampling': sampled
        }

    @torch.no_grad()
    def sampling(self, noise=None, y_cond=None, image_size=None, sample_num=10, model_kwargs={}):
        if noise is None and image_size is None:
            raise RuntimeError('Either noise or image size should be given')
        noise = default(noise, lambda: torch.randn(*image_size))
        if self.use_fast_sampling:
            y_0, y_t_hist = self.fast_sampling(noise, y_cond, model_kwargs)
        else:
            sample_num = min(self.num_timesteps, sample_num)
            sample_inter = (self.num_timesteps//sample_num)
            y_0, y_t_hist = self.restore_from_y_T(noise, y_cond, sample_inter, model_kwargs=model_kwargs)
        return {
            'model_output': y_0,
            'model_history_output': y_t_hist
        }

    def fast_sampling(self, noise, y_cond, model_kwargs=None):
        if self.fast_sampler == 'plms':
            from .PLMSSampler_for_baseline_ldm import PLMSSampler as FastSampler
        else:
            raise NotImplementedError

        model_kwargs = default(model_kwargs, {})
        sampler = FastSampler(self.denoise_fn, self.beta_schedule_args)
        y_0, y_t_hist = sampler.sample(
            S=self.fast_sampling_steps,
            batch_size=noise.shape[0],
            x_T=noise, # when x_T is given, shape is not used
            y_cond=y_cond,
            shape=[3, 64, 64],
            verbose=False,
            eta=0,
            model_kwargs=model_kwargs,
            uncondition_model_kwargs=None,
            guidance_scale=self.guidance_scale
        )
        y_t_hist = torch.stack(y_t_hist['x_inter'], dim=1) # bs, n_timestep, dim, h, w
        y_0 = self.decode_latent_to_image(y_0)
        y_t_hist = [self.decode_latent_to_image(y_t_hist[:, i]) for i in range(y_t_hist.shape[1])]
        y_t_hist = torch.stack(y_t_hist, dim=1) # bs, n_timestep, dim, h, w
        return y_0, y_t_hist

    def restore_from_y_T(self, y_t, y_cond, sample_inter=10000, model_kwargs=None):
        b, *_ = y_t.shape
        y_t = default(y_t, lambda: torch.randn_like(y_t))
        x_t_hist = self.decode_latent_to_image(y_t).unsqueeze(1) # bs, time step, im_dim, im_h, im_w
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=y_t.device, dtype=torch.long)
            y_t = self.p_sample(y_t, y_cond, t, model_kwargs)
            if i % sample_inter == 0:
                x_t = self.decode_latent_to_image(y_t)
                x_t_hist = torch.cat([x_t_hist, x_t.unsqueeze(1)], dim=1)
        x_0 = self.decode_latent_to_image(y_t)
        return x_0, x_t_hist

    @torch.no_grad()
    def p_sample(self, y_t, y_cond, t, model_kwargs=None):
        '''From y_t to the y_{t-1}'''
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, y_cond=y_cond, t=t, model_kwargs=model_kwargs)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    def p_mean_variance(self, y_t, y_cond, t, model_kwargs=None):
        '''From y_t to the mean / std  of y_{t-1} (before adding noise)'''
        model_kwargs = default(model_kwargs, {})
        if self.training_target == 'noise':
            y_0_hat = self.predict_start_from_noise(
                    y_t, t=t, noise=self.denoise_fn(
                        torch.cat([y_t, y_cond], dim=1), 
                        t, **model_kwargs)
                    )
        else:
            y_0_hat = self.denoise_fn(
                torch.cat([y_t, y_cond], dim=1), 
                t, **model_kwargs
            )

        if self.clip_denoised:
            y_0_hat = self.thresholding_y_0(y_0_hat)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

class DDIM_LDM_partial_attn_celeb_mask(DDIM_LDM_pretrained_celeb):
    def __init__(self, *args, freeze_pretrained_weights=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_pretrained_weights = freeze_pretrained_weights

    def process_batch(self, batch, mode='train'):
        y_t, target, t, x_0, model_kwargs = super().process_batch(batch, mode)
        model_kwargs.update({'context': batch['seg_mask']})
        return y_t, target, t, x_0, model_kwargs

    def initialize_unet(self, unet_init_weights):
        if unet_init_weights is not None:
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
        # restored['model_output'] = self.decode_latent_to_image(
        #     restored['model_output']
        # )
        # sampled['model_output'] = self.decode_latent_to_image(
        #     sampled['model_output']
        # )
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