import torch
from torch import nn
import numpy as np
import math
from functools import partial
from inspect import isfunction
from typing import Callable
import pytorch_lightning as pl
from model_utils import exists, default, extract_into_tensor as extract, right_pad_dims_to, make_beta_schedule
# TODO, the full timestep sampling have not added guidance scale and negative prompt, do it later

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class DDIM_LDM(pl.LightningModule):
    def __init__(
        self, 
        denoise_fn,
        beta_schedule_args,
        training_target, 
        loss_fn,
        optim_args,
        clip_denoised=False, # by default, ldm should not clip denoised since the computing is done in the latent space, and the latent space is not between -1 and 1
        learn_logvar=False,
        logvar_init=0.,
        dynamic_thresholding=False,
        dynamic_thresholding_percentile=0.9,
        **kwargs
    ):
        '''
        denoising_fn: a denoising model such as UNet
        beta_schedule_args: a dictionary which contains
            the configurations of the beta schedule
        '''
        super().__init__(**kwargs)
        self.denoise_fn = denoise_fn
        self.training_target = training_target
        self.beta_schedule_args = beta_schedule_args
        self.set_new_noise_schedule(**beta_schedule_args)
        self.optim_args = optim_args
        self.loss = loss_fn
        if loss_fn == 'l2' or loss_fn == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_fn == 'l1' or loss_fn == 'mae':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif isinstance(loss_fn, Callable):
            self.loss_fn = loss_fn
        else:
            raise NotImplementedError
        self.clip_denoised = clip_denoised
        self.dynamic_thresholding = dynamic_thresholding
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def set_new_noise_schedule(self, **beta_schedule_args):
        to_torch = partial(torch.tensor, dtype=torch.float32)
        betas = make_beta_schedule(**beta_schedule_args)
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.training_target == "noise":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.training_target == "y_0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.training_target == 'v':
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def thresholding_y_0(self, y_0):
        if not self.dynamic_thresholding:
            return y_0.clamp_(-1., 1.)

        b = y_0.shape[0]
        s = torch.quantile(
            y_0.view(b, -1).abs(),
            self.dynamic_thresholding_percentile,
            dim = -1
        )

        s.clamp_(min = 1.)
        s = right_pad_dims_to(y_0, s)
        return y_0.clamp(-s, s) / s
    
    def get_v(self, y_t, noise, t):
        return (
                extract(self.sqrt_alphas_cumprod, t, y_t.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, y_t.shape) * y_t
        )

    def predict_start_from_noise(self, y_t, t, noise):
        ''' recover y_0 from predicted noise. Reverse of Eq(4) in DDPM paper
        \hat(y_0) = 1 / sqrt[\bar(a)]*y_t - sqrt[(1-\bar(a)) / \bar(a)]*noise'''
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, y_t.shape) * noise
        )

    def predict_start_from_z_and_v(self, y_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, y_t.shape) * y_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, y_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, y_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, y_t.shape) * v +
                extract(self.sqrt_one_minus_alphas_cumprod, t, y_t.shape) * y_t
        )

    def q_posterior(self, y_0_hat, y_t, t):
        ''' predict y_{t-1} from \hat(y_0) and y_t. Eq(7) in DDPM paper
        h_{t-1} = sqrt(\bar(alpha)_t)*beta_t / (1-\bar(alpha_t)) * y_0_hat + 
        sqrt(alpha_t)*(1-\bar(alpha_t-1)) / (1-\bar(alpha_t)) * y_t'''
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, model_kwargs=None):
        '''From y_t to the mean / std  of y_{t-1} (before adding noise)'''
        model_kwargs = default(model_kwargs, {})
        if self.training_target == 'noise':
            y_0_hat = self.predict_start_from_noise(
                    y_t, t=t, noise=self.denoise_fn(y_t, t, **model_kwargs))
        elif self.training_target == 'v':
            y_0_hat = self.predict_start_from_z_and_v(
                y_t, t=t, v=self.denoise_fn(y_t, t, **model_kwargs)
            )
        else:
            y_0_hat = self.denoise_fn(y_t, t, **model_kwargs)

        if self.clip_denoised:
            y_0_hat = self.thresholding_y_0(y_0_hat)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(
                self.sqrt_alphas_cumprod, t, x_start.shape
            ) * x_start +
            extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            ) * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, model_kwargs=None):
        '''From y_t to the y_{t-1}'''
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, model_kwargs=model_kwargs)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

class DDIM_LDMTraining(DDIM_LDM):
    def __init__(
        self, 
        denoise_fn, 
        beta_schedule_args, 
        training_target='noise', 
        loss_fn='l2',
        optim_args={
            'lr': 1e-3,
            'weight_decay': 5e-4
        },
        loss_simple_weight=1.,
        loss_elbo_weight=1.,
        log_args={}, # for record all arguments with self.save_hyperparameters
        use_fast_sampling=False,
        fast_sampler='ddim',
        fast_sampling_steps=50,
        guidance_scale=5.,
        **kwargs
    ):
        super().__init__(
            denoise_fn=denoise_fn, 
            beta_schedule_args=beta_schedule_args, 
            training_target=training_target, 
            loss_fn=loss_fn, 
            optim_args=optim_args,
            **kwargs)
        self.loss_simple_weight = loss_simple_weight
        self.loss_elbo_weight = loss_elbo_weight
        self.log_args = log_args
        self.call_save_hyperparameters()

        assert fast_sampler in ['ddim', 'plms', None]
        assert not (use_fast_sampling and fast_sampler is None)
        self.use_fast_sampling = use_fast_sampling
        self.fast_sampler = fast_sampler
        self.fast_sampling_steps = fast_sampling_steps
        self.guidance_scale = guidance_scale

    def call_save_hyperparameters(self):
        '''write in a separate function so that the inherit class can overwrite it'''
        self.save_hyperparameters(ignore=['denoise_fn'])

    def process_batch(self, y_0, mode):
        assert mode in ['train', 'val', 'test']
        b, *_ = y_0.shape
        noise = torch.randn_like(y_0)
        if mode == 'train':
            t = torch.randint(0, self.num_timesteps, (b,), device=y_0.device).long()
            y_t = self.q_sample(y_0, t, noise=noise)
        else:
            t = torch.full((b,), self.num_timesteps-1, device=y_0.device, dtype=torch.long)
            y_t = self.q_sample(y_0, t, noise=noise)

        model_kwargs = {}
        '''the order of return is 
            1) model input, 
            2) model pred target, 
            3) model time condition
            4) raw image before adding noise
            5) model_kwargs
        '''
        if self.training_target == 'noise':
            return y_t, noise, t, y_0, model_kwargs
        elif self.training_target == 'v':
            target = self.get_v(y_0, noise, t)
            return y_t, target, t, y_0, model_kwargs
        else:
            return y_t, y_0, t, y_0, model_kwargs

    def forward(self, x):
        return self.restore_from_y_T(x)[0]

    def get_loss(self, pred, target, t):
        loss_raw = self.loss_fn(pred, target)
        loss_flat = mean_flat(loss_raw)

        logvar_t = self.logvar.to(self.device)[t]
        loss_simple = loss_flat / torch.exp(logvar_t) + logvar_t
        loss_simple = loss_simple * self.loss_simple_weight
        loss_simple = loss_simple.mean()

        loss_vlb = (self.lvlb_weights[t] * loss_flat).mean()
        loss_vlb = loss_vlb * self.loss_elbo_weight

        loss = loss_simple + loss_vlb
        return loss, loss_simple, loss_vlb

    def training_step(self, batch, batch_idx):
        self.clip_denoised = False
        x, y, t, raw_image, model_kwargs = self.process_batch(batch, mode='train')
        pred = self.denoise_fn(x, t, **model_kwargs)
        loss, loss_simple, loss_vlb = self.get_loss(pred, y, t)
        with torch.no_grad():
            if self.training_target == 'noise':
                y_0_hat = self.predict_start_from_noise(
                    x, t=t, 
                    noise=pred.detach()
                )
            elif self.training_target == 'v':
                y_0_hat = self.predict_start_from_z_and_v(
                    x, t=t, v=self.denoise_fn(x, t, **model_kwargs)
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
            'model_input': x,
            'model_output': pred,
            'y_0_hat': y_0_hat
        }

    def restore_from_y_T(self, y_t, sample_inter=10000, model_kwargs=None):
        b, *_ = y_t.shape
        y_t = default(y_t, lambda: torch.randn_like(y_t))
        y_t_hist = y_t.unsqueeze(1) # bs, time step, im_dim, im_h, im_w
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=y_t.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, model_kwargs)
            if i % sample_inter == 0:
                y_t_hist = torch.cat([y_t_hist, y_t.unsqueeze(1)], dim=1)
        return y_t, y_t_hist

    def validation_step(self, batch, batch_idx):
        y_t, _, _, y_0_image, model_kwargs = self.process_batch(batch, mode='val')
        restored = self.sampling(noise=y_t, model_kwargs=model_kwargs)
        sampled = self.sampling(noise=torch.randn_like(y_t), model_kwargs=model_kwargs)
        return {
            'y_0_image': y_0_image,
            'restore': restored,
            'sampling': sampled
        }

    def test_step(self, batch, batch_idx):
        '''Test is usually not used in a sampling problem'''
        return self.validation_step(batch, batch_idx)

    @torch.no_grad()
    def sampling(self, noise=None, image_size=None, sample_num=10, model_kwargs={}):
        if noise is None and image_size is None:
            raise RuntimeError('Either noise or image size should be given')
        noise = default(noise, lambda: torch.randn(*image_size))
        if self.use_fast_sampling:
            y_0, y_t_hist = self.fast_sampling(noise, model_kwargs=model_kwargs)
        else:
            sample_num = min(self.num_timesteps, sample_num)
            sample_inter = (self.num_timesteps//sample_num)
            y_0, y_t_hist = self.restore_from_y_T(noise, sample_inter, model_kwargs=model_kwargs)
        return {
            'model_output': y_0,
            'model_history_output': y_t_hist
        }

    @torch.no_grad()
    def fast_sampling(self, noise, model_kwargs={}, uncondition_model_kwargs=None):
        if self.fast_sampler == 'ddim':
            from .DDIMSampler import DDIMSampler as FastSampler
        elif self.fast_sampler == 'plms':
            from .PLMSSampler import PLMSSampler as FastSampler
        model_kwargs = default(model_kwargs, {})
        sampler = FastSampler(self.denoise_fn, self.beta_schedule_args, training_target=self.training_target)
        y_0, y_t_hist = sampler.sample(
            S=self.fast_sampling_steps,
            batch_size=noise.shape[0],
            x_T=noise, # when x_T is given, shape is not used
            shape=[3, 64, 64],
            verbose=False,
            eta=0,
            model_kwargs=model_kwargs,
            uncondition_model_kwargs=uncondition_model_kwargs,
            guidance_scale=self.guidance_scale,
        )
        y_t_hist = torch.stack(y_t_hist['x_inter'], dim=1) # bs, n_timestep, dim, h, w
        return y_0, y_t_hist

    def configure_optimizers(self):
        params = list(self.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        optimizer = torch.optim.Adam(self.parameters(), **self.optim_args)
        return optimizer

class DDIM_LDM_VQVAETraining(DDIM_LDMTraining):
    def __init__(
        self, 
        denoise_fn, 
        vqvae_fn,
        beta_schedule_args, 
        training_target='noise', 
        loss_fn='l2', 
        optim_args={
            'lr': 2e-6,
            'weight_decay': 0.
        }, 
        loss_simple_weight=1, 
        loss_elbo_weight=1, 
        unet_init_weights=None,
        vqvae_init_weights=None,
        scale_factor=1.0,
        log_args={}, 
        freeze_pretrained_weights=True,
        use_different_learning_rate=False,
        **kwargs
    ):
        super().__init__(
            denoise_fn=denoise_fn, 
            beta_schedule_args=beta_schedule_args, 
            training_target=training_target, 
            loss_fn=loss_fn, 
            optim_args=optim_args, 
            loss_simple_weight=loss_simple_weight, 
            loss_elbo_weight=loss_elbo_weight, 
            log_args=log_args, 
            **kwargs
        )
        self.freeze_pretrained_weights = freeze_pretrained_weights
        self.use_different_learning_rate = use_different_learning_rate
        self.vqvae_fn = vqvae_fn
        self.scale_factor = scale_factor
        self.initialize_unet(unet_init_weights)
        self.initialize_vqvae(vqvae_init_weights)

    def initialize_unet(self, unet_init_weights):
        if unet_init_weights is not None:
            print(f'INFO: initialize denoising UNet from {unet_init_weights}')
            sd = torch.load(unet_init_weights)
            self.denoise_fn.load_state_dict(sd)

    def initialize_vqvae(self, vqvae_init_weights):
        if vqvae_init_weights is not None:
            print(f'INFO: initialize VQVAE from {vqvae_init_weights}')
            sd = torch.load(vqvae_init_weights)
            self.vqvae_fn.load_state_dict(sd)
            for param in self.vqvae_fn.parameters():
                param.requires_grad = False

    def call_save_hyperparameters(self):
        '''write in a separate function so that the inherit class can overwrite it'''
        self.save_hyperparameters(ignore=['denoise_fn', 'vqvae_fn'])

    @torch.no_grad()
    def encode_image_to_latent(self, x):
        return self.vqvae_fn.encode(x) * self.scale_factor

    @torch.no_grad()
    def decode_latent_to_image(self, x):
        x = x / self.scale_factor
        return self.vqvae_fn.decode(x)

    def process_batch(self, x_0, mode):
        y_0 = self.encode_image_to_latent(x_0)
        y_t, target, t, _, model_kwargs = super().process_batch(y_0, mode)
        return y_t, target, t, x_0, model_kwargs

    def training_step(self, batch, batch_idx):
        """  res_dict = {
            'loss': loss,
            'raw_image': raw_image,
            'model_input': x,
            'model_output': pred,
            'y_0_hat': y_0_hat
        }"""
        res_dict = super().training_step(batch, batch_idx)
        res_dict['y_0_hat'] = self.decode_latent_to_image(res_dict['y_0_hat'])
        return res_dict

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        y_t, _, _, y_0_image, model_kwargs = self.process_batch(batch, mode='val')
        sampled = self.sampling(noise=torch.randn_like(y_t), model_kwargs=model_kwargs)
        return {
            'sampling': sampled
        }

    def restore_from_y_T(self, y_t, sample_inter=10000, model_kwargs=None):
        b, *_ = y_t.shape
        y_t = default(y_t, lambda: torch.randn_like(y_t))
        x_t_hist = self.decode_latent_to_image(y_t).unsqueeze(1) # bs, time step, im_dim, im_h, im_w
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=y_t.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, model_kwargs)
            if i % sample_inter == 0:
                x_t = self.decode_latent_to_image(y_t)
                x_t_hist = torch.cat([x_t_hist, x_t.unsqueeze(1)], dim=1)
        x_0 = self.decode_latent_to_image(y_t)
        return x_0, x_t_hist

    @torch.no_grad()
    def fast_sampling(self, noise, model_kwargs={}, uncondition_model_kwargs=None):
        y_0, y_t_hist = super().fast_sampling(noise, model_kwargs=model_kwargs, uncondition_model_kwargs=uncondition_model_kwargs)
        y_0 = self.decode_latent_to_image(y_0)
        y_t_hist = [self.decode_latent_to_image(y_t_hist[:, i]) for i in range(y_t_hist.shape[1])]
        y_t_hist = torch.stack(y_t_hist, dim=1) # bs, n_timestep, dim, h, w
        return y_0, y_t_hist

    def configure_optimizers(self):
        if self.use_different_learning_rate:
            assert hasattr(self, 'params_not_pretrained')
            assert hasattr(self, 'params_pretrained')
            # todo make it configurable
            params = [
                {'params': self.params_not_pretrained}, 
                {'params': self.params_pretrained, 'lr': 2e-6}
            ]
        else:
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

class DDIM_LDM_Text_VQVAETraining(DDIM_LDM_VQVAETraining):
    def __init__(
        self, 
        denoise_fn, 
        vqvae_fn,
        text_fn,
        beta_schedule_args, 
        training_target='noise', 
        loss_fn='l2', 
        optim_args={
            'lr': 2e-6,
            'weight_decay': 0.
        }, 
        loss_simple_weight=1, 
        loss_elbo_weight=1, 
        unet_init_weights=None,
        vqvae_init_weights=None,
        text_model_init_weights=None,
        scale_factor=1.0,
        log_args={}, 
        **kwargs
    ):
        super().__init__(
            denoise_fn=denoise_fn, 
            vqvae_fn=vqvae_fn,
            beta_schedule_args=beta_schedule_args, 
            training_target=training_target, 
            loss_fn=loss_fn, 
            optim_args=optim_args, 
            loss_simple_weight=loss_simple_weight, 
            loss_elbo_weight=loss_elbo_weight, 
            unet_init_weights=unet_init_weights,
            vqvae_init_weights=vqvae_init_weights,
            scale_factor=scale_factor,
            log_args=log_args, 
            **kwargs
        )
        self.text_fn = text_fn
        self.initialize_text_model(text_model_init_weights)

    def initialize_text_model(self, text_model_init_weights):
        if text_model_init_weights is not None:
            print(f'INFO: initialize text model from {text_model_init_weights}')
            sd = torch.load(text_model_init_weights)
            self.text_fn.load_state_dict(sd)
            for param in self.text_fn.parameters():
                param.requires_grad = False

    def call_save_hyperparameters(self):
        '''write in a separate function so that the inherit class can overwrite it'''
        self.save_hyperparameters(ignore=['denoise_fn', 'vqvae_fn', 'text_fn'])

    @torch.no_grad()
    def encode_text(self, x):
        if isinstance(x, tuple):
            x = list(x)
        return self.text_fn.encode(x)