"""
copy from https://github.com/CompVis/latent-diffusion/blob/5a6571e384f9a9b492bbfaca594a2b00cad55279/ldm/models/diffusion/plms.py
SAMPLING ONLY.
"""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from model_utils import default, make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, make_beta_schedule, extract_into_tensor


class PLMSSampler(object):
    def __init__(self, model, beta_schedule_args={
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 0.0015,
        "linear_end": 0.0195
    }):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = beta_schedule_args['n_timestep']
        self.make_full_schedule(**beta_schedule_args)

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_full_schedule(self, **beta_schedule_args):
        betas = make_beta_schedule(**beta_schedule_args)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: torch.tensor(x).to(torch.float32).cuda()

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose
        )

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=self.alphas_cumprod.cpu(), 
            ddim_timesteps=self.ddim_timesteps, 
            eta=ddim_eta,verbose=verbose
        )
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               model_kwargs={},
               uncondition_model_kwargs=None,
               callback=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               verbose=True,
               x_T=None,
               log_every_t=100,
               guidance_scale=1.,
               ):

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        if x_T is not None:
            shape = x_T.shape[1:]
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for PLMS sampling is {size}')

        samples, intermediates = self.plms_sampling(
            size,
            model_kwargs=model_kwargs,
            uncondition_model_kwargs=uncondition_model_kwargs,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask, x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=guidance_scale,
        )
        return samples, intermediates

    @torch.no_grad()
    def plms_sampling(
        self, shape,
        x_T=None, 
        model_kwargs={},
        uncondition_model_kwargs=None,
        ddim_use_original_steps=False,
        callback=None, 
        timesteps=None, 
        quantize_denoised=False,
        mask=None, 
        x0=None, 
        img_callback=None, 
        log_every_t=100,
        temperature=1., 
        noise_dropout=0., 
        unconditional_guidance_scale=1., ):
        device = torch.device("cuda")
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = list(reversed(range(0,timesteps))) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        old_eps = []

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_plms(
                img, ts, 
                index=index, 
                model_kwargs=model_kwargs,
                uncondition_model_kwargs=uncondition_model_kwargs,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised, temperature=temperature,
                noise_dropout=noise_dropout, 
                unconditional_guidance_scale=unconditional_guidance_scale,
                old_eps=old_eps, t_next=ts_next
            )
            img, pred_x0, e_t = outs
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_plms(
        self, x, t, index, 
        model_kwargs={},
        uncondition_model_kwargs=None,
        repeat_noise=False, 
        use_original_steps=False, quantize_denoised=False,
        temperature=1., noise_dropout=0.,
        unconditional_guidance_scale=1., 
        old_eps=None, t_next=None):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):
            e_t = self.model(x, t, **model_kwargs)
            if (uncondition_model_kwargs is not None) and (unconditional_guidance_scale != 1.):
                e_t_uncond = self.model(x, t, **uncondition_model_kwargs)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            return e_t

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = get_model_output(x, t)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t