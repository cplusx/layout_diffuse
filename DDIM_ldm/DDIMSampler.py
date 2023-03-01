import torch
import numpy as np
from tqdm import tqdm
from model_utils import default, make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, make_beta_schedule, extract_into_tensor

class DDIMSampler(object):
    def __init__(self, model, beta_schedule_args={
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 0.0015,
        "linear_end": 0.0195
    }, training_target='noise'):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model
        self.ddpm_num_timesteps = beta_schedule_args['n_timestep']
        self.make_full_schedule(**beta_schedule_args)
        self.training_target = training_target

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda") and self.device == torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_full_schedule(self, **beta_schedule_args):
        betas = make_beta_schedule(**beta_schedule_args)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: torch.tensor(x).to(torch.float32).to(self.device)

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

    def predict_start_from_z_and_v(self, y_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, y_t.shape) * y_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, y_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, y_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, y_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, y_t.shape) * y_t
        )

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               eta=0.,
               verbose=True,
               x_T=None,
               log_every_t=100,
               model_kwargs={},
               uncondition_model_kwargs=None,
               guidance_scale=1.
               ):

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        if shape is None:
            shape = x_T.shape[1:]
        C, H, W = shape
        size = (batch_size, C, H, W)

        samples, intermediates = self.ddim_sampling(
            size,
            ddim_use_original_steps=False,
            x_T=x_T,
            log_every_t=log_every_t,
            model_kwargs=model_kwargs,
            uncondition_model_kwargs=uncondition_model_kwargs,
            guidance_scale=guidance_scale
            )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(
        self, 
        shape,
        x_T=None, 
        ddim_use_original_steps=False,
        timesteps=None, 
        log_every_t=100,
        model_kwargs={},
        uncondition_model_kwargs=None,
        guidance_scale=1.
    ):
        device = self.device
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
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        # if guidance_scale > 1 and uncondition_model_kwargs is not None:
        #     print(f'INFO: guidance scale {guidance_scale} during classifier free guidance with {uncondition_model_kwargs}')
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim(
                img, ts, 
                index=index, 
                use_original_steps=ddim_use_original_steps,
                model_kwargs=model_kwargs,
                uncondition_model_kwargs=uncondition_model_kwargs,
                guidance_scale=guidance_scale
            )
            img, pred_x0 = outs

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(
        self, x, t, index, 
        repeat_noise=False, 
        use_original_steps=False, 
        model_kwargs={},
        uncondition_model_kwargs=None,
        guidance_scale=1.
    ):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):
            model_output = self.model(x, t, **model_kwargs)
            if uncondition_model_kwargs is not None and guidance_scale > 1.:
                model_output_uncond = self.model(x, t, **uncondition_model_kwargs)
                model_output = model_output_uncond + guidance_scale * (model_output - model_output_uncond)

            if self.training_target == "v":
                e_t = self.predict_eps_from_z_and_v(x, t, model_output)
            else:
                e_t = model_output
            return e_t, model_output

        e_t, model_output = get_model_output(x, t)

        alphas = self.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.training_target != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.predict_start_from_z_and_v(x, t, model_output)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
