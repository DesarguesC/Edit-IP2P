"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from einops import repeat

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               callback=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,           # noise during timesteps
               img_cond=None,       # conditioned image
               img_uncond=None,    # unconditioned image
               log_every_t=100,
               prompt_guidance_scale=1.,
               image_guidance_scale=1.,
               conditioning=None,
               unconditional_conditioning=None, # prompt = \varnothing
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T, img_cond=img_cond, img_uncond=img_uncond, 
                                                    log_every_t=log_every_t,
                                                    prompt_guidance_scale=prompt_guidance_scale,
                                                    image_guidance_scale=image_guidance_scale,
                                                    conditioning=conditioning,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    **kwargs
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, shape, x_T=None,
                      img_cond=None, img_uncond=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      prompt_guidance_scale=1., image_guidance_scale=1.,
                      conditioning=None, unconditional_conditioning=None, **kwargs):
        device = self.model.betas.device
        b = shape[0]   # batch size
        
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
            
        if image_guidance_scale != None\
                and image_guidance_scale != 1:
            assert img_cond != None
        # unconditioning image: torch.zeros_like(...)[0]

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

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            
            # img: history in diffusion process
            # print(f'Before q_sasmple_ddim img shape: {img.shape}')
            outs = self.p_sample_ddim(x=img, t=ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      prompt_guidance_scale=prompt_guidance_scale,
                                      image_guidance_scale=image_guidance_scale,
                                      cond=conditioning, unconditional_conditioning=unconditional_conditioning,
                                      img_cond=img_cond, img_uncond=img_uncond, **kwargs)

            # TODO: consider img_cond into iteration

            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)
            # print(f'After q_sasmple_ddim img shape: {img.shape}')
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      prompt_guidance_scale=1., image_guidance_scale=1.,
                      cond=None, unconditional_conditioning=None, 
                      img_cond=None, img_uncond=None, **kwargs):
        b, *_, device = *x.shape, x.device
        # assert 'seg_cond_latent' in kwargs.keys() and 'projection' in kwargs.keys() and 'adapter' in kwargs.keys() \
        #             and 'time_emb' in kwargs.keys() and 'seg_uncond_latent' in kwargs.keys(), f'kwargs.keys = {kwargs.keys()}'
        # print(f'single x.shape={x.shape}')
        
        seg_cond_latent = kwargs['seg_cond_latent']
        seg_uncond_latent = kwargs['seg_uncond_latent']
        cond_pm = kwargs['cond_pm']
        uncond_pm = kwargs['uncond_pm']
        
        
        if unconditional_conditioning is None or prompt_guidance_scale == 1. and image_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, cond)
        elif image_guidance_scale == 1. :
            # no unconditioning image guidance, only prompt guidance
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            
            c_in = [
                [unconditional_conditioning, cond],
                [img_uncond, img_cond],
                [seg_uncond_latent, seg_cond_latent],
                [uncond_pm, cond_pm]
            ]

            e_t_uncond_prompt, e_t = self.model.apply_model(x_in, t_in, c_in, **kwargs).chunk(2)
            e_t = e_t_uncond_prompt + prompt_guidance_scale * (e_t - e_t_uncond_prompt)
        else:
            # both image conditioning and prompt conditioning
            x_in = torch.cat([x] * 3)
            t_in = torch.cat([t] * 3)
            
            # print(f'cond.shape = {cond.shape}')
            # print(f'unconditional_conditioning.shpae = {unconditional_conditioning.shape}')
            # print(f'img_cond.shape = {img_cond.shape}')
            # print(f'img_uncond.shape = {img_uncond.shape}')
            # print(f'seg_cond_latent.shape = {seg_cond_latent.shape}')
            # print(f'seg_uncond_latent.shape = {seg_uncond_latent.shape}')
            # print(f'cond_pm.shape = {cond_pm.shape}')
            # print(f'uncond_pm.shape = {uncond_pm.shape}')
            
            c_in = [
                [unconditional_conditioning, unconditional_conditioning, cond],   # prompt
                [img_uncond, img_cond, img_cond],                                 # image latent
                [seg_uncond_latent, seg_cond_latent, seg_cond_latent],            # seg cond latent
                [uncond_pm, cond_pm, cond_pm]                                     # projected seg cond
            ]
            
            # concat prompt vector and latent image to constrain the generation simultaneously, implement cfg for ControlNet constrains.
            e_t_uncond, e_t_uncond_prompt, e_t = self.model.apply_model(x_noisy=x_in, t=t_in, cond=c_in, **kwargs).chunk(3, dim=0)   # self.model.apply_model
            e_t = e_t_uncond + image_guidance_scale * (e_t_uncond_prompt - e_t_uncond) + prompt_guidance_scale * (e_t - e_t_uncond_prompt)
            # if self.model.apply_model.diffusion_model.in_channels == 4:
            #     e_t, _ = e_t.chunk(2,dim=2)
            # print(e_t.shape)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, cond, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
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

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec