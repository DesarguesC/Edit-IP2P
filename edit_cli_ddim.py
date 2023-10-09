from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser

import einops
# import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
from basicsr.utils import tensor2img

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.inference_base import (diffusion_inference, get_base_argument_parser, get_sd_models, str2bool)
from stable_diffusion.ldm.models.diffusion.ddim_edit import DDIMSampler
from stable_diffusion.ldm.util_ddim import instantiate_from_config, get_resize_shape, load_img
from stable_diffusion.ldm.util_ddim import DEFAULT_NEGATIVE_PROMPT as DNP


# think about whether to use the description of original image as negative_prompt?


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)
    
    
        


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def main():
    parser = get_base_argument_parser()
    parser.add_argument("--config", default="configs/ip2p-ddim.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00.ckpt", type=str)
    parser.add_argument("--input", default=None, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--edit_prompt", required=True, type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"


    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()    
    
    sampler = DDIMSampler(model)
    batch_size = args.n_samples

    null_token = model.get_learned_conditioning([DNP() if args.use_neg_prompt else ""])
    seed = random.randint(0, 100000) if args.seed is None else args.seed

    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        
        input_image, args = load_img(args)        
        if args.edit_prompt == "":
            input_image.save(args.output)
            return        
        input_image = input_image.to(args.device)  # ?
        input_image = repeat(input_image, '1 ... -> b ...', b=batch_size)
        print(f'input_image shape: {input_image.shape}')
        
        cond = {}
        cond["c_crossattn"] = model.get_learned_conditioning([args.edit_prompt])
        cond["c_concat"] = model.encode_first_stage(input_image).mode()
        # tt = cond['c_concat']
        # tmp = model.get_first_stage_encoding(cond['c_concat'])
        # print(f'get_first_stage_encoding: {tt.shape} -> {tmp.shape}')
        
        
        # model.get_first_stage_encoding(model.encode_first_stage(input_image).mode())    # .mode() -> ?
        # assert input_image.shape == (1,args.C, args.H, args.W), f'Wrongly shaped input_image as {input_image.shape}, while shape in args is {(1, args.C, args.H, args.W)}'
        
        # init image in latent space
        # model.get_first_stage_encoding(model.encode_first_stage(init_image))
        # TODO: init latent space directly?

        uncond = {}
        uncond["c_crossattn"] = null_token
        uncond["c_concat"] = torch.zeros_like(cond["c_concat"])
        assert uncond['c_concat'] is not None
        
        """
            unconditioned image: torch.zeros_like(cond["c_concat"])
            conditioned image: model.encode_first_stage(input_image).mode()
        """


        # sigmas = model_wrap.get_sigmas(args.steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
        }
        torch.manual_seed(seed)
        # z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        # z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)

        x = diffusion_inference(args, model, sampler, adapter_features=None, append_to_context=None, **extra_args)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")

        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())

    edited_image.save(args.output)


if __name__ == "__main__":
    main()
