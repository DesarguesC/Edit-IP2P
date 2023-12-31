from __future__ import annotations

import math
import random
import sys, os
from argparse import ArgumentParser
from sam.util import show_anns

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast


from basicsr.utils import tensor2img, img2tensor


sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util_ddim import instantiate_from_config, load_model_from_config
from stable_diffusion.ldm.inference_base import str2bool

device = "cuda"

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        # print("crossattn shape in CCFGDenoiser: ", cond["c_crossattn"][0].shape)
        # print("c_concat shape in CCFGDenoiser: ", cond["c_concat"][0].shape)
        # cfg_cond = {
        #     "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], cond["c_crossattn"][0], uncond["c_crossattn"][0]])],
        #     "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0], uncond["c_concat"][0]])],
        # }
        # out_cond, out_img_cond, out_txt_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(4)
        # return out_uncond + image_cfg_scale * (out_cond - out_img_cond) + text_cfg_scale * (out_cond - out_txt_cond)
    
        
        # cfg_cond = {
        #     "c_crossattn": [torch.cat([cond["c_crossattn"][0], cond["c_crossattn"][0], uncond["c_crossattn"][0]])],
        #     "c_concat": [torch.cat([cond["c_concat"][0], uncond["c_concat"][0], uncond["c_concat"][0]])],
        # }
        # out_cond, out_txt_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        # return out_uncond + image_cfg_scale * (out_cond - out_txt_cond) + text_cfg_scale * (out_txt_cond - out_uncond)
        
        # origin
        
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)





def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00.ckpt", type=str)
    parser.add_argument("--pth", default="checkpoints/control_sd15_seg.pth", type=str)
    
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--edit", required=True, type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--SAMldm", type=str2bool, default=True)
    parser.add_argument("--reverse", type=str2bool, default=True)
    parser.add_argument("--sam-ckpt", default="../SAM/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--sam-type", default="vit_h", type=str)
    
    
    
    
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    seed = random.randint(0, 100000) if args.seed is None else args.seed
    input_image = Image.open(args.input).convert("RGB")
    width, height = input_image.size
    factor = args.resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)


    if args.edit == "":
        input_image.save(args.output)
        return

    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
        print(input_image.shape)
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(args.steps)
        
        if not os.path.isfile(args.sam_ckpt):
            raise NotImplememtException('no such file')
        
        # SAM in latent space
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        
        if args.SAMldm:
            np_image = tensor2img(input_image)
            name_list = args.input.split('/')
            name_list = name_list[-1].split('.')
            name = name_list[0]
            
            sam = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt)
            sam.to(device=device)
            mask_generator = SamAutomaticMaskGenerator(sam)
            masks = mask_generator.generate(np_image)
            # sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
            
            li = args.output.split('/')
            base = '.'
            for x in li:
                if x != li[-1]:
                    base = os.path.join(base, x)
            
            valid_seg = show_anns(masks, name, base_path=base, reverse=args.reverse)
            seg_cond = img2tensor(valid_seg, bgr2rgb=True, float32=True) / 255.
            
        extra_args = {
            "cond": cond,
            "uncond": uncond,
            # "seg_cond": seg_cond if args.SAMldm else None,
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
        }
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        print(z.shape, extra_args['cond']['c_concat'][0].shape)
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
    edited_image.save(args.output)


if __name__ == "__main__":
    main()
