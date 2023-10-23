import argparse
import torch, cv2
import numpy as np
from omegaconf import OmegaConf
from einops import repeat, rearrange

from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from stable_diffusion.ldm.models.diffusion.plms import PLMSSampler

from stable_diffusion.ldm.util_ddim import (
    fix_cond_shapes,DEFAULT_NEGATIVE_PROMPT,
    load_model_from_config,
    resize_numpy_image,
    img2latent, img2seg, seg2latent, sl2latent
)
# import mmpose

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_base_argument_parser() -> argparse.ArgumentParser:
    """get the base argument parser for inference scripts"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--outdir',
        type=str,
        help='dir to write results to',
        default='./models/ori_out/',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/ip2p-ddim.yaml',
        help='train / inference configs'
    )
    parser.add_argument(
        '--use_neg_prompt',
        type=str2bool,
        default=1,
        help='whether to use default negative prompt',
    )


    parser.add_argument(
        '--cond_inp_type',
        type=str,
        default='image',
        help='the type of the input condition image, take depth T2I as example, the input can be raw image, '
        'which depth will be calculated, or the input can be a directly a depth map image',
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='number of sampling steps',
    )

    parser.add_argument(
        '--vae_ckpt',
        type=str,
        default=None,
        help='vae checkpoint, anime SD models usually have seperate vae ckpt that need to be loaded',
    )

    parser.add_argument(
        '--adapter_ckpt',
        type=str,
        default=None,
        help='path to checkpoint of adapter',
    )

    parser.add_argument(
        '--max_resolution',
        type=float,
        default=512 * 512,
        help='max image height * width, only for computer with limited vram',
    )

    parser.add_argument(
        '--resize_short_edge',
        type=int,
        default=None,
        help='resize short edge of the input image, if this arg is set, max_resolution will not be used',
    )
    parser.add_argument(
        '--cond_tau',
        type=float,
        default=1.0,
        help='timestamp parameter that determines until which step the adapter is applied, '
        'similar as Prompt-to-Prompt tau')

    parser.add_argument(
        '--cond_weight',
        type=float,
        default=1.0,
        help='the adapter features are multiplied by the cond_weight. The larger the cond_weight, the more aligned '
        'the generated image and condition will be, but the generated quality may be reduced',
    )
    
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1,
        help='namely the batch size, set as 1 in inference'
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512 * 2,            # use cat-control, cat at channels -> 2 * 512
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )

    

    return parser


def get_sd_models(opt):
    """
    build stable diffusion model, sampler
    """
    # SD
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, opt.sd_ckpt, opt.vae_ckpt)
    sd_model = model.to(opt.device)

    # sampler
    if opt.sampler == 'plms':
        sampler = PLMSSampler(model)
    elif opt.sampler == 'ddim':
        sampler = DDIMSampler(model)
    else:
        raise NotImplementedError

    return sd_model, sampler




def diffusion_inference(opt, sd_model, sampler, sam, pm, adapter, img_path, edit):
    # sampler: DDIMSampler
    device = opt.device
    cin_img = cv2.imread(img_path)
    numpy_cin = resize_numpy_image(cin_img, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
    # print(f'numpy_cin.shape = {numpy_cin.shape}')
    
    cin_latent = img2latent(numpy_cin, sd_model, device)   #   => to init inference x_T
    cin_uncond = np.random.randint(low=0, high=256, size=numpy_cin.shape, dtype=np.uint8)
    cin_uncond_latent = img2latent(cin_uncond, sd_model, device)
    
    
    
    # consider whether to init diffusion x_T with input image
    cin_seg = img2seg(numpy_cin, sam, device)
    cin_seg_latent = seg2latent(cin_seg, sd_model, device)
    cin_uncond_seg = img2seg(cin_uncond, sam, device)
    cin_uncond_seg_latent = seg2latent(cin_uncond_seg, sd_model, device)
    # print(f'cin_seg_latent.shape = {cin_seg_latent.shape}')
    # print(f'cin_uncond_seg_latent.shape = {cin_uncond_seg_latent.shape}')
    
    images = {
        'cond': cin_latent.to(device), 
        'uncond': cin_uncond_latent.to(device), 
    }
    
    # seg_latent = seg2latent(cin_seg_latent, pm, device)
    # print(f'seg_latent.shape = {seg_latent.shape}')
    x_T_init = cin_latent
    
    cond_pm = sl2latent(cin_seg_latent, pm, device)
    print(f'cond_pm.shape = {cond_pm.shape}')
    uncond_pm = sl2latent(cin_uncond_latent, pm, device)
    print(f'uncond_pm.shape = {uncond_pm.shape}')
    
    
    prompts = {
        'cond': sd_model.get_learned_conditioning(["do not modify"]).to(device),
        'uncond': sd_model.get_learned_conditioning([edit]).to(device)
    }
    
    

    try:
        prompts['cond'], prompts['uncond'] = fix_cond_shapes(sd_model, prompts['cond'], prompts['uncond'])
    except Exception as err:
        print(err)
        raise NotImplementedError('Possibly caused by keys not in extra_args')
    
    prompts['cond'] = repeat(prompts['cond'], '1 ... -> b ...', b=opt.n_samples).to(device)
    prompts['uncond'] = repeat(prompts['uncond'], '1 ... -> b ...', b=opt.n_samples).to(device)
    # prompts['cond'], prompts['uncond'] = c, uc

    kwargs = {
        'cond_pm': cond_pm.to(device),
        'uncond_pm': uncond_pm.to(device),
        'seg_uncond_latent': cin_uncond_seg_latent.to(device),
        'seg_cond_latent': cin_seg_latent.to(device),
        'adapter': adapter.to(device),
        'use_time_emb': opt.adapter_time_emb
    }

    _, opt.C, opt.H, opt.W = cin_latent.shape

    shape = [opt.C, opt.H , opt.W]

    samples_latents, _ = sampler.sample(
        S=opt.steps,
        batch_size=opt.n_samples,
        shape=shape,
        verbose=False,
        conditioning=prompts['cond'],
        unconditional_conditioning=prompts['uncond'],
        x_T=None,   # TODO: use cin_latent ?
        img_cond=images['cond'],             # cin_image    -> seg_cond image
        img_uncond=images['uncond'],         # random image -> seg_cond image
        prompt_guidance_scale=opt.txt_scale,
        image_guidance_scale=opt.img_scale,
        cond_tau=opt.cond_tau,
        **kwargs
    )

    x_samples = sd_model.decode_first_stage(samples_latents)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    return x_samples


