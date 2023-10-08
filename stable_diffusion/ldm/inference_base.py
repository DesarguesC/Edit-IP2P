import argparse
import torch
from omegaconf import OmegaConf
from einops import repeat

from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from stable_diffusion.ldm.models.diffusion.plms import PLMSSampler
# from ldm.modules.encoders.adapter import Adapter, StyleAdapter, Adapter_light
# from ldm.modules.extra_condition.api import ExtraCondition
from stable_diffusion.ldm.util_ddim import fix_cond_shapes, load_model_from_config, read_state_dict, get_resize_shape
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
        '--use_neg_prompt',
        type=str2bool,
        default=1,
        help='whether to use default negative prompt',
    )

    parser.add_argument(
        '--cond_path',
        type=str,
        default='./models/test.png',
        help='condition image path',
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
        default='./models/t2iadapter_openpose_sd14v1.pth',
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
        '--C',
        type=int,
        default=4,
        help='latent channels / batch size / n_samples',
    )

    parser.add_argument(
        '--f',
        type=int,
        default=8,
        help='downsampling factor (after running Encode, mapped into latent space)',
    )

    parser.add_argument(
        '--scale',
        type=float,
        default=7.5,
        help='unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))',
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




def diffusion_inference(opt, model, sampler, adapter_features=None, append_to_context=None, **extra_args):
    # get text embedding
    # c = model.get_learned_conditioning([opt.edit_prompt])
    # if opt.scale != 1.0:
    #     uc = model.get_learned_conditioning([opt.neg_prompt])
    # else:
    #     uc = None
    try:
        c, uc = fix_cond_shapes(model, extra_args['cond']['c_crossattn'], extra_args['uncond']['c_crossattn'])
    except Exception as err:
        print(err)
        print(extra_args.keys())
        raise NotImplementedError('Possibly caused by keys not in extra_args')
    
    c = repeat(c, '1 ... -> b ...', b=opt.n_samples)
    uc = repeat(uc, '1 ... -> b ...', b=opt.n_samples)

    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

    samples_latents, _ = sampler.sample(
        S=opt.steps,
        conditioning=c,
        batch_size=opt.n_samples,
        shape=shape,
        verbose=False,
        unconditional_conditioning=uc,
        x_T=None,
        img_cond=extra_args['cond']['c_concat'],
        img_uncond=extra_args['uncond']['c_concat'],
        prompt_guidance_scale=extra_args['text_cfg_scale'],
        image_guidance_scale=extra_args['image_cfg_scale'],
        features_adapter=adapter_features,
        append_to_context=append_to_context,
        cond_tau=opt.cond_tau,
    )

    x_samples = model.decode_first_stage(samples_latents)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    return x_samples


