import importlib
import math
from omegaconf import OmegaConf
import cv2
import torch
import numpy as np

import os
from safetensors.torch import load_file

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



def DEFAULT_NEGATIVE_PROMPT() -> str:
    return 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                          'fewer digits, cropped, worst quality, low quality'

def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('assets/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


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

checkpoint_dict_replacements = {
    'cond_stage_model.transformer.text_model.embeddings.': 'cond_stage_model.transformer.embeddings.',
    'cond_stage_model.transformer.text_model.encoder.': 'cond_stage_model.transformer.encoder.',
    'cond_stage_model.transformer.text_model.final_layer_norm.': 'cond_stage_model.transformer.final_layer_norm.',
}


def transform_checkpoint_dict_key(k):
    for text, replacement in checkpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k


def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd


def read_state_dict(checkpoint_file, print_global_state=False):
    _, extension = os.path.splitext(checkpoint_file)
    if extension.lower() == ".safetensors":
        pl_sd = load_file(checkpoint_file, device='cpu')
    else:
        pl_sd = torch.load(checkpoint_file, map_location='cpu')

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = get_state_dict_from_checkpoint(pl_sd)
    return sd


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    sd = read_state_dict(ckpt)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if 'anything' in ckpt.lower() and vae_ckpt is None:
        vae_ckpt = 'models/anything-v4.0.vae.pt'

    if vae_ckpt is not None and vae_ckpt != 'None':
        print(f"Loading vae model from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")
        if "global_step" in vae_sd:
            print(f"Global Step: {vae_sd['global_step']}")
        sd = vae_sd["state_dict"]
        m, u = model.first_stage_model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

    model.cuda()
    model.eval()
    return model


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None, opt=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    
    if opt is not None:
        try:
            h *= opt.fac
            w *= opt.fac
        except:
            raise NotImplementedError
    
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image

def get_resize_shape(image_shape, max_resolution=512 * 512, resize_short_edge=None) -> tuple:
    # print('resize: ', image_shape)
    h, w = image_shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = ((h * k) // 64) * 64 
    w = ((w * k) // 64) * 64 
    return (int)(h), (int)(w)


# make uc and prompt shapes match via padding for long prompts
null_cond = None

def fix_cond_shapes(model, prompt_condition, uc):
    if uc is None:
        return prompt_condition, uc
    global null_cond
    if null_cond is None:
        null_cond = model.get_learned_conditioning([''])
    """
        uc -> null_cond = model.get_learned_conditioning([DNP()])
    """ 
    while prompt_condition.shape[1] > uc.shape[1]:
        uc = torch.cat((uc, null_cond.repeat((uc.shape[0], 1, 1))), axis=1)
    while prompt_condition.shape[1] < uc.shape[1]:
        prompt_condition = torch.cat((prompt_condition, null_cond.repeat((prompt_condition.shape[0], 1, 1))), axis=1)
    print(f'prompt_condition.shape = {prompt_condition.shape}, uc.shape={uc.shape}')
    return prompt_condition, uc


def load_target_model(config, sd_ckpt, vae_ckpt, sam_ckpt, sam_type, device):
    print('loading configs...')

    config = OmegaConf.load(config)
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
    sam.eval().to(device)

    model = load_model_from_config(config, sd_ckpt, vae_ckpt)
    model.eval().to(device)
    # encoder = model.encode_first_stage

    return model, sam, config


def load_img(opt=None, path=None):
    
    if opt != None:
        assert opt.f != None
        assert opt.f > 0, f'downsample factor = {opt.f}'
    
    path = opt.input if opt != None else path
    assert path != None, "no path when reading images"
    if path is not None:
        assert os.path.isfile(path), f'input image path = {path}, file not exists.'
    else:
        import math
        opt.W = opt.H = w = h = (int)(math.sqrt(opt.max_resolution))
        image = torch.randn((1,3,w,h), device=opt.device)
        return 2. * image - 1., opt
    
    image = Image.open(path).convert("RGB")
    w, h = image.size   # check
    h, w = get_resize_shape((h,w), max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge) \
                                                                                    if opt != None else (256,256)
    if opt != None:
        print(f"loaded input image of size ({w}, {h}) from {path}")

    image = np.asarray(image, dtype=np.float32)
    if opt != None:
        opt.W = (int)(w)
        opt.H = (int)(h)
    image = cv2.resize(image, (w,h), interpolation=cv2.INTER_LANCZOS4)

    # used in sam.data, return: np.image ( type: np.float32 )
    
    if opt is None:
        return image, None
    
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    
    return 2. * image - 1., opt

