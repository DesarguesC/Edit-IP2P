import cv2, torch, os, math, importlib
from omegaconf import OmegaConf
import numpy as np
from einops import rearrange
from safetensors.torch import load_file
from sam.seg2latent import ProjectionModel
from sam.util import get_masked_Image_
from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch.distributed as dist


def reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    if not isinstance(tensor, list):
        tensor = tensor.clone()
        dist.all_reduce(tensor, op)
        tensor.div_(world_size)
        return tensor
    else:
        re = []
        for r in tensor:
            r = r.clone()
            dist.all_reduce(r, op)
            tensor.div_(world_size)
            re.append(r)
        return re

@torch.no_grad()
def img2latent(R3: np.ndarray, model, device):
    # print(f'R3.shape = {R3.shape}')
    if not isinstance(R3, np.ndarray): R3 = R3.numpy()
    image = np.array(R3).astype(np.float32) / 255.
    if len(image.shape) == 3:
        image = rearrange(image, 'h w c -> 1 h w c')
    image = 2. * torch.from_numpy(rearrange(image, 'b h w c -> b c h w')) - 1.
    return model.get_first_stage_encoding(model.encode_first_stage(image.clone().to(torch.float32).\
                                                                   detach().requires_grad_(False).to(device)))

@torch.no_grad()
def img2seg(R3: np.ndarray, model, device):
    # [8, 512, 512, 3]
    if not isinstance(R3, np.ndarray): R3 = R3.detach().cpu().numpy()
    assert len(R3.shape) >= 3, f'R3.shape = {R3.shape}'
    R3 = np.array(R3.astype(np.uint8))
    if len(R3.shape) == 3:
        R3 = rearrange(R3, 'h w c -> 1 h w c')
    # print(f'img2seg: {R3.shape}')
    ll = R3.shape[0]
    R3_ = [model(R3[i].squeeze()) for i in range(ll)]
    # each R3[0] -> each object
    mask = get_masked_Image_(R3_, use_alpha=False)
    return torch.cat([torch.from_numpy(rearrange(u, 'h w c -> 1 c h w')).clone().\
                                       to(torch.float32).detach().requires_grad_(False) for u in mask], dim=0).to('cpu')


@torch.no_grad()
def seg2latent(seg: torch.Tensor, model, device):
    # [8, 3, 512, 512]
    if not isinstance(seg, np.ndarray): seg = seg.numpy()
    image = 2. * torch.from_numpy(np.array(seg).astype(np.float32) / 255.) - 1.
    return model.get_first_stage_encoding(model.encode_first_stage(image.clone().to(torch.float32).\
                                                                   detach().requires_grad_(False).to(device)))

@torch.no_grad()
def sl2latent(seg: torch.Tensor, model, device):
    # assert isinstance(R3, torch.Tensor), f'type(R3) = {type(R3)}   ???'
    if isinstance(seg, list):
        return torch.cat([model(seg_) for seg_ in seg], dim=0).to(device)
    return model(seg).to(device)


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

def load_img(Path: str = None, Train: bool = True, max_resolution: int = 512 * 512, device='cuda') -> np.array:
    assert Path != None, f'none path when loading image'
    if Path is None:
        assert os.path.isfile(Path)
        hhh = (int)(math.sqrt(max_resolution))
        image = torch.randn((1, 3, hhh, hhh), device=device) * 255.
        image = image.numpy()
        return image
    else:
        image = Image.open(Path).convert("RGB")
        if Train:
            w = h = (int)(math.sqrt(max_resolution))
        else:
            w, h = image.size
            h, w = get_resize_shape((h, w), max_resolution=max_resolution, resize_short_edge=None)
        image = cv2.resize(np.asarray(image, dtype=np.float32), (w, h), interpolation=cv2.INTER_LANCZOS4)
        return image



def load_inference_train(opt, device):
    print('loading configs...')

    config = OmegaConf.load(opt.config)
    sam = sam_model_registry[opt.sam_type](checkpoint=opt.sam_ckpt)
    sam.eval().to(device)

    model = load_model_from_config(config, opt.sd_ckpt, None)
    model.eval().to(device)
    # encoder = model.encode_first_stage

    pm = ProjectionModel()
    state_dict = torch.load(opt.pm_pth)
    pm.load_state_dict(state_dict)
    pm.eval().to(device)

    return model, sam, pm, config


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
                                                                                    if opt != None else (256, 256)
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





