from torch import nn
from typing import Optional, Union
import numpy as np
from jieba import re
import torch, os, cv2
from PIL import Image
from einops import repeat, rearrange
from torch.nn.parallel import DataParallel, DistributedDataParallel

# from stable_diffusion.ldm.modules.attention import LinearAttention as la
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from stable_diffusion.ldm.modules.diffusionmodules.openaimodel import Upsample, Downsample

from sam.data import get_masked_Image

from sam.dist_util import get_bare_model as bare
"""
                       SAM
        R^3         --------->      segmentation space  (1,3,h,w)
         |                                  |
    D(·) | E(·)                        D(·) | E(·) 
         |       feature extractor          |
    latent space    <---------      latent segmentation
     (1,4,h,w)                           (1,4,h,w)

"""

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


class ProjectionModel(nn.Module):
    # projection via one-shot learning
    # to extract latent feature from segmentation gaining from SAM
    def __init__(self, dim=4, dropout=0.5, use_conv=True):
        # seg 2 latent | judge in latent space
        # channel => dim
        super().__init__()
        # in_channels = out_channels = dim
        if dim == 4:
            self.dim = 4
        elif dim == 8:
            self.dim = dim
            dim //= 2


        self.dropout = dropout

        self.down = Downsample(channels=dim, use_conv=use_conv)    # decline two times via pool, keep channels
        self.up = Upsample(channels=dim, use_conv=use_conv)         # expand two times via conv, keep channels

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(dim)
        )        
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, kernel_size=1, stride=1, padding=1),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(2 * dim),
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(2 * dim),
            nn.Conv2d(2 * dim, dim, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(dim)
        )
        self.conv4 = nn.Conv2d(2 * dim, dim, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # x: 64 * 64
        assert len(x.shape) == 4, f'forward input shape = {x.shape}'
        x0 = self.up(x) if self.dim == 4 else self.up(torch.cat([x] * 2, dim = 1))
                                                                # x0: 4 * 128 * 128
        y0 = self.up(x0)                                        # y0: 4 * 256 * 256
        x1 = self.conv2(x0) + x0                                # x1: 4 * 128 * 128
        y1 = self.up(x1)                                        # y1: 4 * 256 * 256
        x2 = self.conv4(torch.cat([x0, x1], dim = 1))           # x2: 4 * 128 * 128
        y2 = self.down(self.conv2(y1))                          # y2: 4 * 128 * 128
        y3 = self.conv3(y2)                                     # y3: 4 * 128 * 128
        
        xy = self.down(y3) + self.down(x2) + x

        return xy


class ProjectionTo():
    def __init__(self, sam_model, sd_model, pm_model, device='cuda'):
        # sam: mask_generator = SamAutomaticMaskGenerator(sam_model if single_gpu else sam_model.module).generate


        self.device = device
        self.sam_model = bare(sam_model.to(device))
        self.pm_model = bare(pm_model.to(device))
        self.sd_model = bare(sd_model.to(device))

        self.pm_model.eval()
        self.sam_model.eval()       # ???
        self.sd_model.eval()

    def Make_Parralel(self, use_single_gpu=False, local_rank=0):
        if not use_single_gpu:
            if not isinstance(self.pm_model, (DataParallel, DistributedDataParallel)):
                self.pm_model = torch.nn.parallel.DistributedDataParallel(self.pm_model, \
                                            device_ids=[local_rank], output_device=local_rank).module
                self.sd_model = torch.nn.parallel.DistributedDataParallel(self.sd_model, \
                                            device_ids=[local_rank], output_device=local_rank).module
                self.sam_model = torch.nn.parallel.DistributedDataParallel(self.sam_model, \
                                            device_ids=[local_rank], output_device=local_rank).module
        # unused

    @torch.no_grad()
    def load_img(self, Path: str = None, Train: bool = True, max_resolution: int = 512 * 512) -> np.array:
        if Path is not None:
            assert os.path.isfile(Path)
            import math
            hhh = (int)(math.sqrt(max_resolution))
            image = torch.randn((1, 3, hhh, hhh), device=self.device) * 255.
            return image
        else:
            image = Image.open(Path).convert("RGB")
            if Train or Path is None:
                w = h = 512
            else:
                w, h = image.size
                h, w = get_resize_shape((h, w), max_resolution=max_resolution, resize_short_edge=None)
            image = cv2.resize(np.asarray(image, dtype=np.float32), (w, h), interpolation=cv2.INTER_LANCZOS4)
            return image

    @torch.no_grad()
    def MapsTo(self, IMG: Optional[Union[str, np.array]] = None, Type: str = None, Train: bool = True, max_resolution: int = 512*512):
        assert Type in [None, 'R^3=seg', 'R^3=latent', 'seg=seg-latent', 'seg-latent=latent']
        if isinstance(IMG, str):
            R3 = self.load_img(IMG, Type, Train, max_resolution)
        else:
            assert isinstance(IMG, np.array), f'invalid return format in formal step'
            R3 = IMG
        result = []
        Type = re.split(Type.strip(), '[=]')
        assert len(Type) == 2, f'Fatal'

        if Type[0] == 'R^3':
            if Type[1] == 'seg':
                R3 = np.array(R3.astype(np.uint8))
                return get_masked_Image(self.sam(R3)).to(self.device)
            # to np.array
            elif Type[1] == 'latent':
                image = (np.array(R3).astype(np.float32) / 255.)[None].transpose(0, 3, 1, 2)
                image = 2. * torch.from_numpy(image) - 1.
                image = repeat(image, "1 ... -> b ...", b=self.batch_size).to(
                    self.device).clone().detach().requires_grad_(False).to(torch.float32)
                return self.model.get_first_stage_encoding(self.model.encode_first_stage(image))\
                                                                                .to(self.device)
            # to torch.Tensor
            else: raise NotImplementedError('Unrecognized Situation-0')

        elif Type[0] == 'seg':
            # input: np.array
            if Type[1] == 'seg-latent':
                image = (np.array(R3).astype(np.float32) / 255.)[None].transpose(0, 3, 1, 2)
                image = 2. * torch.from_numpy(image) - 1.
                image = repeat(image, "1 ... -> b ...", b=self.batch_size).to(
                    self.device).clone().detach().requires_grad_(False).to(torch.float32)
                return self.model.get_first_stage_encoding(self.model.encode_first_stage(image)) \
                    .to(self.device)
            # to torch.Tensor
            else: raise NotImplementedError('Unrecognized Situation-1')

        elif Type[0] == 'seg-latent':
            # input: torch.Tensor
            if Type[1] == 'latent':
                # return from 'seg=seg-latent'
                assert isinstance(R3, torch.Tensor), f'???'
                return self.pm_model(R3)
            # to torch.Tensor
            else: raise NotImplementedError('Unrecognized Situation-2')

        else: raise NotImplementedError('Unrecognized Situation-3')