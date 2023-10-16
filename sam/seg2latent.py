from torch import nn
import numpy as np
import torch, os, cv2
from PIL import Image
from einops import repeat, rearrange
from torch.nn.parallel import DataParallel, DistributedDataParallel

# from stable_diffusion.ldm.modules.attention import LinearAttention as la
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from stable_diffusion.ldm.modules.diffusionmodules.openaimodel import Upsample, Downsample
from stable_diffusion.ldm.util_ddim import get_resize_shape
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



class ProjectionModel(nn.Module):
    # projection via one-shot learning
    # to extract latent feature from segmentation gaining from SAM
    def __init__(self, dim=4, dropout=0.5, use_conv=True):
        # seg 2 latent | judge in latent space
        # channel => dim
        super().__init__()
        # in_channels = out_channels = dim
        self.dim = dim
        self.dropout = dropout

        self.down = Downsample(channels=dim, use_conv=use_conv)    # decline two times via pool, keep channels
        self.up = Upsample(channels=dim, use_conv=use_conv)         # expand two times via conv, keep channels

        self.conv1 = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(self.dim)
        )        
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.dim, 2 * self.dim, kernel_size=1, stride=1, padding=1),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(2 * self.dim),
            nn.Conv2d(2 * self.dim, 2 * self.dim, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(2 * self.dim),
            nn.Conv2d(2 * self.dim, self.dim, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(self.dim)
        )
        self.conv4 = nn.Conv2d(2 * self.dim, self.dim, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # x: 64 * 64
        assert len(x.shape) == 4, f'forward input shape = {x.shape}'
        x0 = self.up(x)                                         # x0: 4 * 128 * 128
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


    def MapsTo(self, Path: str = None, Type: str = None, Train: bool = True, max_resolution: int = 512*512):
        assert Type in [None, 'R^3', 'seg', 'latent', 'latent-seg']
        R3 = self.load_img(Path, Type, Train, max_resolution)
        if Type == 'R^3':
            return np.array(R3.astype(np.uint8))
        elif Type == 'seg':
            R3 = np.array(R3.astype(np.uint8))
            return get_masked_Image(self.sam(R3)).to(self.device)
        # visible mask image, np.array type, not torch.tensor
        elif Type == 'latent':
            image = ( np.array(R3).astype(np.float32) / 255.0 )[None].transpose(0, 3, 1, 2)
            image = 2. * torch.from_numpy(image) - 1.
            image = repeat(image, "1 ... -> b ...", b=self.batch_size).to(self.device).clone().detach().requires_grad_(False).to(torch.float32)
            return self.model.get_first_stage_encoding(self.model.encode_first_stage(image)).to(self.device)
        elif Type == 'seg-latent':
            R3 = np.array(R3.astype(np.uint8))
            seg = 2. * torch.from_numpy(get_masked_Image(self.sam(R3)).to(self.device)) - 1.
            seg = rearrange(seg, "h w c -> 1 c h w").clone().detach().requires_grad_(False).to(torch.float32)
            seg_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(seg))
            seg_latent = repeat(seg_latent, "1 ... -> b ...", b=self.batch_size).to(self.device)
            return seg_latent

