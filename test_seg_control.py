import torch, cv2, os, sys
import os.path as osp
import numpy as np
import argparse

sys.path.append('../')
sys.path.append('./stable_diffusion')

from stable_diffusion.ldm.util_ddim import (load_inference_train, reduce_tensor, img2latent, img2seg, seg2latent, sl2latent, load_img)
from stable_diffusion.ldm.models.diffusion.ddim_edit import DDIMSampler
from stable_diffusion.ldm.inference_base import (str2bool, diffusion_inference)
from stable_diffusion.eldm.adapter import Adapter
from basicsr.utils import (get_root_logger, get_time_str,
                           scandir, tensor2img)
from sam.dist_util import init_dist, master_only, get_bare_model, get_dist_info
from stable_diffusion.ldm.data.ip2p import Ip2pDatasets
from sam.seg2latent import ProjectionModel as PM
from sam.data import DataCreator
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from stable_diffusion.ldm.inference_base import get_base_argument_parser as gbap


def mkdir(path: str, rank) -> str:
    if not osp.exists(path):
        os.makedirs(path)
    base_count = len(os.listdir(path)) if osp.exists(path) else 0
    path = osp.join(path, f'{base_count:05}--rank:{rank}')

    while True:
        """
            Concurrency:
                when using multi-gpu
        """
        if not osp.exists(path):
            os.makedirs(path)
            break
        else:
            base_count += 1

    os.makedirs(path, exist_ok=True)
    os.makedirs(osp.join(path, 'models'))
    os.makedirs(osp.join(path, 'training_states'))
    os.makedirs(osp.join(path, 'visualization'))
    return path


def parsr_args():
    parser = gbap()
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    # we use DDIM
    parser.add_argument(
        "--sd_ckpt",
        type=str,
        default="./checkpoints/v1-5-pruned.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        '--sam_type',
        type=str,
        default='vit_h',
        help='choose sam_type according to SAM official documentation'
    )
    parser.add_argument(
        '--sam_ckpt',
        type=str,
        default='./checkpoints/sam_vit_h_4b8939.pth',
        help='sam ckpt path'
    )
    parser.add_argument(
        "--pm_pth",
        type=str,
        default="./checkpoints/model.pth",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        '--ls_path',
        type=str,
        default='./checkpoints/ls_model.pth',
        help='LatentSegmentAdapter model path'
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default='./example/1.jpg',
        help='path to an input image'
    )
    parser.add_argument(
        '--edit',
        type=str,
        default='',
        help='edit prompt / guidance / instructions'
    )
    parser.add_argument(
        "--txt_scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--img_scale",
        type=float,
        default=1.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        '--adapter_time_emb',
        type=str2bool,
        default=False,
        help='whether to add embedded time feature into adapter'
    )
    opt = parser.parse_args()
    return opt

def main():
    opt = parsr_args()
    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        
    sd_model, sam_model, pm_model, configs = load_inference_train(opt, opt.device)
    
    # LatentSegAdapter = Adapter(cin=8*16, channels=[64, 128, 256, 64], nums_rb=2, ksize=1, sk=True, use_conv=False, use_time=False)
    # opt.adapter_time_emb = False

    # no time embedding weights have been trained

    LatentSegAdapter = Adapter(cin=8 * 16, channels=[256, 512, 1024, 1024], nums_rb=2, ksize=1, sk=True, use_conv=False,
                               use_time=True).to(opt.device)
    opt.adapter_time_emb = True

    LatentSegAdapter.load_state_dict(torch.load(opt.ls_path))
    LatentSegAdapter.eval().to(opt.device)
    mask_generator = SamAutomaticMaskGenerator(sam_model).generate
    sampler = DDIMSampler(sd_model)
    Cin_Models = {
        'opt': opt,
        'sampler': sampler,
        'sd_model': sd_model,
        'sam': mask_generator,
        'pm': pm_model,
        'adapter': LatentSegAdapter,
        'img_path': opt.input_image,
        'edit': opt.edit
    }

    cout = diffusion_inference(**Cin_Models)
    opt.outdir = os.path.join(opt.outdir, 'output.png') if not opt.outdir.endswith('.png') else opt.outdir
    cv2.imwrite(opt.outdir, tensor2img(cout))


    return


if __name__ == '__main__':
    main()
    print('finished.')
