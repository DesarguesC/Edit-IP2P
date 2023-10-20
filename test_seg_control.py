import torch, cv2, os, sys
import os.path as osp
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import kl_div
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist

sys.path.append('../')
sys.path.append('./stable_diffusion')

from stable_diffusion.ldm.util_ddim import (load_inference_train, reduce_tensor, img2latent, img2seg, seg2latent, sl2latent)
from stable_diffusion.ldm.inference_base import str2bool
from stable_diffusion.eldm.adapter import Adapter
from basicsr.utils import (get_root_logger, get_time_str,
                           scandir, tensor2img)
import logging
from sam.dist_util import init_dist, master_only, get_bare_model, get_dist_info
from stable_diffusion.ldm.data.ip2p import Ip2pDatasets
from sam.seg2latent import ProjectionModel as PM
from sam.data import DataCreator
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bsize",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
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
        "--config",
        type=str,
        default="./configs/ip2p-ddim.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train_seg_control",
        help="experiment name",
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=512 * 512,            # use cat-control, cat at channels: 2 * 512
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512 * 2,            # use cat-control, cat at channels: 2 * 512
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
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
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
    
    LatentSegAdapter = Adapter(cin=8*16, channels=[64, 128, 256, 64], nums_rb=2, ksize=1, sk=True, use_conv=False, use_time=opt.adapter_time_emb)
    LatentSegAdapter.load_state_dict(torch.load(opt.ls_path))
    LatentSegAdapter.eval().to(opt.device)
    mask_generator = SamAutomaticMaskGenerator(sam_model).generate
    
    Models = {
        'projection': pm_model if opt.use_single_gpu else pm_model.module,
        'adapter': LatentSegAdapter if opt.use_single_gpu else LatentSegAdapter.module,
        'time_emb': opt.adapter_time_emb
    }





    return


if __name__ == '__main__':
    main()
    print('finished.')
