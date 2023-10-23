import torch, cv2, os, sys
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import kl_div

sys.path.append('../')
sys.path.append('./stable_diffusion')

from stable_diffusion.ldm.util_ddim import load_target_model
import os.path as osp
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           scandir, tensor2img)
from basicsr.utils.options import copy_opt_file, dict2str
import logging
from sam.dist_util import init_dist, master_only, get_bare_model, get_dist_info

from sam.seg2latent import ProjectionModel as PM
from sam.data import DataCreator
from stable_diffusion.ldm.util_ddim import load_model_from_config
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
        '--bsize',
        type=int,
        default=32,
        help='batch size'
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
        "--local_rank",
        type=int,
        default=0,
        help="local rank",
    )

    

    return parser.parse_args()


def main():
    opt = get_base_argument_parser()
    base_count = 0
    single_gpu = False
    config = './configs/ip2p-ddim.yaml',
    name = 'seg'
    learning_rate = 1.0e-04
    bsize = opt.bsize
    num_workers = 25
    experiments_root = './experiments/'
    
    experiments_root = mkdir(experiments_root)
    log_file = osp.join(experiments_root, f"train_{name}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    
    print_fq = 3
    save_fq = 3
    N = opt.epochs
    
    device_nums = 2
    device = 'cuda'
    
    if not single_gpu:
        
        import torch.multiprocessing as mp
        import torch.distributed as dist
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')
        rank = opt.local_rank
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend='nccl')
        dist.barrier()
        torch.backends.cudnn.benchmark = True
        
    
    loader_params = {
        'config': './configs/ip2p-ddim.yaml',
        'sd_ckpt': './checkpoints/v1-5-pruned-emaonly.ckpt',
        'vae_ckpt': None,
        'sam_ckpt': './checkpoints/sam_vit_h_4b8939.pth',
        'sam_type': 'vit_h',
        'device': device
    }
    
    model, sam_model, configs = load_target_model(**loader_params)
    
    # device = torch.device("cuda:0,1,2" if torch.cuda.is_available() else "cpu")
    # torch.nn.parallel.DistributedDataParallel
    if not single_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank], output_device=local_rank)
        # encoder_model.to(device)
        
    if not single_gpu:
        sam_model = torch.nn.parallel.DistributedDataParallel(
            sam_model,
            device_ids=[local_rank], output_device=local_rank)
        # sam_model.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam_model if single_gpu else sam_model.module).generate
    data_creator_params = {
        # 'image_folder': '../test-dataset/',
        'image_folder': (str)(os.environ['IMAGE_FOLDER']),
        'sd_model': model if single_gpu else model.module,
        'sam': mask_generator,
        'batch_size': 1,
        'downsample_factor': 8,
        'data_scale': 0.8,
        'logger': logger
    }
    
    tot_params = {
        'loader_params': loader_params,
        'data_creator_params': data_creator_params
    }

    data_creator = DataCreator(**data_creator_params)
    
    with torch.no_grad():
        data_creator.MakeData()
    print(f'Randomly loading data with length: {len(data_creator)}')
    
    train_sampler = None if single_gpu else torch.utils.data.distributed.DistributedSampler(data_creator)

    train_dataloader = tqdm(DataLoader(dataset=data_creator, batch_size=bsize, shuffle=True, drop_last=True), \
                            desc='Procedure', total=len(data_creator)) \
        if single_gpu else torch.utils.data.DataLoader(
                                data_creator,
                                batch_size=bsize,
                                shuffle=(train_sampler is None),
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True,
                                sampler=train_sampler)
    
    pm_ = PM().to(device)
    pm_ = pm_ if single_gpu else torch.nn.parallel.DistributedDataParallel(
        pm_,
        device_ids=[local_rank],
        output_device=local_rank)


    # optimizer
    params = list(pm_.parameters() if single_gpu else pm_.module.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate)

    current_iter = 0
    logger.info(f'\n\n\nStart training from epoch: 0, iter: {current_iter}\n')
    
    # training
    for epoch in range(N):
        # train_dataloader.sampler.set_epoch(epoch)
        # train
        if not single_gpu: train_dataloader.sampler.set_epoch(epoch)
        logger.info(f'Current Training Procedure: [{epoch+1}|{N}]')
        
        for _, data in enumerate(train_dataloader):
            current_iter += 1

            """
                input: segmentation
                output: latent image
            """

            cin, cout = data['segmentation'], data['latent-feature']
            cin = torch.tensor(cin.squeeze(), dtype=torch.float32, requires_grad=True)
            cout = torch.tensor(cout.squeeze(), dtype=torch.float32, requires_grad=True)
            
            if np.any(np.isnan(cin.detach().numpy())) or np.any(np.isnan(cout.detach().numpy())):
                continue
            
            assert cin.shape == cout.shape, f'cin.shape = {cin.shape}, cout.shape = {cout.shape}'
            
            optimizer.zero_grad()
            pm_.zero_grad()

            pred = pm_(cin) if single_gpu else pm_.module(cin)
            kl_loss_sum = kl_div(pred, cout, reduction='sum', log_target=True)
            kl_loss_sum.backward()
            optimizer.step()

            if current_iter % opt.print_fq == 0:
                loss_info = 'current_iter: %d \nEPOCH: [%d|%d], L2 Loss in Diffusion Steps: %.6f' % (current_iter, epoch + 1, opt.epochs, l_pixel)
                logger.info(loss_info)
                logger.info(loss_dict)

                # save checkpoint
                rank = opt.local_rank
                logger.info(f'rank = {rank}')
            
            logger.info(f'Current iter done. Iter Num: {current_iter}')
            if (rank == 0) and ((current_iter + 1) % opt.save_fq == 0):
                save_filename = f'model_iter_{current_iter + 1}.pth'
                save_path = os.path.join(experiments_root, 'models', save_filename)
                save_dict = {}
                ad_bare = get_bare_model(LatentSegAdapter)
                state_dict = ad_bare.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    save_dict[key] = param.cpu()

                logger.info(f'saving pth to path: {save_path}')
                torch.save(save_dict, save_path)
                # save state

                state = {'epoch': epoch, 'iter': current_iter + 1, 'optimizers': optimizer.state_dict()}
                save_filename = f'{current_iter + 1}.state'
                save_path = os.path.join(experiments_root, 'training_states', save_filename)
                logger.info(f'saving state to path: {save_path}')
                torch.save(state, save_path)
    
    
    if (rank == 0):
        save_filename = f'model_epo_final.pth'
        save_path = os.path.join(experiments_root, 'models', save_filename)
        save_dict = {}
        ad_bare = get_bare_model(LatentSegAdapter)
        state_dict = ad_bare.state_dict()
        for key, param in state_dict.items():
            if key.startswith('module.'):  # remove unnecessary 'module.'
                key = key[7:]
            save_dict[key] = param.cpu()

        logger.info(f'saving pth to path: {save_path}')
        torch.save(save_dict, save_path)
        # save state

        state = {'epoch': epoch, 'iter': current_iter + 1, 'optimizers': optimizer.state_dict()}
        save_filename = f'{current_iter + 1}.state'
        save_path = os.path.join(experiments_root, 'training_states', save_filename)
        logger.info(f'saving state to path: {save_path}')
        torch.save(state, save_path)

    return



if __name__ == '__main__':
    main()
    print('finished.')
