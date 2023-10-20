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
        "--config",
        type=str,
        default="./configs/stable-diffusion/sd-v1-train.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        '--image_folder',
        type=str,
        default='../autodl-tmp/DATASETS/',
        help='total ip2p dataset folder'
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train_seg_control",
        help="experiment name",
    )
    parser.add_argument(
        "--print_fq",
        type=int,
        default=10,
        help="frequency to print logger infos",
    )
    parser.add_argument(
        "--save_fq",
        type=int,
        default=100,
        help="frequency to save the training datas",
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
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--gpus",
        default=[0, 1, 2, 3],
        help="gpu idx",
    )
    parser.add_argument(
        '--local_rank',
        default=0,
        type=int,
        help='node rank for distributed training'
    )
    parser.add_argument(
        '--data_pro',
        default=8,
        type=int,
        help='read in data pro'
    )
    parser.add_argument(
        '--launcher',
        default='pytorch',
        type=str,
        help='node rank for distributed training'
    )
    parser.add_argument(
        '--use_single_gpu',
        type=str2bool,
        default=False,
        help='node rank for distributed training'
    )
    parser.add_argument(
        '--adapter_time_emb',
        type=str2bool,
        default=False,
        help='whether to add embedded time feature into adapter'
    )
    parser.add_argument(
        '--init',
        type=str2bool,
        default=False,
        help='whether to init latentadapter with pretrained models'
    )
    parser.add_argument(
        '--ls_path',
        type=str,
        default='./checkpoints/ls_model.pth',
        help='LatentSegmentAdapter model path'
    )
    opt = parser.parse_args()
    return opt

def main():
    opt = parsr_args()
    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    experiments_root = './exp-segControlNet/'
    experiments_root = mkdir(experiments_root, opt.local_rank)
    log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    
    
    
    if not opt.use_single_gpu:
        # lauch multi-GPU training process
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')
        rank = opt.local_rank
        num_gpus = torch.cuda.device_count()
        dist.init_process_group(backend='nccl', rank=rank, init_method='env://')
        torch.cuda.set_device(rank % num_gpus)
        dist.barrier()
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda", rank)
        
        print(f'current rank: {rank}')
        
        
    sd_model, sam_model, pm_model, configs = load_inference_train(opt, opt.device)
    print(f'sd-{opt.local_rank}')
    if not opt.use_single_gpu:
        sd_model = torch.nn.parallel.DistributedDataParallel(
            sd_model,
            device_ids=[opt.local_rank], output_device=opt.local_rank)
    torch.backends.cudnn.benchmark = True
    
    
    LatentSegAdapter = Adapter(cin=8*16, channels=[256, 512, 512, 1024], nums_rb=2, ksize=1, sk=True, \
                               use_conv=False, use_time=opt.adapter_time_emb).train().to(opt.device)
    if opt.init:
        LatentSegAdapter.load_state_dict(torch.load(opt.ls_path))
        LatentSegAdapter.train()
    
    
    if not opt.use_single_gpu:
        print(f'adapter-{opt.local_rank}')
        LatentSegAdapter = torch.nn.parallel.DistributedDataParallel(
            LatentSegAdapter,
            device_ids=[opt.local_rank], output_device=opt.local_rank)
        torch.backends.cudnn.benchmark = True
        print(f'sam-{opt.local_rank}')
        sam_model = torch.nn.parallel.DistributedDataParallel(
            sam_model,
            device_ids=[opt.local_rank], output_device=opt.local_rank)
        torch.backends.cudnn.benchmark = True
        print(f'pm-{opt.local_rank}')
        pm_model = torch.nn.parallel.DistributedDataParallel(
            pm_model,
            device_ids=[opt.local_rank], output_device=opt.local_rank)
        torch.backends.cudnn.benchmark = True
    
    mask_generator = SamAutomaticMaskGenerator(sam_model if opt.use_single_gpu else sam_model.module).generate
    data_params = {
        'image_folder': opt.image_folder,
        'sd_model': sd_model,
        'sam_model': mask_generator,
        'pm_model': pm_model,
        'device': opt.device,
        'single_gpu': opt.use_single_gpu,
        'data_pro': opt.data_pro / 10.
    }
    data_creator = Ip2pDatasets(**data_params)
    with torch.no_grad():
        data_creator.MakeData(opt.max_resolution)
    print(f'Randomly loading data with length: {len(data_creator)}')
    
    
    Models = {
        'projection': pm_model if opt.use_single_gpu else pm_model.module,
        'adapter': LatentSegAdapter if opt.use_single_gpu else LatentSegAdapter.module,
        'time_emb': opt.adapter_time_emb
    }
    
    torch.cuda.empty_cache()

    # optimizer
    params = list(LatentSegAdapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=configs['training']['lr'])

    
    current_iter = 0
    logger.info(f'\n\n\nStart training from epoch: 0, iter: {current_iter}\n')
    
    import time
    start_time = time.time()
    
    train_sampler = None if opt.use_single_gpu else torch.utils.data.distributed.DistributedSampler(data_creator)
    train_dataloader = tqdm(DataLoader(dataset=data_creator, batch_size=opt.bsize, shuffle=True, drop_last=True), \
                            desc='Procedure', total=len(data_creator)) \
        if opt.use_single_gpu else torch.utils.data.DataLoader(
        data_creator,
        batch_size=opt.bsize,
        shuffle=(train_sampler is None),
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)
    print('sampler init time cost: %.6f'%(time.time() - start_time))
    
    sd_bare, sam_bare, pm_bare = get_bare_model(sd_model), get_bare_model(sam_model), get_bare_model(pm_model)
    
    
    
    # training
    for epoch in range(opt.epochs):
        if not opt.use_single_gpu: train_sampler.set_epoch(epoch)
        
        logger.info(f'Current Training Procedure: [{epoch + 1}|{opt.epochs}]')

        for _, data in enumerate(train_dataloader):
            current_iter += 1
            """
                data = {
                    'cin': cin_img, 
                    'cout': cout_img, 
                    'edit': edit_prompt, 
                    'seg_cond': seg_cond
                }
            """
            
            cin_pic, cout_pic, edit_prompt = data['cin'], data['cout'], data['edit']
            # seg_cond_latent, seg_cond, c = data['seg_cond_latent'], data['seg_cond'], data['edit']
            # low time cost tested

            with torch.no_grad():
                # cin_pic.shape = [8, 512, 512, 3]
                # print(f'cin_pic.shape = {cin_pic.shape}')
                seg_cond = img2seg(cin_pic, mask_generator, opt.device)
                c = sd_bare.get_learned_conditioning(edit_prompt)
                z_0 = img2latent(cin_pic, sd_bare, opt.device)
                z_T = img2latent(cout_pic, sd_bare, opt.device)
                del cin_pic 
                del cout_pic
                
                seg_cond_latent = seg2latent(seg_cond, sd_bare, opt.device)
                seg_cond_latent = sl2latent(seg_cond_latent, pm_bare, opt.device)
                seg_cond.to(opt.device)

            assert z_0.shape == z_T.shape, f'z0.shape = {z_0.shape}, zT.shape = {z_T.shape}'

            optimizer.zero_grad()
            LatentSegAdapter.zero_grad()

            """
                Models = {
                    'projection': pm_model if opt.use_single_gpu else pm_model.module,
                    'adapter': LatentSegAdapter if opt.use_single_gpu else LatentSegAdapter.module,
                    'time_emb': opt.adapter_time_emb
                }
            """
            # *args: (c, z_0, seg_cond)
            # **kwargs: Models

            l_pixel, loss_dict = sd_model(z_T, c=[c, z_0, seg_cond, seg_cond_latent], **Models)
            l_pixel.backward()
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
