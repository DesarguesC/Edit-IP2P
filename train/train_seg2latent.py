import torch, cv2
import numpy as np
import os, sys
import os.path as osp
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           scandir, tensor2img)
from basicsr.utils.options import copy_opt_file, dict2str
import logging
from dist_util import init_dist, master_only, get_bare_model, get_dist_info
from omegaconf import OmegaConf
from sam.seg2latent import ProjectionModel as PM
from sam.data import DataCreator
from stable_diffusion.ldm.util_ddim import load_model_from_config
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor




def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(osp.join(path, 'models'))
    os.makedirs(osp.join(path, 'training_states'))
    os.makedirs(osp.join(path, 'visualization'))


def load_resume_state(name='seg-sam', auto_resume=None):
    resume_state_path = None
    if auto_resume:
        state_path = osp.join('experiments', name, 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                # opt.resume_state_path = resume_state_path

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
    return resume_state


def load_target_model(config, device):
    print('loading configs...')
    config = OmegaConf.load(config)

    sd_ckpt = './checkpoint/v1-5-pruned-emaonly.ckpt'
    vae_ckpt = None

    sam_ckpt = '../SAM/sam_vit_h_4b8939.pth'
    sam_type = 'vit_h'

    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)


    model = load_model_from_config(config, sd_ckpt, vae_ckpt)
    encoder = model.model.encode_first_stage().to(device)

    return encoder, mask_generator

def main():


    local_rank = 0
    config = './config/ip2p-ddim.yaml'
    name = 'seg-sam-train'





    torch.cuda.set_device(local_rank)
    print('init starts')
    torch.distributions.init_process_group(backend='NCCL')
    print('init ends')

    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(local_rank)

    encoder_model, sam_model = load_target_model(config, device)
    encoder_model = torch.nn.parallel.DistributedDataParallel(
        encoder_model,
        device_ids=[local_rank],
        output_device=local_rank)
    sam_model = torch.nn.parallel.DistributedDataParallel(
        sam_model,
        device_ids=[local_rank],
        output_device=local_rank)


    data_creator_param = {
        'image_folder': '../COCO/train2017/train2017',
        'encoder': encoder_model.module,
        'sam': sam_model.module,
        'batch_size': 1,
        'downsample_factor': 8
    }

    learning_rate = 1.0e-04
    bsize = 8
    num_workers = 25
    save_path = '../Train/'

    data_creator = DataCreator(**data_creator_param)
    with torch.no_grad():
        data_creator.MakeData()
    print(f'loading data with length: {len(data_creator)}')

    train_sampler = torch.utils.data.distributed.DistributedSampler(data_creator)
    train_dataloader = torch.utils.data.DataLoader(
        data_creator,
        batch_size=bsize,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    pm_ = PM().to(device)

    pm_ = torch.nn.parallel.DistributedDataParallel(
        pm_,
        device_ids=[local_rank],
        output_device=local_rank)




    # optimizer
    params = list(pm_.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate)

    experiments_root = osp.join('experiments', 'seg-sam-training')
    # resume state
    resume_state = load_resume_state()
    if resume_state is None:
        mkdir_and_rename(experiments_root)
        start_epoch = 0
        current_iter = 0
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
    else:
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
        resume_optimizers = resume_state['optimizers']
        optimizer.load_state_dict(resume_optimizers)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']

    # copy the yml file to the experiment root
    copy_opt_file(config, experiments_root)

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    print_fq = 100
    N = 10000

    for epoch in range(N):
        train_dataloader.sampler.set_epoch(epoch)
        # train
        for _, data in enumerate(train_dataloader):
            current_iter += 1

            """
                input: segmentation
                output: latent image
            """

            cin, cout = data['segmentation'], data['latent-image']
            optimizer.zero_grad()
            pm_.zero_grad()

            pred = pm_.module(cin)
            kl_loss_sum = torch.nn.function.kl_div(pred, cout, reduction='sum', log_target=True)
            kl_loss_sum.backward()
            optimizer.step()

            if current_iter%print_fq == 0:
                loss_info = '[%d|%d], KL Divergence Loss: %.6f' % (current_iter+1, N, kl_loss_sum)
                logger.info(loss_info)

                # save checkpoint
                rank, _ = get_dist_info()
                if (rank == 0) and ((current_iter + 1) % config['training']['save_freq'] == 0):
                    save_filename = f'model_ad_{current_iter + 1}.pth'
                    save_path = os.path.join(experiments_root, 'models', save_filename)
                    save_dict = {}
                    pm_bare = get_bare_model(pm_)
                    state_dict = pm_bare.state_dict()
                    for key, param in state_dict.items():
                        if key.startswith('module.'):  # remove unnecessary 'module.'
                            key = key[7:]
                        save_dict[key] = param.cpu()
                    torch.save(save_dict, save_path)
                    # save state
                    state = {'epoch': epoch, 'iter': current_iter + 1, 'optimizers': optimizer.state_dict()}
                    save_filename = f'{current_iter + 1}.state'
                    save_path = os.path.join(experiments_root, 'training_states', save_filename)
                    torch.save(state, save_path)



    return



if __name__ == '__main__':
    main()

