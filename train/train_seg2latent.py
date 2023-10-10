import torch, cv2
import numpy as np
import os, sys
import os.path as osp
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           scandir, tensor2img)
import logging
from omegaconf import OmegaConf

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


def load_resume_state(opt):
    resume_state_path = None
    if opt.auto_resume:
        state_path = osp.join('experiments', opt.name, 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt.resume_state_path = resume_state_path

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
    return resume_state



def main():


    config = './config/ip2p-ddim.yaml'
    local_rank = 0




    print('loading configs...')
    config = OmegaConf.load(config)

    torch.cuda.set_device(local_rank)
    print('init starts')
    torch.distributions.init_process_group(backend='NCCL')
    print('init ends')

    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(local_rank)







    return



if __name__ == '__main__':
    main()

