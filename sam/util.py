import numpy as np
import torch, cv2, os
import matplotlib.pyplot as plt
from PIL import Image





def show_anns(anns, name, base_path=None, reverse=False, **kwargs):
    # return the valid mask according to the seq choice
    seq_choice = kwargs['seq_choice'] if 'seq_choice' in kwargs.keys() else [1] * len(anns)
    
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    f = 0
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    use = img
    base = ('./Outputs/' + name) if base_path is None else base_path
    if not os.path.exists(base + '_mask_folder'):
        os.mkdir(base + '_mask_folder')
    if not os.path.exists(base + '_mask_folder/masks/'):
        os.mkdir(base + '_mask_folder/masks/')
        
    for i in range(len(sorted_anns)):
        if seq_choice[i] != 1:
            pass
        
        m = sorted_anns[i]['segmentation']
        # print('seg shape: ', ann['segmentation'].shape)
        color_mask = np.concatenate([[0,0,0] if reverse else np.random.random(3), [0.35]])
        
        img[m] = color_mask
        tmp = use
        tmp[m] = color_mask

        f += 1
        tmp = Image.fromarray( ( ((1.-tmp) if reverse  else tmp) * 255 ).astype(np.uint8) )
        tmp = tmp.convert('RGB')
        tmp.save(base + '_mask_folder/masks/' + str(f) + '.jpg')
        
    mask_vector = img
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = img.convert('RGB')
    img.save(base + '_mask_folder/' + name + '_masked.jpg')
    return mask_vector

@torch.no_grad()
def get_masked_Image(seg: list = None, no_color: bool = False, use_alpha=True):
    assert seg != None
    # seq_choice = kwargs['seq_choice'] if 'seq_choice' in kwargs.keys() else [1] * len(seg)
    if len(seg) == 0:
        return
    sorted_seg = sorted(seg, key=(lambda x: x['area']), reverse=True)
    img = np.ones((sorted_seg[0]['segmentation'].shape[0], sorted_seg[0]['segmentation'].shape[1], 4 if use_alpha else 3))
    img[:,:,3 if use_alpha else 2] = 0
    length = len(seg)
    cut = (int)(length // 3) if length%3==0 else (int)((length + 3 - length%3) // 3)
    c = lambda x: [x / cut if x < cut else (1. - 1.0e-03), (0. + 1.0e-03) if x < cut else \
        ((1. - 1.0e-03) if x >= 2 * cut else (x - cut) / cut), (x - 2 * cut) / cut if x >= 2 * cut else (0. + 1.0e-03)]

    for i in range(len(sorted_seg)):
        m = sorted_seg[i]['segmentation']
        # print('seg shape: ', ann['segmentation'].shape)
        if use_alpha:
            color_mask = np.concatenate([[0, 0, 0] if no_color else c(i*1.), [0.9]])
        else:
            color_mask = [0, 0, 0] if no_color else c(i*1.)

        img[m] = color_mask

    # mask_vector = img
    # img = Image.fromarray((img * 255).astype(np.uint8))
    # img = img.convert('RGB')

    return img


def get_masked_Image_(seg: list = None, no_color: bool = False, use_alpha=True):
    if isinstance(seg[0], dict):
        return get_masked_Image(seg=seg, no_color=no_color, use_alpha=use_alpha)
    elif isinstance(seg[0], list):  # use this
        return [get_masked_Image(seg=seg_, no_color=no_color, use_alpha=use_alpha) for seg_ in seg]