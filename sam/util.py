import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
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