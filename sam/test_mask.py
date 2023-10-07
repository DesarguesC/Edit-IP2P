# from IPython.display import display, HTML
# display(HTML(
# """
# <a target="_blank" href="https://colab.research.google.com/github/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>
# """
# ))

import time

start_time = time.time()

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image

name = 'fig'

def show_anns(anns):
    
    """
        segmentation -> 单个图像掩码
        area         -> 区域面积
    """
    
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    f = 0
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    use = img
    for ann in sorted_anns:
        # if not (f >= 3 and f <=7) or f == 6:
        if not f == 3:
            f += 1
            continue
        m = ann['segmentation']
        # print('seg shape: ', ann['segmentation'].shape)
        # color_mask = np.concatenate([np.random.random(3), [0.35]])
        color_mask = np.concatenate([[0,0,0], [0.35]])   # black and white
        
        
        if f == 0:
            pass
            # print('\ncolor mask: ', color_mask)
            # print(color_mask.shape)
        img[m] = color_mask
        tmp = use
        tmp[m] = color_mask
        if f == 0:
            pass
            # print('\nafter: ', img)
            # print(img.shape)
        f += 1
        # tmp = Image.fromarray(((1.-tmp) * 255).astype(np.uint8))
        tmp = Image.fromarray((tmp * 255).astype(np.uint8))
        tmp = tmp.convert('RGB')
        tmp.save('./images/' + name + '_folder/mask/' + str(f) + '.jpg')
            
        
    
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = img.convert('RGB')
    img.save('./images/' + name + '_folder/fig_mask.jpg')
    print(f'cnt: {f}')

image = cv2.imread('images/' + name + '.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "../autodl-tmp/SAM/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)
sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

# np.savetxt('./images/out.txt', np.c_[sorted_masks[0]['segmentation']], fmt='%d',delimiter='')

# print(len(sorted_masks))



use_time = time.time()-start_time

print('\ntime spent: %.2f(s)' % use_time)

show_anns(masks)


print(sorted_masks[0].keys())
print(sorted_masks[0]['area'])
print(sorted_masks[0]['bbox'], sorted_masks[1]['bbox'], sorted_masks[3]['bbox'])
print(sorted_masks[0]['predicted_iou'])
print(sorted_masks[0]['point_coords'])
print(sorted_masks[0]['stability_score'])
print(sorted_masks[0]['crop_box'], sorted_masks[1]['crop_box'], sorted_masks[3]['crop_box'])

