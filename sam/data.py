import torch, os, cv2
from cv2 import resize
import numpy as np
from stable_diffusion.ldm.util_ddim import load_img as loads
from basicsr import img2tensor, tensor2img
from einops import repeat, rearrange
from tqdm import tqdm
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


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


# Training dataset: part of COCO, and other random generated by sd.
# Considering some little differences between generated and ground truth contributions

class DataCreator():
    def __init__(self, image_folder: any = None, encoder: any = None, sam: any = None, batch_size: int = 1, downsample_factor=8):
        assert isinstance(image_folder, list) or isinstance(image_folder, str), 'path error when getting DataCreator initialized'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = image_folder if isinstance(image_folder, list) else [image_folder]
        self.latent_encoder = encoder
        self.sam = sam
        self.factor = downsample_factor

        """
            Before DataCreator Declaration:
                sam = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt)
                sam.to(device=device)
                mask_generator = SamAutomaticMaskGenerator(sam)
                masks = mask_generator.generate(np_image)
            
            Input:
                sam = masks
            
        """

        self.batch_size = batch_size
        self.seg_list = []
        self.latent_list = []
        self.data_dict_list = []         # [dict]

    def make_data(self):
        for i in range(len(self.path)):
            folder = self.path[i]
            dir = os.listdir(folder)
            iter_ = tqdm(dir, desc=f'Adding Images in Folder to the list: [{i+1}|{len(self.path)}]', total=len(dir))
            for j, file in enumerate(iter_):
                if file.endswith('.png') or file.endswith('.jpg'):
                    
                    file = os.path.join(folder, file)  # absolute file path
                    image, _ = loads(opt=None, path=file)
                    self.seg_list.append(image)
                    
                    image = np.array(image).astype(np.float32) / 255.0
                    image = image[None].transpose(0, 3, 1, 2)
                    image = torch.from_numpy(image)
                    image = repeat(image, "1 ... -> b ...", b=self.batch_size)
                    # print(type(image))
                    self.latent_list.append(image)
                    
                else:
                    continue
        # assert 0, f'types: {type(self.latent_list[0].shape)}, {self.seg_list[0].shape}'
        return

    def MakeData(self):
        self.make_data()
        assert len(self.seg_list) == len(self.latent_list), 'error occurred when making data -> make'
        self.data_dict_list = [ {'latent-feature':self.latent_list[i], 'segmentation': self.seg_list[i]} for i in range(len(self.seg_list))]

    def __len__(self):
        assert len(self.seg_list) == len(self.latent_list), 'error occurred when making data -> len'
        return len(self.seg_list)

    def __getitem__(self, item):
        i = self.data_dict_list[item]
        u, v = i['latent-feature'], i['segmentation']

        latent = self.latent_encoder(torch.tensor(u.clone().detach().requires_grad_(True), \
                                                   dtype=torch.float32, requires_grad=True).to(self.device)).mode()
        # .clone().detach().requires_grad_(True)
        seg = np.array(v.astype(np.uint8))
        seg = self.sam(seg)
        seg = torch.from_numpy(get_masked_Image(seg, use_alpha=False)).to(self.device)
        # assert 0, f'seg.shape = {seg.shape}'
        # torch.Size([512, 512, 4])
        seg = rearrange(seg, "h w c -> 1 c h w").to(self.device)
        seg_latent = self.latent_encoder(torch.tensor(seg.clone().detach().requires_grad_(True), \
                                                       dtype=torch.float32, requires_grad=True)).mode()
        seg_latent = repeat(seg_latent, "1 ... -> b ...", b=self.batch_size)
        assert seg_latent.shape == latent.shape, f'seg_latent.shape={seg_latent.shape}, latent.shape={latent.shape}'
        
        return {'latent-feature':latent, 'segmentation': seg_latent}