import torch, os
import numpy as np
from stable_diffusion.ldm.util_ddim import load_img as loads
from sam.util import get_masked_Image
from basicsr import img2tensor, tensor2img
from einops import repeat, rearrange
from tqdm import tqdm
from random import randint
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor




# Training dataset: part of COCO, and other random generated by sd, or any other image dataset
# Considering some little differences between generated and ground truth contributions

class DataCreator():
    def __init__(self, image_folder: any = None, sd_model: any = None, sam: any = None, \
                 batch_size: int = 1, downsample_factor=8, data_scale=0.1, logger = None):
        assert isinstance(image_folder, list) or isinstance(image_folder, str), 'path error when getting DataCreator initialized'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = image_folder if isinstance(image_folder, list) else [image_folder]
        self.model = sd_model
        self.sam = sam
        self.factor = downsample_factor
        self.data_scale = data_scale
        self.logger = logger

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

    def printer(self, msg: str = None):
        if msg != None and self.logger != None:
            self.logger(msg)
        return

    @torch.no_grad()
    def make_data(self):
        for i in range(len(self.path)):
            folder = self.path[i]
            dir = os.listdir(folder)
            iter_ = tqdm(dir, desc=f'Adding Images in Folder to the list: [{i+1}|{len(self.path)}]', total=len(dir))
            for j, file in enumerate(iter_):

                if (j+1)%100 == 0:
                    self.printer(f'Loading Data Process: [{j+1}|{len(dir)}]')

                if file.endswith('.png') or file.endswith('.jpg'):
                    xx = randint(0,5000)
                    if xx > self.data_scale * 5000:
                        continue
                    file = os.path.join(folder, file)  # absolute file path
                    image, _ = loads(opt=None, path=file)
                    seg = self.sam(np.array(image.astype(np.uint8)))
                    seg = torch.from_numpy(get_masked_Image(seg, use_alpha=False)).to(self.device)
                    seg = rearrange(seg, "h w c -> 1 c h w").clone().detach().requires_grad_(False).to(torch.float32)
                    seg_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(seg))
                    seg_latent = repeat(seg_latent, "1 ... -> b ...", b=self.batch_size)
                    self.seg_list.append(seg_latent)
                    
                    image = np.array(image).astype(np.float32) / 255.0
                    image = image[None].transpose(0, 3, 1, 2)
                    image = 2. * torch.from_numpy(image) - 1.
                    # image = torch.from_numpy(image)           # ===> Wrongly Calculated !
                    image = repeat(image, "1 ... -> b ...", b=self.batch_size).to(self.device).clone().detach().requires_grad_(False).to(torch.float32)
                    latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(image))
                    self.latent_list.append(latent)
                    
                else:
                    continue
        return

    @torch.no_grad()
    def MakeData(self):
        self.make_data()
        assert len(self.seg_list) == len(self.latent_list), 'error occurred when making data -> make'
        self.data_dict_list = [ {'latent-feature': self.latent_list[i], 'segmentation': self.seg_list[i]} \
                                                for i in range(len(self.seg_list))]

    def __len__(self):
        assert len(self.seg_list) == len(self.latent_list), 'error occurred when making data -> len'
        return len(self.seg_list)

    def __getitem__(self, item):
        i = self.data_dict_list[item]
        u, v = i['latent-feature'], i['segmentation']
        assert u.shape == v.shape, f'seg_latent.shape={v.shape}, latent.shape={u.shape}'
        
        return {'latent-feature': u, 'segmentation': v}