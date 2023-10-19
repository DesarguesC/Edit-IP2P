import torch, os, json, cv2
from random import randint
import os.path as osp
from sam.seg2latent import ProjectionTo
from jieba import re
from tqdm import tqdm

def get_current_File(folder_path: str = None, base_path: str = None) -> list:

    """
        folder_path is a folder (folder name is a bunch of number) under big folder 'clip-diltered' or 'randomly-sample'
        simultaneously, add base_path to create absolute path

        e.g.:       base_path = '../autodl-tmp/DATSETS/clip-filtered'
                    folder_path = '0040048'         ->      in one 'shard' folder       ->    CREATE a list[dict] for the folder

                    return:   {

                    [   {'0':..., '1':...,
                        #image     #cin     #cout
                            'edit-prompt':...,  'in':...,   'out':...
                                 #edit-prompt      #ori-input   #ori-output
                                                },
                                                {...}, {...}   ],      [], [], []
                                                #following term
                    }

        Just put all dict {'0':..., '1':..., 'edit-prompt':..., ...} in single list,
        we only distinguish 'clip-filtered' and 'random-sample'
        
    """

    assert folder_path != None, 'no input path'
    name_dict = {}
    absolute_folder_path = osp.join(base_path, folder_path)
    
    for file in os.listdir(absolute_folder_path):
        # e.g.: file = '09960897_0.jpg' -> ['09960897', '0', 'jpg']
        if file.endswith('.jpg') or file.endswith('.png'):
            # print(f'file: {file}')
            name = re.split('[_\.]', file.strip())
            # print(f'name: {name}')
            assert len(name) == 3, f'unusual file name; name = {name}'
            if name[0] not in name_dict.keys():
                name_dict[name[0]] = {}
            assert name[1] in ['0', '1'], 'unrecognized image suffix'

            absolute_file_path = osp.join(absolute_folder_path, file)
            assert osp.isfile(absolute_file_path)
            name_dict[name[0]][name[1]] = absolute_file_path
        else:
            continue
            
    file = 'prompt.json'
    absolute_path = osp.join(base_path, folder_path, file)
    assert osp.isfile(absolute_file_path)
    
    with open(absolute_path, 'r', encoding='ISO-8859-1') as f:
        content = json.load(f)
        
    for k, _ in name_dict.items():
        name_dict[k]['edit-prompt'] = content['edit']
        name_dict[k]['in'] = content['input']
        name_dict[k]['out'] = content['output']
    current_folder_FileList = [name_dict[n] for n in name_dict.keys()]
    
    return current_folder_FileList

class Ip2pDatasets(ProjectionTo):
    def __init__(self, image_folder, sd_model, sam_model, pm_model, device='cuda', single_gpu=True, data_pro=0.5):
        super().__init__(sam_model=sam_model, sd_model=sd_model, pm_model=pm_model, device=device)
        self.image_folder = image_folder
        """
        e.g.:    image_folder = '..autodl-tmp/DATASETS/'
        
        In The Folder:
        
             '../autodl-tmp/DATSETS/clip-filtered/'
                                            |
                                            |-- shard-00
                                            |       |--34523465 - (.jpg, .json, .jsonl)
                                            |       |--23545433 - (.jps, .json, .jsonl)
                                            |
                                            |-- shard-01
                                            |-- ...
                                            
            '../autodl-tmp/DATSETS/random-sample/'
                                            |
                                            |-- shard-00
                                            |-- shard-01
                                            |-- ...
        """
        self.single_gpu = single_gpu
        self.max_resolution = 512*512
        self.total_data_path_list = []
        self.length = 0
        self.pro = data_pro
        self.max_resolution=512*512

    @torch.no_grad()
    def make_total_path_intoLIST(self):

        folder_list = os.listdir(self.image_folder)
        assert 'clip-filtered' in folder_list and 'random-sample' in folder_list
        base_paths = [osp.join(self.image_folder, file_folder) for file_folder in folder_list \
                                        if not file_folder.endswith('.ipynb_checkpoints')]
        # ['clip-filtered', 'random-sample']

        for i in range(len(base_paths)):
            base_path = base_paths[i]
            if base_path.endswith('.ipynb_checkpoints') or osp.isfile(base_path): continue
            shard_list = []
            base_path_list = os.listdir(base_path)
            # print(f'base_path = {base_path}')
            
            base_path_tqdm = tqdm(base_path_list, desc=f'Path Procedure [{i}|{len(base_paths)}]: ', total=len(base_path_list))

            for _, shard in enumerate(base_path_tqdm):
                # shard = 'shard-xx'
                if 'shard' not in shard or '.' in shard: continue
                now_path = osp.join(base_path, shard)
                # shard = '........./shard-xx'
                # print(f'now_path = {now_path}')
                for image_prompt_folder in os.listdir(now_path):
                    x = randint(0, 1000)
                    if x > 1000 * self.pro:
                        continue
                    if not image_prompt_folder.endswith('.ipynb_checkpoints'):
                        shard_list.extend(get_current_File(image_prompt_folder, now_path))
                        # corresponds to base_path: shard-00, shard-01, ..., shard-29
                        
                        # except Exception as err:
                        #     print(err)

            self.total_data_path_list.extend(shard_list)

        return

    def MakeData(self, max_resolution=512*512):
        self.make_total_path_intoLIST()
        self.max_resolution=max_resolution
        assert self.total_data_path_list != None, 'No Data Add'
        for u in self.total_data_path_list:
            assert isinstance(u, dict)
        return

    def __len__(self):
        return len(self.total_data_path_list)

    @torch.no_grad()
    def __getitem__(self, item):
        item = self.total_data_path_list[item]
        cin_img_path, cout_img_path, edit_prompt = item['0'], item['1'], item['edit-prompt']

        assert osp.isfile(cin_img_path) or not osp.exists(cin_img_path), f'\'0\' -> not a file or file not exists'
        assert osp.isfile(cout_img_path) or not osp.exists(cout_img_path), f'\'1\' -> not a file or file not exists'

        cin_img, cout_img = self.load_img(cin_img_path, Train=True, max_resolution=self.max_resolution), \
                                            self.load_img(cout_img_path, Train=True, max_resolution=self.max_resolution)
        # (np.ndarray, np.ndarray)

        # seg_cond = self.MapsTo(IMG=cin_img, Type='R^3=seg')
        # edit = self.sd_model.get_learned_conditioning(edit_prompt)
        # seg_cond_latent = self.MapsTo(IMG=seg_cond, Type='seg=seg-latent').squeeze()
        # map R3 seg condition into latent space
        
        # cin_img = self.MapsTo(IMG=cin_img, Type='R^3=latent').squeeze()
        # cout_img = self.MapsTo(IMG=cout_img, Type='R^3=latent').squeeze()

        # return {'cin': cin_img, 'cout': cout_img, 'edit': edit_prompt, 'seg_cond': seg_cond, 'seg_cond_latent': seg_cond_latent}
        return {'cin': cin_img, 'cout': cout_img, 'edit': edit_prompt}