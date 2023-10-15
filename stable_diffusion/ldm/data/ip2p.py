import torch, os, json
import os.path as osp
from sam.seg2latent import ProjectionTo
from jieba import re
from tqdm import tqdm

def get_current_File(folder_path: str = None, base_path: str = None) -> dict[dict]:

    """
        folder_path is a folder (folder name is a bunch of number) under big folder 'clip-diltered' or 'randomly-sample'
        simultaneously, add base_path to create absolute path

        e.g.:       base_path = '../autodl-tmp/DATSETS/clip-filtered'
                    folder_path = '0040048'

                    return:   {

                    [   {'0':..., '1':...,
                        #image     #cin     #cout
                            'edit-prompt':...,  'in':...,   'out':...
                                 #edit-prompt      #ori-input   #ori-output
                                                },
                                                {...}, {...}   ],      [], [], []
                                                #following term
                    }

    """

    assert folder_path != None, 'no input path'
    name_dict = {}
    for file in os.listdir(folder_path):
        # e.g.: file = '09960897_0.jpg' -> ['09960897', '0', 'jpg']
        if file.endswith('.jpg') or file.endswith('.png'):
            name = re.split(file.strip(), '[_\.]')
            assert len(name) == 3, f'unusual file name'
            if name[0] not in name_dict.keys():
                name_dict[name[0]] = {}
            assert name[1] in ['0', '1'], 'unrecognized image suffix'

            absolute_path = osp.join(base_path, folder_path, file)
            assert osp.isfile(absolute_path)
            name_dict[name[0]][name[1]] = absolute_path
        else:
            continue

    file = 'prompt.json'
    absolute_path = osp.join(base_path, folder_path, file)
    assert osp.isfile(absolute_path)
    content = json.load(osp.join(base_path, folder_path, file))
    for k, _ in name_dict.items():
        name_dict[k]['edit-prompt'] = content['edit']
        name_dict[k]['in'] = content['input']
        name_dict[k]['out'] = content['output']

    return name_dict

class Ip2pDatasets(ProjectionTo):
    def __init__(self, image_folder, sam_model, sd_model, pth_path, device='cuda', single_gpu=True):
        super().__init__(self, sam_model, sd_model, pth_path, device=device)
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
        self.total_data_path_dict = {}


    def make_total_path_intoDICT(self):

        folder_list = os.listdir(self.image_folder)
        assert 'clip-filtered' in folder_list and 'random-sample' in folder_list
        base_paths = [osp.join(self.image_folder, file_folder) for file_folder in folder_list \
                                        if not file_folder.endswith('.ipynb_checkpoints')]
        # ['clip-filtered', 'random-sample']

        for i in range(len(base_paths)):
            base_path = base_path[i]
            if base_path.endswith('.ipynb_checkpoints'): continue
            shard_list = []
            base_path = osp.join(base_paths, base_path)
            base_path_list = os.listdir(base_path)
            base_path_tqdm = tqdm(base_path_list, desc=f'Path Procedure [{i}|{len(base_paths)}]: ', total=len(base_path_list))

            for _, shard in enumerate(base_path_tqdm):
                # shard = '........./shard-xx'
                if 'shard' not in shard or shard.endswith('/ipynb_checkpoints'): continue
                now_path = osp.join(base_path, shard)
                path_list = [get_current_File(image_prompt_folder, now_path) for image_prompt_folder in os.listdir(shard) \
                                                            if not image_prompt_folder.endswith('.ipynb_checkpoints')]
                # corresponds to base_path: shard-00, shard-01, ..., shard-29
                shard_list.append(path_list)

            self.total_data_path_dict[base_path] = shard_list

        return

    def MakeData(self):
        self.make_total_path_intoDICT()
        keys = self.total_data_path_dict.keys()
        assert keys == ['clip-filtered', 'random-sample'], f'total_data_path_dict.keys() = {keys}'
        clip_filtered = self.total_data_path_dict['clip-filtered']
        random_sample = self.total_data_path_dict['random-sample']

        assert isinstance(self.total_data_path_dict['clip-filtered'], list), f'TYPE of clip-filtered: {type(clip_filtered)}'
        assert isinstance(self.total_data_path_dict['random-sample'], list), f'TYPE of clip-filtered: {type(random_sample)}'

        assert len(self.total_data_path_dict['clip-filtered']) == 30, f'len of clip-diltered: {len(clip_filtered)}'
        assert len(self.total_data_path_dict['random-sample']) == 30, f'len of random-sample: {len(random_sample)}'

        return

    def __len__(self):
        length = 0




    def __getitem__(self, item):

