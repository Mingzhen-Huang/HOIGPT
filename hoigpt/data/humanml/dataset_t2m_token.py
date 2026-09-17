import random
import numpy as np
from torch.utils import data
from .dataset_t2m import Text2MotionDataset
import codecs as cs
from os.path import join as pjoin

obj_class = ["box", "capsulemachine", "espressomachine", "ketchup", "laptop", "microwave", "mixer", "notebook", "phone", "scissors", "waffleiron"]

class Text2MotionDatasetToken(Text2MotionDataset):

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        
        # Data mean and std
        self.mean = mean
        self.std = std
        
        # Data path
        self.data_root = data_root
        
        split_file = pjoin(data_root, split + '.txt')
        motion_dir = pjoin(data_root, 'new_joints') #if "hoi" not in data_root else pjoin(data_root, 'new_joints')
        self.dataname = kwargs.get('name', 'arctic')
        # import pdb; pdb.set_trace()
        text_dir = pjoin(data_root, 'texts')

        # Data id list
        self.id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())
                
        new_name_list = []
        length_list = []
        data_dict = {}
        self.object_name_list = []
        self.object_pc_dict = {}

        for name in self.id_list:
            # try:
            motion = np.load(pjoin(motion_dir, name + '.npy'))
            # Match the original training/evaluation loader's temporal sampling.
            if motion.shape[0] > 400:
                motion = motion[::4]
            obj = name.split("_")[1]
            # motion = motion[::4]
            # if (len(motion)) <  self.min_motion_length or (len(motion) >= 200):
            #     continue

            data_dict[name] = {'motion': motion,
                            'length': len(motion),
                            'name': name}
            new_name_list.append(name)
            length_list.append(len(motion))
            # import pdb; pdb.set_trace()
            obj_name = name.split('_')[1]
            if obj_name not in self.object_name_list:
                self.object_name_list.append(obj_name)
        # except:
        #     # Some motion may not exist in KIT dataset
        #     pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list
        if not self.name_list:
            raise ValueError(f"No motions found in {split_file}")
        self.nfeats = data_dict[self.name_list[0]]['motion'].shape[-1]

        for obj in self.object_name_list:
            dataname = "arctic" if obj in obj_class else "grab"
            normalized_obj_pc,_,obj_verts_org = self.get_object_hand_info(obj, self.dataname)[2:5]
            self.object_pc_dict[obj] = [normalized_obj_pc, obj_verts_org]
        # import pdb; pdb.set_trace()
    
    
    def __len__(self):
        return len(self.data_dict)  
        
    def __getitem__(self, item):
        name = self.name_list[item]
        # obj_name = self.name_list[idx].split('_')[1]
        data = self.data_dict[name]
        obj_name = name.split('_')[1]
        # name = self.name_list[idx]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return name, motion, m_length, True, True, True, True, True, True,  name, self.object_pc_dict[obj_name]
