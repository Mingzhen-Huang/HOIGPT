import random
import codecs as cs
import numpy as np
import copy

from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from .dataset_m import MotionDataset
from .dataset_t2m import Text2MotionDataset

# obj_class = ["box", "capsulemachine", "espressomachine", "ketchup", "laptop", "microwave", "mixer", "notebook", "phone", "scissors", "waffleiron"]
obj_class = []

class MotionDatasetVQ(Text2MotionDataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length,
        min_motion_length,
        win_size,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        # Filter out the motions that are too short
        self.window_size = win_size
        
        name_list = list(self.name_list)
        for name in self.name_list:
            motion = self.data_dict[name]["motion"]
            if motion.shape[0] < self.window_size/4:
                name_list.remove(name)
                self.data_dict.pop(name)
            elif motion.shape[0] < self.window_size: 
                motion = np.concatenate([motion, np.flip(motion, 0)], 0)
                if motion.shape[0] < self.window_size:
                    motion = np.concatenate([motion, motion], 0)
                self.data_dict[name]["length"] = len(motion)
                self.data_dict[name]["motion"] = motion
        self.name_list = name_list
        

    def __len__(self):
        return len(self.name_list) - self.pointer
    
    def rebase_motion(self, motion):
        base_trans = copy.deepcopy(motion[:1, -3:])
        motion[:, -3:] -= base_trans
        motion[:, 96:99] -= base_trans
        motion[:, 195:198] -= base_trans
        return motion
        

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        obj_name = self.name_list[idx].split('_')[1]
        name = self.name_list[idx]
        motion, length = data["motion"], data["length"]

        # if motion
        # motion = motion[::10]

        idx = random.randint(0, motion.shape[0] - self.window_size)
        motion = motion[idx:idx + self.window_size]
        # motion = self.rebase_motion(motion)


        motion = (motion - self.mean) / self.std

        subset = self.dataname

        return None, motion, length, None, None, None, None, None, None, name, self.object_pc_dict[obj_name], subset
