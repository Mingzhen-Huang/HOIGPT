import numpy as np
import torch
import os 
from os.path import join as pjoin
from .humanml.utils.word_vectorizer import WordVectorizer
from . import BASEDataModule
from .humanml import Text2MotionDatasetEval, Text2MotionDataset, Text2MotionDatasetCB, MotionDataset, MotionDatasetVQ, Text2MotionDatasetToken, Text2MotionDatasetM2T
from .utils import humanml3d_collate


class HOIDataModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):

        super().__init__(collate_fn=humanml3d_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        
        # Basic info of the dataset
        cfg.DATASET.JOINT_TYPE = 'grab'
        self.name = "grab"
        self.njoints = 15
        
        # Path to the dataset
        data_root = cfg.DATASET.GRAB.ROOT
        self.hparams.name = "grab"
        self.hparams.data_root = data_root
        self.hparams.downsample = 4
        self.hparams.text_dir = pjoin(data_root, "texts")
        self.hparams.motion_dir = pjoin(data_root, 'new_joints')
        self.hparams.obj = cfg.DATASET.OBJ
        
        # # Mean and std of the dataset
        # dis_data_root = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m','t2m', "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
        self.hparams.mean = np.load(pjoin(data_root, "mean.npy"))
        self.hparams.std = np.load(pjoin(data_root, "std.npy"))
        
        # Mean and std for fair evaluation
        # dis_data_root_eval = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m', 't2m', "Comp_v6_KLD01", "meta")
        self.hparams.mean_eval = np.load(pjoin(data_root, "mean.npy"))
        self.hparams.std_eval = np.load(pjoin(data_root, "std.npy"))
        
        # Length of the dataset
        self.hparams.max_motion_length = cfg.DATASET.GRAB.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.GRAB.MIN_MOTION_LEN
        self.hparams.max_text_len = cfg.DATASET.GRAB.MAX_TEXT_LEN
        self.hparams.unit_length = cfg.DATASET.GRAB.UNIT_LEN

        # Additional parameters
        self.hparams.debug = cfg.DEBUG
        self.hparams.stage = cfg.TRAIN.STAGE
        self.hparams.w_vectorizer = WordVectorizer(
            cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

        # Dataset switch
        self.DatasetEval = Text2MotionDatasetEval
        self.hparams.win_size = 64
        if cfg.TRAIN.STAGE == "vae":
            if cfg.model.params.motion_vae.target.split('.')[-1].lower() == "vqvae":
                
                self.Dataset = MotionDatasetVQ
            else:
                self.Dataset = MotionDataset
        elif 'lm' in cfg.TRAIN.STAGE:
            self.hparams.code_path = cfg.DATASET.CODE_PATH
            self.hparams.code_format = cfg.DATASET.CODE_FORMAT
            self.hparams.task_path = cfg.DATASET.TASK_PATH
            self.hparams.std_text = cfg.DATASET.GRAB.STD_TEXT
            self.Dataset = Text2MotionDatasetCB
        elif cfg.TRAIN.STAGE == "token":
            self.Dataset = Text2MotionDatasetToken
            self.DatasetEval = Text2MotionDatasetToken
        elif cfg.TRAIN.STAGE == "m2t":
            self.Dataset = Text2MotionDatasetM2T
            self.DatasetEval = Text2MotionDatasetM2T
        else:
            self.Dataset = Text2MotionDataset
        # import pdb; pdb.set_trace()
        # Get additional info of the dataset
        self._sample_set = self.get_sample_set(overrides={"split": "val", "tiny": True})
        self.nfeats = self._sample_set.nfeats
        cfg.DATASET.NFEATS = self.nfeats
        # print("total length of arctic is "+str(len(self.Dataset)))
        
    def feats2joints(self, features):
        raise NotImplementedError(
            "HOI uses 208-dimensional MANO hand/object features. "
            "Use the model's MANO layers for joints; HumanML3D recovery is not applicable."
        )

    def joints2feats(self, features):
        raise NotImplementedError(
            "HOI features require MANO pose and object parameters, not HumanML3D joints."
        )

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=True)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list

class HOIDataSYNModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):

        super().__init__(collate_fn=humanml3d_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        
        # Basic info of the dataset
        cfg.DATASET.JOINT_TYPE = 'grab'
        self.name = "grab"
        self.njoints = 15
        
        # Path to the dataset
        data_root = cfg.DATASET.GRABSYN.ROOT
        self.hparams.name = "grab"
        self.hparams.data_root = data_root
        self.hparams.downsample = 4
        self.hparams.text_dir = pjoin(data_root, "texts")
        self.hparams.motion_dir = pjoin(data_root, 'new_joints')
        self.hparams.obj = cfg.DATASET.OBJ
        
        # # Mean and std of the dataset
        # dis_data_root = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m','t2m', "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
        self.hparams.mean = np.load(pjoin(data_root, "mean.npy"))
        self.hparams.std = np.load(pjoin(data_root, "std.npy"))
        
        # Mean and std for fair evaluation
        # dis_data_root_eval = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m', 't2m', "Comp_v6_KLD01", "meta")
        self.hparams.mean_eval = np.load(pjoin(data_root, "mean.npy"))
        self.hparams.std_eval = np.load(pjoin(data_root, "std.npy"))
        
        # Length of the dataset
        self.hparams.max_motion_length = cfg.DATASET.GRAB.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.GRAB.MIN_MOTION_LEN
        self.hparams.max_text_len = cfg.DATASET.GRAB.MAX_TEXT_LEN
        self.hparams.unit_length = cfg.DATASET.GRAB.UNIT_LEN

        # Additional parameters
        self.hparams.debug = cfg.DEBUG
        self.hparams.stage = cfg.TRAIN.STAGE
        self.hparams.w_vectorizer = WordVectorizer(
            cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

        # Dataset switch
        self.DatasetEval = Text2MotionDatasetEval
        self.hparams.win_size = 64
        if cfg.TRAIN.STAGE == "vae":
            if cfg.model.params.motion_vae.target.split('.')[-1].lower() == "vqvae":
                
                self.Dataset = MotionDatasetVQ
            else:
                self.Dataset = MotionDataset
        elif 'lm' in cfg.TRAIN.STAGE:
            self.hparams.code_path = cfg.DATASET.CODE_PATH
            self.hparams.code_format = cfg.DATASET.CODE_FORMAT
            self.hparams.task_path = cfg.DATASET.TASK_PATH
            self.hparams.std_text = cfg.DATASET.GRAB.STD_TEXT
            self.Dataset = Text2MotionDatasetCB
        elif cfg.TRAIN.STAGE == "token":
            self.Dataset = Text2MotionDatasetToken
            self.DatasetEval = Text2MotionDatasetToken
        elif cfg.TRAIN.STAGE == "m2t":
            self.Dataset = Text2MotionDatasetM2T
            self.DatasetEval = Text2MotionDatasetM2T
        else:
            self.Dataset = Text2MotionDataset
        # import pdb; pdb.set_trace()
        # Get additional info of the dataset
        self._sample_set = self.get_sample_set(overrides={"split": "val", "tiny": True})
        self.nfeats = self._sample_set.nfeats
        cfg.DATASET.NFEATS = self.nfeats
        # print("total length of arctic is "+str(len(self.Dataset)))
        
    def feats2joints(self, features):
        raise NotImplementedError(
            "HOI uses 208-dimensional MANO hand/object features. "
            "Use the model's MANO layers for joints; HumanML3D recovery is not applicable."
        )

    def joints2feats(self, features):
        raise NotImplementedError(
            "HOI features require MANO pose and object parameters, not HumanML3D joints."
        )

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=True)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list

class GrabFullModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):

        super().__init__(collate_fn=humanml3d_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        
        # Basic info of the dataset
        cfg.DATASET.JOINT_TYPE = 'grab'
        self.name = "grabfull"
        self.njoints = 15
        
        # Path to the dataset
        data_root = cfg.DATASET.GRABFULL.ROOT
        self.hparams.name = "grab"
        self.hparams.data_root = data_root
        self.hparams.downsample = 4
        self.hparams.text_dir = pjoin(data_root, "texts")
        self.hparams.motion_dir = pjoin(data_root, 'new_joints')
        self.hparams.obj = cfg.DATASET.OBJ
        
        # # Mean and std of the dataset
        # dis_data_root = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m','t2m', "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
        self.hparams.mean = np.load(pjoin(data_root, "mean.npy"))
        self.hparams.std = np.load(pjoin(data_root, "std.npy"))
        
        # Mean and std for fair evaluation
        # dis_data_root_eval = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m', 't2m', "Comp_v6_KLD01", "meta")
        self.hparams.mean_eval = np.load(pjoin(data_root, "mean.npy"))
        self.hparams.std_eval = np.load(pjoin(data_root, "std.npy"))
        
        # Length of the dataset
        self.hparams.max_motion_length = cfg.DATASET.GRAB.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.GRAB.MIN_MOTION_LEN
        self.hparams.max_text_len = cfg.DATASET.GRAB.MAX_TEXT_LEN
        self.hparams.unit_length = cfg.DATASET.GRAB.UNIT_LEN

        # Additional parameters
        self.hparams.debug = cfg.DEBUG
        self.hparams.stage = cfg.TRAIN.STAGE
        self.hparams.w_vectorizer = WordVectorizer(
            cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

        # Dataset switch
        self.DatasetEval = Text2MotionDatasetEval
        self.hparams.win_size = 64
        if cfg.TRAIN.STAGE == "vae":
            if cfg.model.params.motion_vae.target.split('.')[-1].lower() == "vqvae":
                
                self.Dataset = MotionDatasetVQ
            else:
                self.Dataset = MotionDataset
        elif 'lm' in cfg.TRAIN.STAGE:
            self.hparams.code_path = cfg.DATASET.CODE_PATH
            self.hparams.code_format = cfg.DATASET.CODE_FORMAT
            self.hparams.task_path = cfg.DATASET.TASK_PATH
            self.hparams.std_text = cfg.DATASET.GRAB.STD_TEXT
            self.Dataset = Text2MotionDatasetCB
        elif cfg.TRAIN.STAGE == "token":
            self.Dataset = Text2MotionDatasetToken
            self.DatasetEval = Text2MotionDatasetToken
        elif cfg.TRAIN.STAGE == "m2t":
            self.Dataset = Text2MotionDatasetM2T
            self.DatasetEval = Text2MotionDatasetM2T
        else:
            self.Dataset = Text2MotionDataset
        # import pdb; pdb.set_trace()
        # Get additional info of the dataset
        self._sample_set = self.get_sample_set(overrides={"split": "val", "tiny": True})
        self.nfeats = self._sample_set.nfeats
        cfg.DATASET.NFEATS = self.nfeats
        # print("total length of arctic is "+str(len(self.Dataset)))
        
    def feats2joints(self, features):
        raise NotImplementedError(
            "HOI uses 208-dimensional MANO hand/object features. "
            "Use the model's MANO layers for joints; HumanML3D recovery is not applicable."
        )

    def joints2feats(self, features):
        raise NotImplementedError(
            "HOI features require MANO pose and object parameters, not HumanML3D joints."
        )

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=True)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
