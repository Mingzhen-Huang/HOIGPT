from omegaconf import OmegaConf
from os.path import join as pjoin
from hoigpt.config import instantiate_from_config


def build_data(cfg, phase="train"):
    data_config = OmegaConf.to_container(cfg.DATASET, resolve=True)
    data_config['params'] = {'cfg': cfg, 'phase': phase}
    # import pdb; pdb.set_trace()
    if isinstance(data_config['target'], str):
        return instantiate_from_config(data_config)
    elif isinstance(data_config['target'], list):
        raise ValueError(
            "DATASET.target must select one ARCTIC or GRAB data module. "
            "The original multi-dataset Concat module is not available."
        )
