import os
import sys
import numpy as np
import pytorch_lightning as pl
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tqdm import tqdm
from hoigpt.config import parse_args
from hoigpt.data.build_data import build_data
from hoigpt.models.build_model import build_model
from hoigpt.utils.load_checkpoint import load_pretrained_vae

def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.METRIC.TYPE = []

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    datasets = build_data(cfg, phase='token')
    print("datasets module initialized")
    # import pdb; pdb.set_trace()
    output_dir = os.path.join(datasets.hparams.data_root, cfg.DATASET.CODE_PATH)

    os.makedirs(output_dir, exist_ok=True)

    # create model
    model = build_model(cfg, datasets)
    if hasattr(model, "motion_vae"):
        model.vae = model.motion_vae
    print("model loaded")

    # Strict load vae model
    if not cfg.TRAIN.PRETRAINED_VAE:
        raise ValueError("Set TRAIN.PRETRAINED_VAE to the trained Stage 1 checkpoint.")
    load_pretrained_vae(cfg, model)

    device = torch.device(f"cuda:{cfg.DEVICE[0]}" if cfg.ACCELERATOR == "gpu" else "cpu")
    model = model.to(device).eval()

    for batch in tqdm(datasets.train_dataloader(),
                      desc=f'motion tokenize'):
        name = batch['name'] if 'name' in batch else batch['text']
        # import pdb; pdb.set_trace()
        
        pose = batch['motion']
        pose = pose.to(device).float()
        
        pc_cond = batch['pc'].to(device).float() if 'pc' in batch else None
        # import pdb; pdb.set_trace()

        if pose.shape[1] == 0:
            continue
        with torch.inference_mode():
            target, _ = model.vae.encode(pose, pc_cond)
        sample_name = name[0]

        if isinstance(target, dict):
            target_np = {key: value.detach().cpu().numpy() for key, value in target.items()}
            target_path = os.path.join(output_dir, sample_name + '.npz')
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                target_path,
                left=target_np['left'],
                right=target_np['right'],
                obj=target_np['obj'],
                length=np.array(target_np['left'].shape[-1], dtype=np.int64),
                format=np.array('hoi_triplet'),
            )
        else:
            target = target.to('cpu').numpy()
            target_path = os.path.join(output_dir, sample_name + '.npy')
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(target_path, target)
        

    print(
        f'Motion tokenization done, the motion tokens are saved to {output_dir}'
    )


if __name__ == "__main__":
    main()
