"""Checkpoint loading for the original Lightning training pipeline."""

from collections import OrderedDict

import torch


def _load_training_state(path):
    # Historical Lightning checkpoints contain OmegaConf metadata. These must
    # be trusted local training artifacts; never use this on untrusted files.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a checkpoint/state dictionary.")
    return checkpoint.get("state_dict", checkpoint.get("net", checkpoint))


def load_pretrained(cfg, model, logger=None, phase="train"):
    path = cfg.TRAIN.PRETRAINED if phase == "train" else cfg.TEST.CHECKPOINTS
    if logger is not None:
        logger.info(f"Loading model from {path}")
    model.load_state_dict(_load_training_state(path), strict=True)
    return model


def load_pretrained_vae(cfg, model, logger=None):
    path = cfg.TRAIN.PRETRAINED_VAE
    if logger is not None:
        logger.info(f"Loading tokenizer from {path}")
    state_dict = _load_training_state(path)
    vae_dict = OrderedDict()
    for prefix in ("vae.", "motion_vae.", "vqvae.", "model.vae.", "module.vae."):
        matched = {key[len(prefix):]: value for key, value in state_dict.items()
                   if key.startswith(prefix)}
        if matched:
            vae_dict.update(matched)
            break
    if not vae_dict:
        vae_dict.update(state_dict)
    tokenizer = model.vae if hasattr(model, "vae") else model.motion_vae
    tokenizer.load_state_dict(vae_dict, strict=True)
    return model
