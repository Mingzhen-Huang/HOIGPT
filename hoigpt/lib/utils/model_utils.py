"""PointNet loading used by the original HOIGPT training pipeline.

Adapted from Text2HOI (MIT); see THIRD_PARTY_NOTICES.md.
"""

from pathlib import Path

import torch

from hoigpt.lib.networks.pointnet import PointNetfeat


def build_pointnetfeat(weight_path="checkpoints/arctic/pointfeat.pth"):
    """Load the pretrained object encoder without choosing a CUDA device."""
    if weight_path is None:
        weight_path = "checkpoints/arctic/pointfeat.pth"
    if not Path(weight_path).is_file():
        raise FileNotFoundError(
            f"PointNet checkpoint not found: {weight_path!r}. "
            "Set POINTNET.CHECKPOINT in configs/assets.yaml to your pretrained weights."
        )
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    point_encoder = PointNetfeat(global_feat=True, feature_transform=False, in_dim=3)
    point_encoder.load_state_dict(state, strict=True)
    point_encoder.eval().requires_grad_(False)
    return point_encoder
