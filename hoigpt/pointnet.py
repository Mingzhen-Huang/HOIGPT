"""PointNet encoder used for HOIGPT object conditioning.

Adapted from Text2HOI (Copyright (c) 2024 Junuk Cha, MIT), which
incorporates pointnet.pytorch (Copyright (c) 2017 Fei Xia, MIT).
This version was modified for HOIGPT. See ``THIRD_PARTY_NOTICES.md`` and
the license texts in ``LICENSES/``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class SpatialTransformer(nn.Module):
    """Learn a k-by-k alignment transform for a point feature tensor."""

    def __init__(self, dimensions: int = 64) -> None:
        super().__init__()
        if dimensions <= 0:
            raise ValueError("Point dimensions must be positive.")
        self.dimensions = dimensions
        self.conv1 = nn.Conv1d(dimensions, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, dimensions * dimensions)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != self.dimensions:
            raise ValueError(
                f"SpatialTransformer expects [batch, {self.dimensions}, points]; "
                f"received {tuple(inputs.shape)}."
            )
        batch_size = inputs.shape[0]
        outputs = functional.relu(self.bn1(self.conv1(inputs)))
        outputs = functional.relu(self.bn2(self.conv2(outputs)))
        outputs = functional.relu(self.bn3(self.conv3(outputs)))
        outputs = outputs.amax(dim=2)
        outputs = functional.relu(self.bn4(self.fc1(outputs)))
        outputs = functional.relu(self.bn5(self.fc2(outputs)))
        outputs = self.fc3(outputs)
        identity = torch.eye(
            self.dimensions,
            dtype=outputs.dtype,
            device=outputs.device,
        ).reshape(1, -1)
        return (outputs + identity.expand(batch_size, -1)).reshape(
            batch_size,
            self.dimensions,
            self.dimensions,
        )


class PointNetEncoder(nn.Module):
    """Encode an object point cloud into a 1024-dimensional global feature."""

    output_dim = 1024

    def __init__(self, input_dim: int = 3, *, feature_transform: bool = False) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("Point input dimension must be positive.")
        self.input_dim = input_dim
        self.feature_transform = feature_transform
        self.stn = SpatialTransformer(input_dim)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, self.output_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.output_dim)
        if feature_transform:
            self.fstn = SpatialTransformer(64)

    def forward(self, points: Tensor) -> Tensor:
        if points.ndim != 3 or points.shape[-1] != self.input_dim:
            raise ValueError(
                f"PointNetEncoder expects [batch, points, {self.input_dim}]; "
                f"received {tuple(points.shape)}."
            )
        if points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError("Point clouds must contain at least one batch item and one point.")
        if not points.is_floating_point():
            raise TypeError("Point clouds must use a floating-point dtype.")

        outputs = points.transpose(2, 1)
        transform = self.stn(outputs)
        outputs = torch.bmm(points, transform).transpose(2, 1)
        outputs = functional.relu(self.bn1(self.conv1(outputs)))

        if self.feature_transform:
            feature_transform = self.fstn(outputs)
            outputs = torch.bmm(outputs.transpose(2, 1), feature_transform).transpose(2, 1)

        outputs = functional.relu(self.bn2(self.conv2(outputs)))
        outputs = self.bn3(self.conv3(outputs))
        return outputs.amax(dim=2)


# Historical class names are aliases so existing PointNet state-dict keys remain usable.
STNkd = SpatialTransformer
PointNetfeat = PointNetEncoder
