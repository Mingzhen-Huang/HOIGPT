"""One-dimensional residual blocks used by the HOIGPT tokenizer.

Portions are derived from T2M-GPT (Apache-2.0) and MotionGPT (MIT).
This version was modified for HOIGPT. See ``THIRD_PARTY_NOTICES.md`` and
the license texts in ``LICENSES/``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Swish(nn.Module):
    """The parameter-free SiLU/Swish activation."""

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * torch.sigmoid(inputs)


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized in {"silu", "swish"}:
        return Swish()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name!r}")


def _normalization(name: str | None, channels: int) -> nn.Module:
    if name is None:
        return nn.Identity()
    normalized = name.upper()
    if normalized == "LN":
        return nn.LayerNorm(channels)
    if normalized == "GN":
        if channels % 32 != 0:
            raise ValueError("GroupNorm requires the channel count to be divisible by 32.")
        return nn.GroupNorm(32, channels, eps=1e-6, affine=True)
    if normalized == "BN":
        return nn.BatchNorm1d(channels, eps=1e-6, affine=True)
    raise ValueError(f"Unsupported normalization: {name!r}")


class ResidualConv1DBlock(nn.Module):
    """A dilated residual Conv1d block."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        dilation: int = 1,
        activation: str = "silu",
        normalization: str | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0 or hidden_channels <= 0:
            raise ValueError("Channel counts must be positive.")
        if dilation <= 0:
            raise ValueError("Dilation must be positive.")

        self.normalization = normalization.upper() if normalization else None
        self.norm1 = _normalization(normalization, channels)
        self.norm2 = _normalization(normalization, channels)
        self.activation1 = _activation(activation)
        self.activation2 = _activation(activation)
        self.conv1 = nn.Conv1d(
            channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(hidden_channels, channels, kernel_size=1)

    def _normalize_and_activate(
        self,
        inputs: Tensor,
        normalization: nn.Module,
        activation: nn.Module,
    ) -> Tensor:
        if self.normalization == "LN":
            return activation(normalization(inputs.transpose(-2, -1)).transpose(-2, -1))
        return activation(normalization(inputs))

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        outputs = self._normalize_and_activate(inputs, self.norm1, self.activation1)
        outputs = self.conv1(outputs)
        outputs = self._normalize_and_activate(outputs, self.norm2, self.activation2)
        return self.conv2(outputs) + residual


class Resnet1D(nn.Module):
    """A stack of dilated residual blocks.

    The ``model`` attribute is retained for compatibility with existing
    HOIGPT tokenizer state dictionaries.
    """

    def __init__(
        self,
        channels: int,
        depth: int,
        dilation_growth_rate: int = 1,
        *,
        reverse_dilation: bool = True,
        activation: str = "relu",
        normalization: str | None = None,
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError("Residual depth cannot be negative.")
        if dilation_growth_rate <= 0:
            raise ValueError("Dilation growth rate must be positive.")

        blocks = [
            ResidualConv1DBlock(
                channels,
                channels,
                dilation=dilation_growth_rate**index,
                activation=activation,
                normalization=normalization,
            )
            for index in range(depth)
        ]
        if reverse_dilation:
            blocks.reverse()
        self.model = nn.Sequential(*blocks)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)
