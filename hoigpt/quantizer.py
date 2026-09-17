"""EMA vector quantization for structured hand-object tokens.

Portions are derived from T2M-GPT (Apache-2.0) and MotionGPT (MIT).
The shared-hand/separate-object design and this implementation were modified
for HOIGPT. See ``THIRD_PARTY_NOTICES.md`` and ``LICENSES/``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class EMACodebook(nn.Module):
    """A vector-quantization codebook updated with exponential moving averages.

    All training state is stored as module buffers, so device transfers and
    interrupted-training checkpoints preserve the EMA state. Older HOIGPT
    checkpoints that only contain ``codebook`` are upgraded during loading.
    """

    def __init__(self, size: int, dimension: int, decay: float = 0.99) -> None:
        super().__init__()
        if size <= 0:
            raise ValueError("Codebook size must be positive.")
        if dimension <= 0:
            raise ValueError("Code dimension must be positive.")
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1).")

        self.size = size
        self.dimension = dimension
        self.decay = decay
        self.register_buffer("codebook", torch.zeros(size, dimension))
        self.register_buffer("code_sum", torch.zeros(size, dimension))
        self.register_buffer("code_count", torch.zeros(size))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def reset(self) -> None:
        """Reset the codebook and all EMA statistics in place."""

        self.codebook.zero_()
        self.code_sum.zero_()
        self.code_count.zero_()
        self.initialized.fill_(False)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: Mapping[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Historical HOIGPT checkpoints persisted the codebook but not the EMA
        # accumulators or initialization flag. Seed the missing fields without
        # overwriting a trained codebook on the next training step.
        codebook_key = f"{prefix}codebook"
        if codebook_key in state_dict:
            loaded_codebook = state_dict[codebook_key]
            state_dict.setdefault(f"{prefix}code_sum", loaded_codebook.clone())
            state_dict.setdefault(
                f"{prefix}code_count",
                torch.ones(self.size, dtype=loaded_codebook.dtype, device=loaded_codebook.device),
            )
            state_dict.setdefault(
                f"{prefix}initialized",
                torch.tensor(True, dtype=torch.bool, device=loaded_codebook.device),
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _validate_vectors(self, vectors: Tensor) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Codebook vectors must have shape [items, {self.dimension}]; "
                f"received {tuple(vectors.shape)}."
            )
        if vectors.shape[0] == 0:
            raise ValueError("Cannot quantize an empty tensor.")
        if not vectors.is_floating_point():
            raise TypeError("Codebook vectors must use a floating-point dtype.")

    def _tile(self, vectors: Tensor) -> Tensor:
        if vectors.shape[0] >= self.size:
            return vectors
        repeats = math.ceil(self.size / vectors.shape[0])
        tiled = vectors.repeat(repeats, 1)
        noise_scale = 0.01 / math.sqrt(self.dimension)
        return tiled + torch.randn_like(tiled) * noise_scale

    @torch.no_grad()
    def initialize(self, vectors: Tensor) -> None:
        """Initialize code vectors from a representative training batch."""

        self._validate_vectors(vectors)
        initial = self._tile(vectors.detach())[: self.size]
        self.codebook.copy_(initial)
        self.code_sum.copy_(initial)
        self.code_count.fill_(1.0)
        self.initialized.fill_(True)

    def _require_initialized(self) -> None:
        if not bool(self.initialized.item()):
            raise RuntimeError(
                "The EMA codebook is uninitialized. Run one training forward pass "
                "or load a trained tokenizer checkpoint before encode/decode."
            )

    def indices(self, vectors: Tensor) -> Tensor:
        """Return nearest-code indices for a two-dimensional vector tensor."""

        self._validate_vectors(vectors)
        self._require_initialized()
        distances = (
            vectors.square().sum(dim=1, keepdim=True)
            - 2.0 * vectors @ self.codebook.t()
            + self.codebook.square().sum(dim=1).unsqueeze(0)
        )
        return distances.argmin(dim=1)

    def lookup(self, indices: Tensor) -> Tensor:
        """Map integer code indices to codebook vectors."""

        self._require_initialized()
        if indices.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("Token indices must use an integer dtype.")
        if indices.numel() == 0:
            raise ValueError("Token indices cannot be empty.")
        minimum = int(indices.min().item())
        maximum = int(indices.max().item())
        if minimum < 0 or maximum >= self.size:
            raise ValueError(
                f"Token indices must be in [0, {self.size - 1}], received [{minimum}, {maximum}]."
            )
        return functional.embedding(indices.to(torch.long), self.codebook)

    def perplexity(self, indices: Tensor) -> Tensor:
        """Compute code usage perplexity for a one-dimensional index tensor."""

        flat = indices.reshape(-1).to(torch.long)
        if flat.numel() == 0:
            raise ValueError("Cannot compute perplexity for empty indices.")
        counts = torch.bincount(flat, minlength=self.size).to(self.codebook.dtype)
        probabilities = counts / counts.sum()
        return torch.exp(-(probabilities * torch.log(probabilities + 1e-7)).sum())

    @torch.no_grad()
    def update(self, vectors: Tensor, indices: Tensor) -> None:
        """Apply one EMA update from sufficient statistics."""

        self._validate_vectors(vectors)
        flat_indices = indices.reshape(-1).to(torch.long)
        if flat_indices.shape[0] != vectors.shape[0]:
            raise ValueError("The number of code indices must match the number of vectors.")

        assignments = functional.one_hot(flat_indices, num_classes=self.size).to(vectors.dtype)
        batch_count = assignments.sum(dim=0).to(self.code_count.dtype)
        batch_sum = (assignments.t() @ vectors).to(self.code_sum.dtype)

        self.code_sum.mul_(self.decay).add_(batch_sum, alpha=1.0 - self.decay)
        self.code_count.mul_(self.decay).add_(batch_count, alpha=1.0 - self.decay)

        replacement = self._tile(vectors.detach())[: self.size].to(self.codebook.dtype)
        updated = self.code_sum / self.code_count.clamp_min(1e-7).unsqueeze(1)
        used = self.code_count.unsqueeze(1) >= 1.0
        self.codebook.copy_(torch.where(used, updated, replacement))


class HOIQuantizer(nn.Module):
    """Shared hand codebook plus an independent object codebook."""

    def __init__(self, codebook_size: int, code_dim: int, decay: float = 0.99) -> None:
        super().__init__()
        # Attribute names intentionally match historical HOIGPT checkpoints.
        self.quantizer_hands = EMACodebook(codebook_size, code_dim, decay)
        self.quantizer_obj = EMACodebook(codebook_size, code_dim, decay)
        self.codebook_size = codebook_size
        self.code_dim = code_dim

    def _flatten(self, features: Tensor, name: str) -> tuple[Tensor, int, int]:
        if features.ndim != 3 or features.shape[1] != self.code_dim:
            raise ValueError(
                f"{name} features must have shape [batch, {self.code_dim}, steps]; "
                f"received {tuple(features.shape)}."
            )
        batch_size, _, steps = features.shape
        if batch_size == 0 or steps == 0:
            raise ValueError(f"{name} features cannot have an empty batch or time axis.")
        flattened = features.permute(0, 2, 1).contiguous().reshape(-1, self.code_dim)
        return flattened, batch_size, steps

    @staticmethod
    def _restore(vectors: Tensor, batch_size: int, steps: int) -> Tensor:
        return vectors.reshape(batch_size, steps, -1).permute(0, 2, 1).contiguous()

    def quantize(
        self,
        left: Tensor,
        right: Tensor,
        obj: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Convert encoded feature maps into structured integer tokens."""

        left_flat, left_batch, left_steps = self._flatten(left, "Left-hand")
        right_flat, right_batch, right_steps = self._flatten(right, "Right-hand")
        obj_flat, obj_batch, obj_steps = self._flatten(obj, "Object")
        if (left_batch, left_steps) != (right_batch, right_steps) or (
            left_batch,
            left_steps,
        ) != (obj_batch, obj_steps):
            raise ValueError("Left-hand, right-hand, and object features must share batch and time shapes.")

        left_indices = self.quantizer_hands.indices(left_flat).reshape(left_batch, left_steps)
        right_indices = self.quantizer_hands.indices(right_flat).reshape(right_batch, right_steps)
        obj_indices = self.quantizer_obj.indices(obj_flat).reshape(obj_batch, obj_steps)
        return left_indices, right_indices, obj_indices

    def dequantize(
        self,
        left: Tensor,
        right: Tensor,
        obj: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Convert structured token tensors into channel-first latent maps."""

        if left.ndim != 2 or right.ndim != 2 or obj.ndim != 2:
            raise ValueError("Each structured token tensor must have shape [batch, steps].")
        if left.shape != right.shape or left.shape != obj.shape:
            raise ValueError("Left-hand, right-hand, and object tokens must have identical shapes.")
        if left.shape[0] == 0 or left.shape[1] == 0:
            raise ValueError("Structured token tensors cannot be empty.")

        batch_size, steps = left.shape
        left_vectors = self.quantizer_hands.lookup(left)
        right_vectors = self.quantizer_hands.lookup(right)
        obj_vectors = self.quantizer_obj.lookup(obj)
        return (
            self._restore(left_vectors, batch_size, steps),
            self._restore(right_vectors, batch_size, steps),
            self._restore(obj_vectors, batch_size, steps),
        )

    def forward(
        self,
        left: Tensor,
        right: Tensor,
        obj: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Quantize three streams and update codebooks once per training step."""

        left_flat, batch_size, steps = self._flatten(left, "Left-hand")
        right_flat, right_batch, right_steps = self._flatten(right, "Right-hand")
        obj_flat, obj_batch, obj_steps = self._flatten(obj, "Object")
        if (batch_size, steps) != (right_batch, right_steps) or (
            batch_size,
            steps,
        ) != (obj_batch, obj_steps):
            raise ValueError("Left-hand, right-hand, and object features must share batch and time shapes.")

        hands_flat = torch.cat([left_flat, right_flat], dim=0)
        if self.training and not bool(self.quantizer_hands.initialized.item()):
            self.quantizer_hands.initialize(hands_flat)
        if self.training and not bool(self.quantizer_obj.initialized.item()):
            self.quantizer_obj.initialize(obj_flat)

        left_indices = self.quantizer_hands.indices(left_flat)
        right_indices = self.quantizer_hands.indices(right_flat)
        obj_indices = self.quantizer_obj.indices(obj_flat)

        left_quantized = self.quantizer_hands.lookup(left_indices)
        right_quantized = self.quantizer_hands.lookup(right_indices)
        obj_quantized = self.quantizer_obj.lookup(obj_indices)

        if self.training:
            hand_indices = torch.cat([left_indices, right_indices], dim=0)
            self.quantizer_hands.update(hands_flat, hand_indices)
            self.quantizer_obj.update(obj_flat, obj_indices)

        commitment_loss = (
            functional.mse_loss(left_flat, left_quantized.detach())
            + functional.mse_loss(right_flat, right_quantized.detach())
            + functional.mse_loss(obj_flat, obj_quantized.detach())
        ) / 3.0

        left_quantized = left_flat + (left_quantized - left_flat).detach()
        right_quantized = right_flat + (right_quantized - right_flat).detach()
        obj_quantized = obj_flat + (obj_quantized - obj_flat).detach()

        perplexity = (
            self.quantizer_hands.perplexity(left_indices)
            + self.quantizer_hands.perplexity(right_indices)
            + self.quantizer_obj.perplexity(obj_indices)
        ) / 3.0

        return (
            self._restore(left_quantized, batch_size, steps),
            self._restore(right_quantized, batch_size, steps),
            self._restore(obj_quantized, batch_size, steps),
            commitment_loss,
            perplexity,
        )
