"""The release-facing HOIGPT hand-object tokenizer.

The encoder/decoder backbone is derived in part from T2M-GPT (Apache-2.0)
and MotionGPT (MIT). The structured hand/object design was developed for
HOIGPT. See ``THIRD_PARTY_NOTICES.md`` and ``LICENSES/``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .pointnet import PointNetEncoder
from .quantizer import HOIQuantizer
from .resnet import Resnet1D


class HOITokens(TypedDict):
    """Structured HOI token streams with shape ``[batch, steps]``."""

    left: Tensor
    right: Tensor
    obj: Tensor


class Encoder(nn.Module):
    """Temporal convolutional encoder used for each HOI stream."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        downsample_layers: int,
        width: int,
        residual_depth: int,
        dilation_growth_rate: int,
        activation: str,
        normalization: str | None,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = [nn.Conv1d(input_dim, width, 3, 1, 1), nn.ReLU()]
        for _ in range(downsample_layers):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(width, width, kernel_size=4, stride=2, padding=1),
                    Resnet1D(
                        width,
                        residual_depth,
                        dilation_growth_rate,
                        activation=activation,
                        normalization=normalization,
                    ),
                )
            )
        blocks.append(nn.Conv1d(width, latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)


class Decoder(nn.Module):
    """Temporal convolutional decoder used for each HOI stream."""

    def __init__(
        self,
        output_dim: int,
        input_dim: int,
        *,
        upsample_layers: int,
        width: int,
        residual_depth: int,
        dilation_growth_rate: int,
        activation: str,
        normalization: str | None,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = [nn.Conv1d(input_dim, width, 3, 1, 1), nn.ReLU()]
        for _ in range(upsample_layers):
            blocks.append(
                nn.Sequential(
                    Resnet1D(
                        width,
                        residual_depth,
                        dilation_growth_rate,
                        reverse_dilation=True,
                        activation=activation,
                        normalization=normalization,
                    ),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv1d(width, width, 3, 1, 1),
                )
            )
        blocks.extend(
            [
                nn.Conv1d(width, width, 3, 1, 1),
                nn.ReLU(),
                nn.Conv1d(width, output_dim, 3, 1, 1),
            ]
        )
        self.model = nn.Sequential(*blocks)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)


class HOITokenizer(nn.Module):
    """Paper-aligned HOIGPT tokenizer.

    The input feature layout is fixed to 208 channels:

    - left hand: ``0:99``
    - right hand: ``99:198``
    - object: ``198:208``

    Left and right hands share a codebook. The object uses a separate
    codebook. Object motion is decoded first and then conditions both hand
    decoders.
    """

    hand_feature_dim = 99
    object_feature_dim = 10
    feature_dim = 208

    def __init__(
        self,
        *,
        codebook_size: int = 512,
        latent_dim: int = 512,
        downsample_layers: int = 2,
        width: int = 512,
        residual_depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = "relu",
        normalization: str | None = None,
        ema_decay: float = 0.99,
        point_conditioning: bool = True,
        condition_dim: int = 1024,
        point_encoder: nn.Module | None = None,
        freeze_point_encoder: bool = True,
    ) -> None:
        super().__init__()
        if downsample_layers < 0:
            raise ValueError("downsample_layers cannot be negative.")
        if width <= 0 or latent_dim <= 0:
            raise ValueError("width and latent_dim must be positive.")
        if point_conditioning and condition_dim <= 0:
            raise ValueError("condition_dim must be positive when point conditioning is enabled.")
        if not point_conditioning and point_encoder is not None:
            raise ValueError("point_encoder cannot be supplied when point conditioning is disabled.")
        if point_encoder is not None and not isinstance(point_encoder, nn.Module):
            raise TypeError("point_encoder must be a torch.nn.Module.")
        if (
            point_conditioning
            and point_encoder is None
            and condition_dim != PointNetEncoder.output_dim
        ):
            raise ValueError(
                f"The default PointNet produces {PointNetEncoder.output_dim} features; "
                "supply a matching point_encoder to use a different condition_dim."
            )

        self.code_dim = latent_dim
        self.output_emb_width = latent_dim
        self.downsample_layers = downsample_layers
        self.downsample_factor = 2**downsample_layers
        self.use_pointnet = point_conditioning
        self.cond_dim = condition_dim if point_conditioning else 0
        self.freeze_point_encoder = freeze_point_encoder

        if point_conditioning:
            self.pointnet = point_encoder if point_encoder is not None else PointNetEncoder()
            if freeze_point_encoder:
                self.pointnet.requires_grad_(False)
                self.pointnet.eval()
        else:
            self.pointnet = None

        hand_encoder_dim = self.hand_feature_dim + self.cond_dim
        object_encoder_dim = self.object_feature_dim + self.cond_dim

        encoder_options = dict(
            downsample_layers=downsample_layers,
            width=width,
            residual_depth=residual_depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            normalization=normalization,
        )
        decoder_options = dict(
            upsample_layers=downsample_layers,
            width=width,
            residual_depth=residual_depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            normalization=normalization,
        )

        # Attribute names match the paper-mode model used during development,
        # allowing its state dictionaries to be migrated without key rewrites.
        self.encoder_hand = Encoder(hand_encoder_dim, latent_dim, **encoder_options)
        self.encoder_obj = Encoder(object_encoder_dim, latent_dim, **encoder_options)
        self.left_hand_pos_emb_enc = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.right_hand_pos_emb_enc = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.left_hand_pos_emb_dec = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.right_hand_pos_emb_dec = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.decoder_hand = Decoder(
            self.hand_feature_dim,
            latent_dim * 2,
            **decoder_options,
        )
        self.decoder_obj = Decoder(
            self.object_feature_dim,
            latent_dim + self.cond_dim,
            **decoder_options,
        )
        self.quantizer = HOIQuantizer(codebook_size, latent_dim, ema_decay)

    def train(self, mode: bool = True) -> "HOITokenizer":
        super().train(mode)
        if self.pointnet is not None and self.freeze_point_encoder:
            self.pointnet.eval()
        return self

    def _validate_features(self, features: Tensor) -> None:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"HOI features must have shape [batch, time, {self.feature_dim}]; "
                f"received {tuple(features.shape)}."
            )
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("HOI features cannot have an empty batch or time axis.")
        if features.shape[1] % self.downsample_factor != 0:
            raise ValueError(
                f"The time length must be divisible by {self.downsample_factor}; "
                f"received {features.shape[1]}."
            )
        if not features.is_floating_point():
            raise TypeError("HOI features must use a floating-point dtype.")

    def _validate_point_cloud(self, point_cloud: Tensor, reference: Tensor) -> None:
        if point_cloud.ndim != 3 or point_cloud.shape[-1] != 3:
            raise ValueError(
                "Object point clouds must have shape [batch, points, 3]; "
                f"received {tuple(point_cloud.shape)}."
            )
        if point_cloud.shape[0] != reference.shape[0]:
            raise ValueError("Point-cloud and feature/token batch sizes must match.")
        if point_cloud.shape[1] == 0:
            raise ValueError("Object point clouds cannot be empty.")
        if not point_cloud.is_floating_point():
            raise TypeError("Object point clouds must use a floating-point dtype.")
        if point_cloud.device != reference.device:
            raise ValueError("Point clouds and model inputs must be on the same device.")
        if point_cloud.dtype != reference.dtype:
            raise ValueError("Point clouds and model inputs must use the same dtype.")

    def _point_features(
        self,
        point_cloud: Tensor | None,
        reference: Tensor,
        target_steps: int,
    ) -> Tensor | None:
        if not self.use_pointnet:
            if point_cloud is not None:
                raise ValueError("This tokenizer was constructed without point conditioning.")
            return None
        if point_cloud is None:
            raise ValueError("Object point cloud `point_cloud` is required for this tokenizer.")
        assert self.pointnet is not None
        self._validate_point_cloud(point_cloud, reference)

        if self.freeze_point_encoder:
            with torch.no_grad():
                condition = self.pointnet(point_cloud)
        else:
            condition = self.pointnet(point_cloud)
        if condition.ndim != 2 or condition.shape != (reference.shape[0], self.cond_dim):
            raise ValueError(
                f"The point encoder must return [batch, {self.cond_dim}]; "
                f"received {tuple(condition.shape)}."
            )
        if condition.device != reference.device:
            raise ValueError("The point encoder output must be on the input device.")
        if not condition.is_floating_point():
            raise TypeError("The point encoder output must use a floating-point dtype.")
        # Autocast may legitimately make PointNet return float16/bfloat16 while
        # the original feature tensor remains float32. Normalize the internal
        # condition before concatenating it with the feature streams.
        condition = condition.to(dtype=reference.dtype)
        return condition.unsqueeze(-1).expand(-1, -1, target_steps)

    @staticmethod
    def _resize_time(features: Tensor, steps: int) -> Tensor:
        if features.shape[-1] == steps:
            return features
        return functional.interpolate(features, size=steps, mode="linear", align_corners=False)

    def _encode_latents(
        self,
        features: Tensor,
        point_cloud: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        self._validate_features(features)
        channel_first = features.permute(0, 2, 1)
        condition = self._point_features(point_cloud, channel_first, features.shape[1])

        left = channel_first[:, : self.hand_feature_dim]
        right = channel_first[:, self.hand_feature_dim : self.hand_feature_dim * 2]
        obj = channel_first[:, self.hand_feature_dim * 2 :]
        if condition is not None:
            left = torch.cat([left, condition], dim=1)
            right = torch.cat([right, condition], dim=1)
            obj = torch.cat([obj, condition], dim=1)

        left = self.encoder_hand(left) + self.left_hand_pos_emb_enc.reshape(1, -1, 1)
        right = self.encoder_hand(right) + self.right_hand_pos_emb_enc.reshape(1, -1, 1)
        obj = self.encoder_obj(obj)
        return left, right, obj, condition

    def _decode_latents(
        self,
        left: Tensor,
        right: Tensor,
        obj: Tensor,
        condition: Tensor | None,
    ) -> Tensor:
        left = left + self.left_hand_pos_emb_dec.reshape(1, -1, 1)
        right = right + self.right_hand_pos_emb_dec.reshape(1, -1, 1)

        latent_condition = None
        if condition is not None:
            latent_condition = self._resize_time(condition, obj.shape[-1])
            object_input = torch.cat([obj, latent_condition], dim=1)
        else:
            object_input = obj
        decoded_obj = self.decoder_obj(object_input)

        if condition is not None:
            full_condition = self._resize_time(condition, decoded_obj.shape[-1])
            reencoded_obj = self.encoder_obj(torch.cat([decoded_obj, full_condition], dim=1))
        else:
            reencoded_obj = self.encoder_obj(decoded_obj)
        reencoded_obj = self._resize_time(reencoded_obj, left.shape[-1])

        decoded_left = self.decoder_hand(torch.cat([left, reencoded_obj], dim=1))
        decoded_right = self.decoder_hand(torch.cat([right, reencoded_obj], dim=1))
        return torch.cat([decoded_left, decoded_right, decoded_obj], dim=1)

    def forward(
        self,
        features: Tensor,
        point_cloud: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Reconstruct features and return commitment loss and perplexity."""

        left, right, obj, condition = self._encode_latents(features, point_cloud)
        left, right, obj, commitment_loss, perplexity = self.quantizer(left, right, obj)
        reconstructed = self._decode_latents(left, right, obj, condition)
        return reconstructed.permute(0, 2, 1), commitment_loss, perplexity

    @torch.no_grad()
    def encode(
        self,
        features: Tensor,
        point_cloud: Tensor | None = None,
    ) -> HOITokens:
        """Encode HOI features into left/right/object token streams."""

        left, right, obj, _ = self._encode_latents(features, point_cloud)
        left_tokens, right_tokens, object_tokens = self.quantizer.quantize(left, right, obj)
        return HOITokens(left=left_tokens, right=right_tokens, obj=object_tokens)

    def _validate_tokens(self, tokens: Mapping[str, Tensor]) -> HOITokens:
        expected = {"left", "right", "obj"}
        actual = set(tokens)
        if actual != expected:
            raise ValueError(f"Structured tokens must have exactly {sorted(expected)}; received {sorted(actual)}.")
        left = tokens["left"]
        right = tokens["right"]
        obj = tokens["obj"]
        if not all(isinstance(value, Tensor) for value in (left, right, obj)):
            raise TypeError("Every structured token stream must be a torch.Tensor.")
        if left.ndim != 2 or right.ndim != 2 or obj.ndim != 2:
            raise ValueError("Every structured token stream must have shape [batch, steps].")
        if left.shape != right.shape or left.shape != obj.shape:
            raise ValueError("Left-hand, right-hand, and object token shapes must match.")
        if left.shape[0] == 0 or left.shape[1] == 0:
            raise ValueError("Structured token streams cannot be empty.")
        if left.device != right.device or left.device != obj.device:
            raise ValueError("Structured token streams must be on the same device.")
        return HOITokens(left=left, right=right, obj=obj)

    @torch.no_grad()
    def decode(
        self,
        tokens: Mapping[str, Tensor],
        point_cloud: Tensor | None = None,
    ) -> Tensor:
        """Decode structured tokens into 208-dimensional HOI features."""

        validated = self._validate_tokens(tokens)
        left, right, obj = self.quantizer.dequantize(
            validated["left"],
            validated["right"],
            validated["obj"],
        )
        output_steps = validated["left"].shape[1] * self.downsample_factor
        condition = self._point_features(point_cloud, left, output_steps)
        decoded = self._decode_latents(left, right, obj, condition)
        return decoded.permute(0, 2, 1)

    def load_checkpoint_state(
        self,
        checkpoint: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> nn.modules.module._IncompatibleKeys:
        """Load a tokenizer or Lightning checkpoint without unpickling a model.

        The mapping may be a raw tokenizer state dict or contain a ``state_dict``
        entry. Common full-model prefixes such as ``vae.`` are recognized.
        """

        if not isinstance(checkpoint, Mapping):
            raise TypeError("checkpoint must be a mapping containing tensor state.")

        candidate: Mapping[str, Any]
        nested = checkpoint.get("state_dict")
        if isinstance(nested, Mapping):
            candidate = nested
        else:
            candidate = checkpoint
        if not candidate or not all(isinstance(key, str) for key in candidate):
            raise ValueError("Checkpoint does not contain a string-keyed state dict.")

        raw_state = {key: value for key, value in candidate.items() if isinstance(value, Tensor)}
        model_keys = set(self.state_dict())
        direct_score = len(model_keys.intersection(raw_state))
        best_state = raw_state
        best_score = direct_score
        best_prefix = ""
        for prefix in ("vae.", "model.vae.", "module.vae.", "module."):
            stripped = {
                key[len(prefix) :]: value
                for key, value in raw_state.items()
                if key.startswith(prefix)
            }
            score = len(model_keys.intersection(stripped))
            if score > best_score:
                best_state = stripped
                best_score = score
                best_prefix = prefix
        if best_score == 0:
            raise ValueError("Checkpoint contains no HOIGPT tokenizer parameters.")

        return self.load_state_dict(best_state, strict=strict)


class VQVae(HOITokenizer):
    """Compatibility adapter for the historical paper-mode ``VQVae`` API.

    New code should instantiate :class:`HOITokenizer` directly. Legacy
    non-paper quantizers and the broken dual-decoder path are intentionally not
    part of the core release.
    """

    def __init__(
        self,
        nfeats: int = 208,
        quantizer: str = "ema_reset",
        code_num: int = 512,
        code_dim: int = 512,
        output_emb_width: int = 512,
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        norm: str | None = None,
        activation: str = "relu",
        pointnet: bool = True,
        dual_decoder: bool = False,
        dual_codebook: bool = True,
        mae_mask_ratio: float = 0.5,
        cond_dim: int = 1024,
        **kwargs: Any,
    ) -> None:
        if nfeats != self.feature_dim:
            raise ValueError(f"Paper-mode VQVae requires nfeats={self.feature_dim}.")
        if quantizer != "ema_reset":
            raise ValueError("The core release only supports the paper EMA quantizer.")
        if code_dim != output_emb_width:
            raise ValueError("code_dim and output_emb_width must match.")
        if stride_t != 2:
            raise ValueError("The paper tokenizer only supports stride_t=2.")
        if dual_decoder or not dual_codebook:
            raise ValueError("The core release only supports paper-mode dual_codebook=True.")

        normalized_norm = norm
        if isinstance(norm, str) and norm.lower() in {"none", "null"}:
            normalized_norm = None

        point_encoder = kwargs.pop("point_encoder", None)
        if point_encoder is not None and not isinstance(point_encoder, nn.Module):
            raise TypeError("point_encoder must be a torch.nn.Module.")
        freeze_point_encoder = bool(kwargs.pop("freeze_point_encoder", True))
        ema_decay = float(kwargs.pop("ema_decay", 0.99))
        # Historical configs carry these legacy-only options. They have no
        # effect in paper mode and are accepted solely for config migration.
        del mae_mask_ratio
        kwargs.pop("ablation", None)
        if kwargs:
            raise TypeError(f"Unsupported VQVae options: {sorted(kwargs)}")

        super().__init__(
            codebook_size=code_num,
            latent_dim=code_dim,
            downsample_layers=down_t,
            width=width,
            residual_depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            normalization=normalized_norm,
            ema_decay=ema_decay,
            point_conditioning=pointnet,
            condition_dim=cond_dim,
            point_encoder=point_encoder,
            freeze_point_encoder=freeze_point_encoder,
        )

    def forward(
        self,
        features: Tensor,
        pc: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the tokenizer using the historical ``pc`` keyword."""

        return super().forward(features, pc)

    @torch.no_grad()
    def encode(
        self,
        features: Tensor,
        pc: Tensor | None = None,
    ) -> tuple[HOITokens, None]:
        return super().encode(features, pc), None

    @torch.no_grad()
    def decode(
        self,
        tokens: Mapping[str, Tensor] | tuple[Tensor, Tensor, Tensor],
        pc: Tensor | None = None,
    ) -> Tensor:
        if isinstance(tokens, tuple):
            if len(tokens) != 3:
                raise ValueError("A token tuple must contain left, right, and object streams.")
            structured: Mapping[str, Tensor] = {
                "left": tokens[0],
                "right": tokens[1],
                "obj": tokens[2],
            }
        elif isinstance(tokens, Mapping):
            structured = tokens
        else:
            raise TypeError("Paper-mode VQVae decode expects a token mapping or three-tensor tuple.")

        values = tuple(structured.values())
        if len(values) == 3 and all(isinstance(value, Tensor) and value.ndim == 1 for value in values):
            structured = {key: value.unsqueeze(0) for key, value in structured.items()}
        return super().decode(structured, pc)
