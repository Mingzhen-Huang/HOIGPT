# Partially from https://github.com/Mael-zys/T2M-GPT

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.distribution import Distribution

from .tools.quantize_cnn import (
    LegacyQuantizeHOI,
    QuantizeDual,
    QuantizeEMA,
    QuantizeEMAReset,
    QuantizeHOI,
    QuantizeReset,
    Quantizer,
)
from .tools.resnet import Resnet1D
from hoigpt.lib.utils.model_utils import build_pointnetfeat


class VQVae(nn.Module):
    def __init__(
        self,
        nfeats: int,
        quantizer: str = "ema_reset",
        code_num: int = 512,
        code_dim: int = 512,
        output_emb_width: int = 512,
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        norm=None,
        activation: str = "relu",
        pointnet: bool = True,
        dual_decoder: bool = False,
        dual_codebook: bool = False,
        mae_mask_ratio: float = 0.5,
        cond_dim: int = 1024,
        pointnet_checkpoint: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.code_dim = code_dim
        self.output_emb_width = output_emb_width
        self.mae_mask_ratio = mae_mask_ratio
        self.dual_decoder = dual_decoder
        self.dual_codebook = dual_codebook
        self.use_pointnet = pointnet
        self.cond_dim = cond_dim if pointnet else 0

        if self.dual_codebook and self.dual_decoder:
            raise ValueError("`dual_codebook` paper mode and legacy `dual_decoder` mode are mutually exclusive.")

        self.pointnet = build_pointnetfeat(weight_path=pointnet_checkpoint).eval() if pointnet else None

        encoder_input_dim = nfeats + self.cond_dim
        decoder_input_dim = output_emb_width + self.cond_dim

        if self.dual_codebook:
            hand_dim = 99
            obj_dim = 10

            hand_encoder_dim = hand_dim + self.cond_dim
            obj_encoder_dim = obj_dim + self.cond_dim

            self.encoder_hand = Encoder(
                hand_encoder_dim,
                output_emb_width,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
            )
            self.encoder_obj = Encoder(
                obj_encoder_dim,
                output_emb_width,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
            )

            self.left_hand_pos_emb_enc = nn.Parameter(torch.randn(output_emb_width) * 0.02)
            self.right_hand_pos_emb_enc = nn.Parameter(torch.randn(output_emb_width) * 0.02)
            self.left_hand_pos_emb_dec = nn.Parameter(torch.randn(output_emb_width) * 0.02)
            self.right_hand_pos_emb_dec = nn.Parameter(torch.randn(output_emb_width) * 0.02)

            self.decoder_hand = Decoder(
                hand_dim,
                output_emb_width * 2,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
            )
            self.decoder_obj = Decoder(
                obj_dim,
                decoder_input_dim,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
            )
        elif self.dual_decoder:
            decoder_hand_dim = 198
            decoder_obj_dim = 10

            self.obj_decoder = Decoder(
                decoder_obj_dim,
                decoder_input_dim,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
            )
            self.hands_decoder = Decoder(
                decoder_hand_dim,
                output_emb_width + decoder_input_dim,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
            )
            self.obj_encoder2 = Encoder(
                decoder_obj_dim,
                output_emb_width,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
            )
            self.hand_encoder = Encoder(
                decoder_hand_dim + self.cond_dim,
                output_emb_width,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
            )
            self.encoder = Encoder(
                decoder_obj_dim + decoder_hand_dim + self.cond_dim,
                output_emb_width,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
            )
        else:
            self.encoder = Encoder(
                encoder_input_dim,
                output_emb_width,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
            )
            self.decoder = Decoder(
                nfeats,
                decoder_input_dim,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
            )

        if self.dual_codebook:
            self.quantizer = QuantizeHOI(code_num, code_dim, mu=0.99)
        elif self.dual_decoder:
            self.quantizer = LegacyQuantizeHOI(2 * code_num, code_dim, mu=0.99)
        elif quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99)
        elif quantizer == "dual":
            self.quantizer = QuantizeDual(code_num, code_dim, mu=0.99)
        elif quantizer == "orig":
            self.quantizer = Quantizer(code_num, code_dim, beta=1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(code_num, code_dim, mu=0.99)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(code_num, code_dim)
        else:
            raise ValueError(f"Unsupported quantizer: {quantizer}")

    def train(self, mode=True):
        super().train(mode)
        if self.pointnet is not None:
            self.pointnet.eval()
        return self

    def preprocess(self, x: Tensor) -> Tensor:
        return x.permute(0, 2, 1)

    def postprocess(self, x: Tensor) -> Tensor:
        return x.permute(0, 2, 1)

    def _pointnet_features(
        self,
        pc: Optional[Tensor],
        batch_size: int,
        target_steps: int,
        device: torch.device,
        detach: bool = True,
        required: bool = False,
    ) -> Optional[Tensor]:
        if not self.use_pointnet:
            return None

        if pc is None:
            if required:
                raise ValueError("Paper-mode HOI tokenization requires an object point cloud `pc`.")
            return torch.zeros(batch_size, self.cond_dim, target_steps, device=device)

        cond = self.pointnet(pc)
        if detach:
            cond = cond.detach()
        cond = cond.view(batch_size, self.cond_dim, -1)
        if cond.shape[-1] == target_steps:
            return cond
        if cond.shape[-1] == 1:
            return cond.repeat(1, 1, target_steps)
        return F.interpolate(cond, size=target_steps, mode="linear", align_corners=False)

    def _require_pc_for_paper_mode(self, pc: Optional[Tensor]) -> None:
        if self.dual_codebook and self.use_pointnet and pc is None:
            raise ValueError("Paper-mode encode/decode requires `pc`; legacy/demo flows should use legacy token mode.")

    def forward_dualdecoder(self, x_quantized_concat: Tensor, extra_dict=None) -> Tensor:
        obj_decoded = self.obj_decoder(x_quantized_concat)
        x_obj = self.obj_encoder2(obj_decoded)
        x_hoi = torch.cat([x_quantized_concat, x_obj], dim=1)
        x_hoi = self.hands_decoder(x_hoi)
        return torch.cat([x_hoi, obj_decoded], dim=1)

    def _encode_dual_codebook(self, x_in: Tensor, pc: Tensor) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        batch_size, _, time_steps = x_in.shape
        cond = self._pointnet_features(
            pc,
            batch_size=batch_size,
            target_steps=time_steps,
            device=x_in.device,
            detach=True,
            required=True,
        )

        x_left = x_in[:, :99, :]
        x_right = x_in[:, 99:198, :]
        x_obj = x_in[:, 198:208, :]

        if cond is not None:
            x_left_input = torch.cat([x_left, cond], dim=1)
            x_right_input = torch.cat([x_right, cond], dim=1)
            x_obj_input = torch.cat([x_obj, cond], dim=1)
        else:
            x_left_input = x_left
            x_right_input = x_right
            x_obj_input = x_obj

        x_left_encoded = self.encoder_hand(x_left_input)
        x_right_encoded = self.encoder_hand(x_right_input)
        x_obj_encoded = self.encoder_obj(x_obj_input)

        x_left_encoded = x_left_encoded + self.left_hand_pos_emb_enc.view(1, -1, 1)
        x_right_encoded = x_right_encoded + self.right_hand_pos_emb_enc.view(1, -1, 1)
        return x_left_encoded, x_right_encoded, x_obj_encoded, cond

    def _decode_dual_codebook(
        self,
        x_left: Tensor,
        x_right: Tensor,
        x_obj: Tensor,
        cond: Optional[Tensor],
    ) -> Tensor:
        x_left = x_left + self.left_hand_pos_emb_dec.view(1, -1, 1)
        x_right = x_right + self.right_hand_pos_emb_dec.view(1, -1, 1)

        if cond is not None:
            if cond.shape[-1] != x_obj.shape[-1]:
                cond = F.interpolate(cond, size=x_obj.shape[-1], mode="linear", align_corners=False)
            x_obj_input = torch.cat([x_obj, cond], dim=1)
        else:
            x_obj_input = x_obj

        x_obj_decoded = self.decoder_obj(x_obj_input)
        if cond is not None:
            cond_for_reencode = cond
            if cond_for_reencode.shape[-1] != x_obj_decoded.shape[-1]:
                cond_for_reencode = F.interpolate(
                    cond_for_reencode,
                    size=x_obj_decoded.shape[-1],
                    mode="linear",
                    align_corners=False,
                )
            x_obj_reencoded = self.encoder_obj(torch.cat([x_obj_decoded, cond_for_reencode], dim=1))
        else:
            x_obj_reencoded = self.encoder_obj(x_obj_decoded)
        if x_obj_reencoded.shape[-1] != x_left.shape[-1]:
            x_obj_reencoded = F.interpolate(
                x_obj_reencoded,
                size=x_left.shape[-1],
                mode="linear",
                align_corners=False,
            )

        x_left_decoded = self.decoder_hand(torch.cat([x_left, x_obj_reencoded], dim=1))
        x_right_decoded = self.decoder_hand(torch.cat([x_right, x_obj_reencoded], dim=1))
        return torch.cat([x_left_decoded, x_right_decoded, x_obj_decoded], dim=1)

    def forward(self, features: Tensor, pc: Optional[Tensor] = None):
        x_in = self.preprocess(features)
        batch_size = features.shape[0]

        if self.dual_codebook:
            self._require_pc_for_paper_mode(pc)
            x_left_encoded, x_right_encoded, x_obj_encoded, cond = self._encode_dual_codebook(x_in, pc)
            x_left_quantized, x_right_quantized, x_obj_quantized, loss, perplexity = self.quantizer(
                x_left_encoded, x_right_encoded, x_obj_encoded
            )
            x_decoder = self._decode_dual_codebook(
                x_left_quantized,
                x_right_quantized,
                x_obj_quantized,
                cond,
            )
            return self.postprocess(x_decoder), loss, perplexity

        cond = self._pointnet_features(
            pc,
            batch_size=batch_size,
            target_steps=x_in.shape[-1],
            device=x_in.device,
            detach=True,
            required=False,
        )
        x_input = torch.cat([x_in, cond], dim=1) if cond is not None else x_in

        if self.dual_decoder:
            x_encoder_hand = self.hand_encoder(torch.cat([x_in[:, :198, :], cond], dim=1) if cond is not None else x_in[:, :198, :])
            x_encoder_obj = self.encoder(x_input)
            x_encoder = torch.cat([x_encoder_hand, x_encoder_obj], dim=0)
        else:
            x_encoder = self.encoder(x_input)

        x_quantized, loss, perplexity = self.quantizer(x_encoder)

        if self.training and self.dual_decoder and self.mae_mask_ratio > 0:
            mask_prob = self.mae_mask_ratio
            mask_hand = (torch.rand(batch_size, 1, x_quantized.shape[2], device=x_quantized.device) > mask_prob).float()
            mask_obj = (torch.rand(batch_size, 1, x_quantized.shape[2], device=x_quantized.device) > mask_prob).float()
            x_quantized_hand = x_quantized[:batch_size] * mask_hand
            x_quantized_obj = x_quantized[batch_size:] * mask_obj
            x_quantized = torch.cat([x_quantized_hand, x_quantized_obj], dim=0)

        if cond is not None:
            cond_for_decode = cond
            if cond_for_decode.shape[-1] != x_quantized.shape[-1]:
                cond_for_decode = F.interpolate(
                    cond_for_decode,
                    size=x_quantized.shape[-1],
                    mode="linear",
                    align_corners=False,
                )
            x_quantized_concat = torch.cat([x_quantized, cond_for_decode], dim=1)
        else:
            x_quantized_concat = x_quantized

        if self.dual_decoder:
            x_decoder = self.forward_dualdecoder(x_quantized_concat)
        else:
            x_decoder = self.decoder(x_quantized_concat)
        return self.postprocess(x_decoder), loss, perplexity

    def encode(
        self,
        features: Tensor,
        pc: Optional[Tensor] = None,
    ) -> Union[Tensor, Dict[str, Tensor], Distribution]:
        n_batch = features.shape[0]
        x_in = self.preprocess(features)

        if self.dual_codebook:
            self._require_pc_for_paper_mode(pc)
            x_left_encoded, x_right_encoded, x_obj_encoded, _ = self._encode_dual_codebook(x_in, pc)
            code_idx_left, code_idx_right, code_idx_obj = self.quantizer.quantize(
                x_left_encoded,
                x_right_encoded,
                x_obj_encoded,
            )
            return {
                "left": code_idx_left.view(n_batch, -1),
                "right": code_idx_right.view(n_batch, -1),
                "obj": code_idx_obj.view(n_batch, -1),
            }, None

        cond = self._pointnet_features(
            pc,
            batch_size=n_batch,
            target_steps=x_in.shape[-1],
            device=x_in.device,
            detach=True,
            required=False,
        )
        x_encoder_input = torch.cat([x_in, cond], dim=1) if cond is not None else x_in
        x_encoder = self.encoder(x_encoder_input)
        x_encoder = self.postprocess(x_encoder).contiguous().view(-1, self.code_dim)
        code_idx = self.quantizer.quantize(x_encoder).view(n_batch, -1)
        return code_idx, None

    def decode(
        self,
        z: Union[Tensor, Dict[str, Tensor], Tuple[Tensor, Tensor, Tensor]],
        pc: Optional[Tensor] = None,
    ) -> Tensor:
        if self.dual_codebook:
            self._require_pc_for_paper_mode(pc)
            if isinstance(z, dict):
                z_left = z["left"]
                z_right = z["right"]
                z_obj = z["obj"]
            else:
                z_left, z_right, z_obj = z

            if z_left.dim() == 1:
                z_left = z_left.unsqueeze(0)
                z_right = z_right.unsqueeze(0)
                z_obj = z_obj.unsqueeze(0)

            n_batch, n_steps = z_left.shape
            x_left, x_right, x_obj = self.quantizer.dequantize(
                z_left.reshape(-1),
                z_right.reshape(-1),
                z_obj.reshape(-1),
            )
            x_left = x_left.view(n_batch, n_steps, self.code_dim).permute(0, 2, 1).contiguous()
            x_right = x_right.view(n_batch, n_steps, self.code_dim).permute(0, 2, 1).contiguous()
            x_obj = x_obj.view(n_batch, n_steps, self.code_dim).permute(0, 2, 1).contiguous()

            cond = self._pointnet_features(
                pc,
                batch_size=n_batch,
                target_steps=n_steps,
                device=x_left.device,
                detach=False,
                required=True,
            )
            x_decoder = self._decode_dual_codebook(x_left, x_right, x_obj, cond)
            return self.postprocess(x_decoder)

        if isinstance(z, dict):
            raise ValueError("Legacy decode expects a flat token tensor, not a structured HOI token dict.")
        if z.dim() == 1:
            z = z.unsqueeze(0)

        x_d = self.quantizer.dequantize(z).permute(0, 2, 1).contiguous()
        cond = self._pointnet_features(
            pc,
            batch_size=x_d.shape[0],
            target_steps=x_d.shape[-1],
            device=x_d.device,
            detach=False,
            required=False,
        )
        x_decoder_input = torch.cat([x_d, cond], dim=1) if cond is not None else x_d

        if self.dual_decoder:
            x_decoder = self.forward_dualdecoder(x_decoder_input)
        else:
            x_decoder = self.decoder(x_decoder_input)
        return self.postprocess(x_decoder)


class Encoder(nn.Module):
    def __init__(
        self,
        input_emb_width: int = 3,
        output_emb_width: int = 512,
        down_t: int = 3,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = "relu",
        norm=None,
    ):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for _ in range(down_t):
            block = nn.Sequential(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                Resnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    activation=activation,
                    norm=norm,
                ),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_emb_width: int = 3,
        output_emb_width: int = 512,
        down_t: int = 3,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = "relu",
        norm=None,
    ):
        super().__init__()
        blocks = []

        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for _ in range(down_t):
            block = nn.Sequential(
                Resnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    reverse_dilation=True,
                    activation=activation,
                    norm=norm,
                ),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(width, width, 3, 1, 1),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
