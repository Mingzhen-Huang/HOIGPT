# Partially from https://github.com/Mael-zys/T2M-GPT

from typing import List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from .tools.resnet import Resnet1D
from .tools.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset, QuantizeHOI
from collections import OrderedDict

from lib.utils.model_utils import (
    build_pointnetfeat, 
)

class VQVae(nn.Module):

    def __init__(self,
                 nfeats: int,
                 quantizer: str = "ema_reset",
                 code_num=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 pointnet = True,
                 cond_dim = 1024,
                 **kwargs) -> None:

        super().__init__()

        self.code_dim = code_dim
        self.output_emb_width = output_emb_width
        self.pointnet = pointnet
        self.dual_codebook = kwargs.get('dual_codebook', False)

        if pointnet:
            encoder_dim_encoder = nfeats + cond_dim 
            output_emb_width_decoder = output_emb_width + cond_dim
            self.pointnet = build_pointnetfeat().eval()
        else:
            encoder_dim_encoder = nfeats
            output_emb_width_decoder = output_emb_width

        
        
        if self.dual_codebook:
            # Dual codebook mode: shared encoder/decoder for both hands with positional embedding
            hand_dim = 99  # Features for one hand
            decoder_obj_dim = 10
            
            # Shared hand encoder for both left and right hands
            if pointnet:
                self.encoder_hand = Encoder(hand_dim + cond_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
                self.encoder_obj = Encoder(decoder_obj_dim + cond_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
            else:
                self.encoder_hand = Encoder(hand_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
                self.encoder_obj = Encoder(decoder_obj_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
            
            # Learnable positional embeddings to distinguish left and right hands
            # Encoder positional embeddings (added after encoding)
            self.left_hand_pos_emb_enc = nn.Parameter(torch.randn(output_emb_width) * 0.02)
            self.right_hand_pos_emb_enc = nn.Parameter(torch.randn(output_emb_width) * 0.02)
            
            # Decoder positional embeddings (added before decoding, after dequantization)
            self.left_hand_pos_emb_dec = nn.Parameter(torch.randn(output_emb_width) * 0.02)
            self.right_hand_pos_emb_dec = nn.Parameter(torch.randn(output_emb_width) * 0.02)
            
            # Shared decoder for both hands, separate decoder for object
            # Hand decoder input: hand_quantized (output_emb_width) + obj_reencoded (output_emb_width)
            # Note: pointnet features are NOT used during decode
            hand_decoder_input_dim = output_emb_width * 2
            self.decoder_hand = Decoder(hand_dim, hand_decoder_input_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
            self.decoder_obj = Decoder(decoder_obj_dim, output_emb_width_decoder, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        else:
            self.encoder = Encoder(encoder_dim_encoder,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)
            self.decoder = Decoder(nfeats,
                                output_emb_width_decoder,
                                down_t,
                                stride_t,
                                width,
                                depth,
                                dilation_growth_rate,
                                activation=activation,
                                norm=norm)

        if self.dual_codebook:
            self.quantizer = QuantizeHOI(code_num, code_dim, mu=0.99)
        elif quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99)
        elif quantizer == "orig":
            self.quantizer = Quantizer(code_num, code_dim, beta=1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(code_num, code_dim, mu=0.99)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(code_num, code_dim)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: Tensor, pc: Tensor):
        # Preprocess
        x_in = self.preprocess(features)
        bs = features.shape[0]

        # import pdb; pdb.set_trace()

        if self.pointnet:
            cond = self.pointnet(pc).detach()
            cfeats = cond.view(bs, 1024, -1)
            if cfeats.shape[2] != x_in.shape[2]:
                crepfeats = cfeats.repeat([1,1, int(x_in.shape[2])])
            
            x_input = torch.cat([x_in, crepfeats], dim=1)

        # Encode
        if self.dual_codebook:
            # Split features into left hand (0:99), right hand (99:198), and object (198:208)
            x_left = x_in[:, :99, :]
            x_right = x_in[:, 99:198, :]
            x_obj = x_in[:, 198:208, :]
            
            if self.pointnet:
                x_left_input = torch.cat([x_left, crepfeats], dim=1)
                x_right_input = torch.cat([x_right, crepfeats], dim=1)
                x_obj_input = torch.cat([x_obj, crepfeats], dim=1)
            else:
                x_left_input = x_left
                x_right_input = x_right
                x_obj_input = x_obj
            
            # Encode hands with shared encoder
            x_encoder_left = self.encoder_hand(x_left_input)
            x_encoder_right = self.encoder_hand(x_right_input)
            
            # Add encoder positional embeddings to distinguish left and right hands
            # Reshape positional embeddings: (output_emb_width,) -> (1, output_emb_width, 1)
            left_pos_enc = self.left_hand_pos_emb_enc.view(1, -1, 1)
            right_pos_enc = self.right_hand_pos_emb_enc.view(1, -1, 1)
            x_encoder_left = x_encoder_left + left_pos_enc
            x_encoder_right = x_encoder_right + right_pos_enc
            
            # Encode object separately
            x_encoder_obj = self.encoder_obj(x_obj_input)
            
            # Quantize with three separate codebooks
            x_quantized_left, x_quantized_right, x_quantized_obj, loss, perplexity = self.quantizer(
                x_encoder_left, x_encoder_right, x_encoder_obj
            )
            
            # # Prepare for decoding
            if self.pointnet:
                if cfeats.shape[2] != x_quantized_left.shape[2]:
                    crepfeats_dec = cfeats.repeat([1,1, int(x_quantized_left.shape[2]/cfeats.shape[2])])
                else:
                    crepfeats_dec = crepfeats
                
                x_quantized_obj_concat = torch.cat([x_quantized_obj, crepfeats_dec], dim=1)
            else:
                x_quantized_obj_concat = x_quantized_obj
            
            # Step 1: Decode object first
            x_decoder_obj = self.decoder_obj(x_quantized_obj_concat)
            
            # Step 2: Re-encode the decoded object motion
            x_obj_reencoded = self.encoder_obj(x_decoder_obj)
            
            # Step 3: Concatenate object features to hand features
            # Ensure temporal dimensions match
            if x_obj_reencoded.shape[2] != x_quantized_left.shape[2]:
                x_obj_reencoded = torch.nn.functional.interpolate(
                    x_obj_reencoded, 
                    size=x_quantized_left.shape[2], 
                    mode='linear', 
                    align_corners=False
                )
            
            x_quantized_left_with_obj = torch.cat([x_quantized_left, x_obj_reencoded], dim=1)
            x_quantized_right_with_obj = torch.cat([x_quantized_right, x_obj_reencoded], dim=1)
            
            # Step 4: Decode hands with object context (without pointnet features)
            x_decoder_left = self.decoder_hand(x_quantized_left_with_obj)
            x_decoder_right = self.decoder_hand(x_quantized_right_with_obj)
            
            # Concatenate decoded features
            x_decoder = torch.cat([x_decoder_left, x_decoder_right, x_decoder_obj], dim=1)
        else:
            x_encoder = self.encoder(x_input)
            
            # quantization
            x_quantized, loss, perplexity = self.quantizer(x_encoder)
            
            if self.pointnet:
                if cfeats.shape[2] != x_quantized.shape[2]:
                    crepfeats = cfeats.repeat([1,1, int(x_quantized.shape[2]/cfeats.shape[2])])
                
                x_quantized_concat = torch.cat([x_quantized, crepfeats], dim=1)
            
            x_decoder = self.decoder(x_quantized_concat)
            
        x_out = self.postprocess(x_decoder)

        

        return x_out, loss, perplexity

    def encode(
        self,
        features: Tensor, pc: Tensor
    ) -> Union[Tensor, Distribution]:

        N, T, _ = features.shape
        x_in = self.preprocess(features)

        if self.dual_codebook:
            # Split features into left hand, right hand, and object
            x_left = x_in[:, :99, :]
            x_right = x_in[:, 99:198, :]
            x_obj = x_in[:, 198:208, :]
            
            if self.pointnet:
                cond = self.pointnet(pc).detach()
                cfeats = cond.view(N, 1024, -1)
                crep = cfeats.repeat(1, 1, x_in.shape[-1])
                
                x_left_input = torch.cat([x_left, crep], dim=1)
                x_right_input = torch.cat([x_right, crep], dim=1)
                x_obj_input = torch.cat([x_obj, crep], dim=1)
            else:
                x_left_input = x_left
                x_right_input = x_right
                x_obj_input = x_obj
            
            # Encode hands with shared encoder
            x_encoder_left = self.encoder_hand(x_left_input)
            x_encoder_right = self.encoder_hand(x_right_input)
            
            # Add encoder positional embeddings to distinguish left and right hands
            left_pos_enc = self.left_hand_pos_emb_enc.view(1, -1, 1)
            right_pos_enc = self.right_hand_pos_emb_enc.view(1, -1, 1)
            x_encoder_left = x_encoder_left + left_pos_enc
            x_encoder_right = x_encoder_right + right_pos_enc
            
            # Encode object separately
            x_encoder_obj = self.encoder_obj(x_obj_input)
            
            # Quantize to get code indices
            code_idx_left, code_idx_right, code_idx_obj = self.quantizer.quantize(
                x_encoder_left, x_encoder_right, x_encoder_obj
            )
            
            # Reshape to (N, T)
            code_idx_left = code_idx_left.view(N, -1)
            code_idx_right = code_idx_right.view(N, -1)
            code_idx_obj = code_idx_obj.view(N, -1)
            
            # Return as a dictionary to distinguish three token sequences
            return {
                'left': code_idx_left,
                'right': code_idx_right,
                'obj': code_idx_obj
            }, None
        else:
            if self.pointnet:
                cond = self.pointnet(pc).detach()
                cfeats = cond.view(N, 1024, -1)
                crep = cfeats.repeat(1, 1, x_in.shape[-1])
                x_in = torch.cat([x_in, crep], dim=1)

            x_encoder = self.encoder(x_in)
            x_encoder = self.postprocess(x_encoder)
            x_encoder = x_encoder.contiguous().view(-1,
                                                    x_encoder.shape[-1])  # (NT, C)
            code_idx = self.quantizer.quantize(x_encoder)
   
            code_idx = code_idx.view(N, -1)

            # latent, dist
            return code_idx, None

    def decode(self, z, pc):
        if self.dual_codebook:
            # z is a dictionary with keys 'left', 'right', 'obj'
            if isinstance(z, dict):
                z_left = z['left']
                z_right = z['right']
                z_obj = z['obj']
                N = z_left.shape[0]  # batch size
                T = z_left.shape[1]  # time steps
                z_left = z_left.flatten()
                z_right = z_right.flatten()
                z_obj = z_obj.flatten()
            else:
                # Assume z is a tuple of three tensors
                z_left, z_right, z_obj = z
                N = z_left.shape[0]
                T = z_left.shape[1]
                z_left = z_left.flatten()
                z_right = z_right.flatten()
                z_obj = z_obj.flatten()
            
            # Dequantize each part
            x_d_left, x_d_right, x_d_obj = self.quantizer.dequantize(z_left, z_right, z_obj)
            
            # Reshape: (NT, code_dim) -> (N, code_dim, T)
            x_d_left = x_d_left.view(N, T, self.code_dim).permute(0, 2, 1).contiguous()
            x_d_right = x_d_right.view(N, T, self.code_dim).permute(0, 2, 1).contiguous()
            x_d_obj = x_d_obj.view(N, T, self.code_dim).permute(0, 2, 1).contiguous()
            
            # Add decoder positional embeddings
            left_pos_dec = self.left_hand_pos_emb_dec.view(1, -1, 1)
            right_pos_dec = self.right_hand_pos_emb_dec.view(1, -1, 1)
            x_d_left = x_d_left + left_pos_dec
            x_d_right = x_d_right + right_pos_dec
            
            if self.pointnet:
                cond = self.pointnet(pc)
                cfeats = cond.view(N, 1024, -1)
                if cfeats.shape[2] != x_d_left.shape[2]:
                    cfeats = cfeats.repeat([1, 1, int(x_d_left.shape[2]/cfeats.shape[2])])
                
                x_d_left = torch.cat([x_d_left, cfeats], dim=1)
                x_d_right = torch.cat([x_d_right, cfeats], dim=1)
                x_d_obj = torch.cat([x_d_obj, cfeats], dim=1)
            
            # Step 1: Decode object first
            x_decoder_obj = self.decoder_obj(x_d_obj)
            
            # Step 2: Re-encode the decoded object motion
            x_obj_reencoded = self.encoder_obj(x_decoder_obj)
            
            # Step 3: Concatenate object features to hand features
            # Ensure temporal dimensions match
            if x_obj_reencoded.shape[2] != x_d_left.shape[2]:
                x_obj_reencoded = torch.nn.functional.interpolate(
                    x_obj_reencoded, 
                    size=x_d_left.shape[2], 
                    mode='linear', 
                    align_corners=False
                )
            
            # Remove pointnet features temporarily to add object features
            if self.pointnet:
                x_d_left_no_pc = x_d_left[:, :-1024, :]
                x_d_right_no_pc = x_d_right[:, :-1024, :]
            else:
                x_d_left_no_pc = x_d_left
                x_d_right_no_pc = x_d_right
            
            x_d_left_with_obj = torch.cat([x_d_left_no_pc, x_obj_reencoded], dim=1)
            x_d_right_with_obj = torch.cat([x_d_right_no_pc, x_obj_reencoded], dim=1)
            
            # Step 4: Decode hands with object context
            x_decoder_left = self.decoder_hand(x_d_left_with_obj)
            x_decoder_right = self.decoder_hand(x_d_right_with_obj)
            
            # Concatenate decoded HOI
            x_decoder = torch.cat([x_decoder_left, x_decoder_right, x_decoder_obj], dim=1)
            x_out = self.postprocess(x_decoder)
            return x_out
            
        else:
            # Original decode logic
            # if self.dual_decoder:
            #     lz = (z/1000).int()
            #     rz = (z%1000).int()
            #     z = torch.cat([lz,rz])
            
            x_d = self.quantizer.dequantize(z)
            x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
            N = x_d.shape[0]
            if self.pointnet:
                cond = self.pointnet(pc)
                cfeats = cond.view(N, 1024, -1)
                if cfeats.shape[2] != x_d.shape[2]:
                    try:
                        cfeats = cfeats.repeat([1, 1, int(x_d.shape[2]/cfeats.shape[2])])
                    except:
                        import pdb; pdb.set_trace()
                x_d = torch.cat([x_d, cfeats], dim=1)

            # decoder
            x_decoder = self.decoder(x_d)
            x_out = self.postprocess(x_decoder)
            return x_out


class Encoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), nn.Upsample(scale_factor=2,
                                                 mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)