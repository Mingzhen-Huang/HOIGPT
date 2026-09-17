import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()
        
    def reset_codebook(self):
        if 'codebook' not in self._buffers:
            self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim))
            self.register_buffer('code_sum', torch.zeros(self.nb_code, self.code_dim))
            self.register_buffer('code_count', torch.zeros(self.nb_code))
            self.register_buffer('initialized', torch.tensor(False))
        else:
            self.codebook.zero_()
            self.code_sum.zero_()
            self.code_count.zero_()
            self.initialized.fill_(False)

    @property
    def init(self):
        return bool(self.initialized.item())

    @init.setter
    def init(self, value):
        self.initialized.fill_(value)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Original checkpoints stored only the codebook. Do not reinitialize
        # that trained codebook on the first resumed training batch.
        key = prefix + 'codebook'
        if key in state_dict:
            codebook = state_dict[key]
            state_dict.setdefault(prefix + 'code_sum', codebook.clone())
            state_dict.setdefault(prefix + 'code_count', codebook.new_ones(self.nb_code))
            state_dict.setdefault(prefix + 'initialized', codebook.new_tensor(True, dtype=torch.bool))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    @torch.no_grad()
    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook.copy_(out[:self.nb_code])
        self.code_sum.copy_(self.codebook)
        self.code_count.fill_(1.0)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

            
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity

class LegacyQuantizeHOI(QuantizeEMAReset):
    def __init__(self, nb_code, code_dim, mu):
        super().__init__(nb_code, code_dim, mu)

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
            
        return perplexity
    
    # def quantize(self, x):
    #     # Calculate latent code x_l
    #     k_w = self.codebook.t()
    #     # xl, xr = x[:,:512], x[:,512:]
           
    #     try:
    #         l_distance = torch.sum(xl ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(xl, k_w) + torch.sum(k_w ** 2, dim=0,
    #                                                                                         keepdim=True)  # (N * L, b)
    #     except:
    #         import pdb; pdb.set_trace()                                        
    #     _, lcode_idx = torch.min(l_distance, dim=-1)

    #     r_distance = torch.sum(xr ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(xr, k_w) + torch.sum(k_w ** 2, dim=0,
    #                                                                                         keepdim=True)  # (N * L, b)
    #     _, rcode_idx = torch.min(r_distance, dim=-1)

    #     # import pdb; pdb.set_trace()
    #     return torch.cat([lcode_idx,rcode_idx ])

    def quantize(self, x):
        # Calculate latent code x_l
        if x.shape[-1] == 1024:
            xl, xr = x[:,:512], x[:,512:]
            x = torch.cat([xl,xr])
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        bs = int(code_idx.shape[0] / 2)
        x = F.embedding(code_idx, self.codebook)
        x = x[bs:] + x[:bs]
        # x = torch.cat([x[bs:], x[:bs]], axis = 1)
        return x

    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        
        x = self.preprocess(x)
        ori_x = x.clone()
        # import pdb; pdb.set_trace()
        # x = torch.cat([x[:,:self.nb_code], x[:,self.nb_code:]])

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # import pdb; pdb.set_trace()
        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        perplexity = 0
        # for code_id in code_idx:
        if self.training:
            perplexity += self.update_codebook(x, code_idx)
        else : 
            perplexity += self.compute_perplexity(code_idx)
        
        # import pdb; pdb.set_trace()
        ori_x = ori_x[int(x.shape[0]/2):]
        # Loss
        try:
            commit_loss = F.mse_loss(ori_x, x_d.detach())
        except RuntimeError as exc:
            raise ValueError('Invalid legacy quantizer feature shape') from exc

        # Passthrough
        # import pdb; pdb.set_trace()
        x_d = ori_x + (x_d - ori_x).detach()

        # Postprocess
        x_d = x_d.view(int(N/2), T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        # import pdb; pdb.set_trace()  
        return x_d, commit_loss, perplexity
    
class QuantizeDual(LegacyQuantizeHOI):
    def __init__(self, nb_code, code_dim, mu):
        super().__init__(nb_code, code_dim, mu)

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        


        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity


class QuantizeHOI(nn.Module):
    def __init__(self, nb_code, code_dim, mu):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.quantizer_hands = QuantizeEMAReset(nb_code, code_dim, mu)
        self.quantizer_obj = QuantizeEMAReset(nb_code, code_dim, mu)

    def quantize(self, x_left, x_right, x_obj):
        x_left = self.quantizer_hands.preprocess(x_left)
        x_right = self.quantizer_hands.preprocess(x_right)
        x_obj = self.quantizer_obj.preprocess(x_obj)

        code_idx_left = self.quantizer_hands.quantize(x_left)
        code_idx_right = self.quantizer_hands.quantize(x_right)
        code_idx_obj = self.quantizer_obj.quantize(x_obj)
        return code_idx_left, code_idx_right, code_idx_obj

    def dequantize(self, code_idx_left, code_idx_right, code_idx_obj):
        x_left = self.quantizer_hands.dequantize(code_idx_left)
        x_right = self.quantizer_hands.dequantize(code_idx_right)
        x_obj = self.quantizer_obj.dequantize(code_idx_obj)
        return x_left, x_right, x_obj

    def forward(self, x_left, x_right, x_obj):
        n_batch, _, n_steps = x_left.shape

        x_left_flat = self.quantizer_hands.preprocess(x_left)
        x_right_flat = self.quantizer_hands.preprocess(x_right)
        x_obj_flat = self.quantizer_obj.preprocess(x_obj)

        if self.training and not self.quantizer_hands.init:
            self.quantizer_hands.init_codebook(torch.cat([x_left_flat, x_right_flat], dim=0))
        if self.training and not self.quantizer_obj.init:
            self.quantizer_obj.init_codebook(x_obj_flat)

        code_idx_left = self.quantizer_hands.quantize(x_left_flat)
        code_idx_right = self.quantizer_hands.quantize(x_right_flat)
        code_idx_obj = self.quantizer_obj.quantize(x_obj_flat)

        x_left_q = self.quantizer_hands.dequantize(code_idx_left)
        x_right_q = self.quantizer_hands.dequantize(code_idx_right)
        x_obj_q = self.quantizer_obj.dequantize(code_idx_obj)

        if self.training:
            perplexity_left = self.quantizer_hands.update_codebook(x_left_flat, code_idx_left)
            perplexity_right = self.quantizer_hands.update_codebook(x_right_flat, code_idx_right)
            perplexity_obj = self.quantizer_obj.update_codebook(x_obj_flat, code_idx_obj)
        else:
            perplexity_left = self.quantizer_hands.compute_perplexity(code_idx_left)
            perplexity_right = self.quantizer_hands.compute_perplexity(code_idx_right)
            perplexity_obj = self.quantizer_obj.compute_perplexity(code_idx_obj)

        commit_loss_left = F.mse_loss(x_left_flat, x_left_q.detach())
        commit_loss_right = F.mse_loss(x_right_flat, x_right_q.detach())
        commit_loss_obj = F.mse_loss(x_obj_flat, x_obj_q.detach())

        x_left_q = x_left_flat + (x_left_q - x_left_flat).detach()
        x_right_q = x_right_flat + (x_right_q - x_right_flat).detach()
        x_obj_q = x_obj_flat + (x_obj_q - x_obj_flat).detach()

        x_left_q = x_left_q.view(n_batch, n_steps, -1).permute(0, 2, 1).contiguous()
        x_right_q = x_right_q.view(n_batch, n_steps, -1).permute(0, 2, 1).contiguous()
        x_obj_q = x_obj_q.view(n_batch, n_steps, -1).permute(0, 2, 1).contiguous()

        commit_loss = (commit_loss_left + commit_loss_right + commit_loss_obj) / 3.0
        perplexity = (perplexity_left + perplexity_right + perplexity_obj) / 3.0
        return x_left_q, x_right_q, x_obj_q, commit_loss, perplexity


class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        
        N, width, T = z.shape
        z = self.preprocess(z)
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q - z.detach())**2) + self.beta * \
               torch.mean((z_q.detach() - z)**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        z_q = z_q.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))
        return z_q, loss, perplexity

    def quantize(self, z):

        assert z.shape[-1] == self.e_dim

        # B x V
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def dequantize(self, indices):

        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim, )).contiguous()
        return z_q

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x



class QuantizeReset(nn.Module):
    def __init__(self, nb_code, code_dim):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.reset_codebook()
        self.codebook = nn.Parameter(torch.randn(nb_code, code_dim))
        
    def reset_codebook(self):
        self.init = False
        self.code_count = None

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = nn.Parameter(out[:self.nb_code])
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_count = code_count  # nb_code
        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()

        self.codebook.data = usage * self.codebook.data + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

            
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        N, width, T = x.shape
        # Preprocess
        x = self.preprocess(x)
        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)
        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)
        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity

    
class QuantizeEMA(nn.Module):
    def __init__(self, nb_code, code_dim, mu):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = code_update
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
            
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity
