from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained
from .clip_model import build_SIGLIP2_from_openai_pretrained, convert_weights
# import match
import numpy as np
import torch
import torch.nn as nn
# import clip
from collections import OrderedDict
from .bert import Bert
from .clip import VisionTransformer_clip
from . import clip
from .CRLoss import CRLoss
# from .Loss import Loss
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
from config.base_config import Config
import math
# from . import open_clip # training 
from . import open_clip
from .cve import ContrastiveVisualEnhancer
from .dcc import DiscriminativeConceptCalibrator

# from test_nltk import texts_nltk_clip
# print(clip.__file__)

# def weights_init_kaiming(m):
#     classname = m.__class__.__name__
#     # print(classname)
#     if classname.find('Conv') != -1:
#         init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#     elif classname.find('Linear') != -1:
#         init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
#         init.constant_(m.bias.data, 0.0)
#     elif classname.find('BatchNorm1d') != -1:
#         init.normal_(m.weight.data, 1.0, 0.02)
#         init.constant_(m.bias.data, 0.0)

# def weights_init_classifier(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         init.normal_(m.weight.data, std=0.001)
#         init.constant_(m.bias.data, 0.0)

# class ChannelCompress(nn.Module):
#     def __init__(self, in_ch=2048, out_ch=256):
#         """
#         reduce the amount of channels to prevent final embeddings overwhelming shallow feature maps
#         out_ch could be 512, 256, 128
#         """
#         super(ChannelCompress, self).__init__()
#         num_bottleneck = 1000
#         add_block = []
#         add_block += [nn.Linear(in_ch, num_bottleneck)]
#         add_block += [nn.BatchNorm1d(num_bottleneck)]
#         add_block += [nn.ReLU()]

#         add_block += [nn.Linear(num_bottleneck, 500)]
#         add_block += [nn.BatchNorm1d(500)]
#         add_block += [nn.ReLU()]
#         add_block += [nn.Linear(500, out_ch)]

#         # Extra BN layer, need to be removed
#         #add_block += [nn.BatchNorm1d(out_ch)]

#         add_block = nn.Sequential(*add_block)
#         add_block.apply(weights_init_kaiming)
#         self.model = add_block

#     def forward(self, x):
#         x=x.to(torch.float16)
#         x = self.model(x)
#         return x
    
# class MultiViewMatching(nn.Module):
#     def __init__(self, ):
#         super(MultiViewMatching, self).__init__()

#     def forward(self, imgs, caps):
#         # caps -- (num_caps, dim), imgs -- (num_imgs, r, dim)
#         num_caps  = caps.size(0)
#         num_imgs, r = imgs.size()[:2]
        
#         if num_caps == num_imgs:
#             scores = torch.matmul(imgs, caps.transpose(-2, -1))  # (num_imgs, r, num_caps)

#             # scores = torch.matmul(imgs, caps.t()) #(num_imgs, r, num_caps)
#             scores = scores.max(1)[0]  #(num_imgs, num_caps)
#         else:   
#             scores = []
#             score_ids = []
#             for i in range(num_caps):
#                 cur_cap = caps[i].unsqueeze(0).unsqueeze(0)  #(1, 1, dim)
#                 cur_cap = cur_cap.expand(num_imgs, -1, -1)   #(num_imgs, 1, dim)
#                 cur_score = torch.matmul(cur_cap, imgs.transpose(-2, -1)).squeeze()    #(num_imgs, r)
#                 cur_score = cur_score.max(1, keepdim=True)[0]   #(num_imgs, 1)
#                 scores.append(cur_score)
#             scores = torch.cat(scores, dim=1)   #(num_imgs, num_caps)

#         return scores

class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_0 = LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor, pad_mask: torch.Tensor = None):
        ##### self
        x0 = self.ln_0(x)
        x0_ = self.self_attn(x0, x0, x0)[0]
        x = x + x0_

        ##### cross
        x_ = self.attn(query = self.ln_1(x),
                       key = y, 
                       value = y,
                       key_padding_mask = pad_mask)[0]
        x = x + x_
        x = x + self.mlp(self.ln_2(x))
        return x

class Attention_ast(nn.Module):
    def __init__(self, code_len):
        super(Attention_ast, self).__init__()
        # self.weight = torch.rand(100, 1)
        self.fc = nn.Linear(512, 1)
    def forward(self, x):
        x2 = x
        x2 = x2.half()
        x2 = self.fc(x2)
        x2 = torch.sigmoid(x2)
        return x2

class ResidualSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x_ = self.attention(self.ln_1(x))
        x = x + x_
        x = x + self.mlp(self.ln_2(x))
        return x

class ResidualAttention(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 n_head,
                 att_type = 'cross',
                 out_norm = None):
        super().__init__()
        self.att_type = att_type
        if self.att_type == 'self':
            ResidualAttentionBlock = ResidualSelfAttentionBlock(d_model=d_model, n_head=n_head)
        elif self.att_type == 'cross':
            ResidualAttentionBlock = ResidualCrossAttentionBlock(d_model=d_model, n_head=n_head)
        self.layers = nn.ModuleList([
            ResidualAttentionBlock for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = out_norm

    def forward(self, x, y=None, pad_mask=None):
        '''
            x: b, Lx, dx
            y: b, Ly, dy
            pad_mask: b, Ly
        '''
        x = x.permute(1, 0, 2)
        if self.att_type == 'cross':
            y = y.permute(1, 0, 2)
        output = x
        for layer in self.layers:
            if self.att_type == 'cross':
                output = layer(output, y, pad_mask=pad_mask)
            else:
                output = layer(output)

        if self.norm is not None:
            # Lx, b, dx -> b, Lx, dx
            output = self.norm(output).permute(1, 0, 2)
        else:
            return output.permute(1, 0, 2)
        return output


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


# class ViT_CLIP_custom(nn.Module):
#     def __init__(self, model_name='ViT-B/16', device=None):
#         super(ViT_CLIP_custom, self).__init__()
#         self.device = device
#         clip_model, _ = clip.load(model_name, device=self.device)
#         state_dict = clip_model.visual.state_dict()
#         vision_width = state_dict["conv1.weight"].shape[0]
#         vision_layers = len(
#             [k for k in state_dict.keys() if k.startswith("transformer.") and k.endswith(".attn.in_proj_weight")])
#         vision_patch_size = state_dict["conv1.weight"].shape[-1]
#         grid_size = round((state_dict["positional_embedding"].shape[0] - 1) ** 0.5)
#         image_resolution = vision_patch_size * grid_size
#         vision_heads = vision_width // 64
#         self.clip_model = VisionTransformer_clip(input_resolution=image_resolution,
#                                             patch_size=vision_patch_size,
#                                             width=vision_width,
#                                             layers=vision_layers,
#                                             heads=vision_heads)
#         if "proj" in state_dict:
#             del state_dict["proj"]
#         self.clip_model.load_state_dict(state_dict)

#     def forward(self, x):

#         x, att_score = self.clip_model(x)
#         return x, att_score

# class InterCorrelationReasoning(nn.Module):
#     """
#     Perform the inter-correlation reasoning with a full-connected graph
#     Args: - sim_emb: intra-correlation vector, shape: (batch_size, max_turns + 1, embed_size)
#     Returns; - sim_icr: inter-correlation reasoned graph nodes, shape: (batch_size, max_turns + 1, embed_size)
#     """
#     def __init__(self, sim_dim):
#         super(InterCorrelationReasoning, self).__init__()

#         self.graph_query_w = nn.Linear(sim_dim, sim_dim)
#         self.graph_key_w = nn.Linear(sim_dim, sim_dim)
#         self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
#         self.relu = nn.ReLU()

#         self.init_weights()

#     def forward(self, sim_emb):
#         sim_query = self.graph_query_w(sim_emb)                                             
#         sim_key = self.graph_key_w(sim_emb)                                                 
#         sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)    
#         sim_icr = torch.bmm(sim_edge, sim_emb)                                              
#         sim_icr = self.relu(self.sim_graph_w(sim_icr))
#         return sim_icr

#     def init_weights(self):
#         for m in self.children():
#             if isinstance(m, nn.Linear):
#                 r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
#                 m.weight.data.uniform_(-r, r)
#                 m.bias.data.fill_(0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(val, divisor):
    return (val % divisor) == 0

def unfold_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

# def l2norm(X, dim, eps=1e-8):
#     """L2-normalize columns of X
#     """
#     norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
#     X = torch.div(X, norm)
#     return X

# class VGMF_Fusion(nn.Module):
#     def __init__(self, opt = {}):
#         super(VGMF_Fusion, self).__init__()
#         self.gate = nn.Linear(1024, 512)

#     def forward(self, sv, kv):
#         # l2 norm
#         sv = l2norm(sv, dim=-1)
#         kv = l2norm(kv, dim=-1)
#         # print(sv.shape)
#         # print(kv.shape)

#         # concat fc
#         sw_s = F.sigmoid(self.gate(torch.cat([sv, kv], dim=-1)))
#         ones = torch.ones(sw_s.shape).cuda()
#         sw_k = ones - sw_s

#         out = sw_s*sv + sw_k*kv
#         return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x.to(torch.float32)), **kwargs)

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads =  heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        _, _, _, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x.to(torch.float16)).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x.to(torch.float16))

# class TNT(nn.Module):
#     def __init__(
#         self,
#         *,
#         image_size=256,
#         patch_dim=512,
#         pixel_dim=24,
#         patch_size=16,
#         pixel_size=4,
#         # depth=6,
#         # depth=3,  # 爆显存了
#         depth=2,
#         num_classes=1000,
#         channels = 3,
#         heads = 8,
#         dim_head = 64,
#         ff_dropout = 0.1,
#         attn_dropout = 0.1,
#         unfold_args = None
#     ):
#         super().__init__()
#         assert divisible_by(image_size, patch_size), 'image size must be divisible by patch size'
#         assert divisible_by(patch_size, pixel_size), 'patch size must be divisible by pixel size for now'

#         num_patch_tokens = (image_size // patch_size) ** 2

#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.patch_tokens = nn.Parameter(torch.randn(num_patch_tokens + 1, patch_dim))

#         unfold_args = default(unfold_args, (pixel_size, pixel_size, 0))
#         unfold_args = (*unfold_args, 0) if len(unfold_args) == 2 else unfold_args
#         kernel_size, stride, padding = unfold_args

#         pixel_width = unfold_output_size(patch_size, kernel_size, stride, padding)
#         num_pixels = pixel_width ** 2

#         self.to_pixel_tokens = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1 = patch_size, p2 = patch_size),
#             nn.Unfold(kernel_size = kernel_size, stride = stride, padding = padding),
#             Rearrange('... c n -> ... n c'),
#             nn.Linear(channels * kernel_size ** 2, pixel_dim)
#         )

#         # self.patch_pos_emb = nn.Parameter(torch.randn(num_patch_tokens + 1, patch_dim))
#         # self.pixel_pos_emb = nn.Parameter(torch.randn(num_pixels, pixel_dim))

#         layers = nn.ModuleList([])
#         for _ in range(depth):

#             pixel_to_patch = nn.Sequential(
#                 nn.LayerNorm(pixel_dim),
#                 Rearrange('... n d -> ... (n d)'),
#                 nn.Linear(pixel_dim * num_pixels, patch_dim),
#             )

#             layers.append(nn.ModuleList([
#                 PreNorm(pixel_dim, Attention(dim = pixel_dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
#                 PreNorm(pixel_dim, FeedForward(dim = pixel_dim, dropout = ff_dropout)),
#                 pixel_to_patch,
#                 PreNorm(patch_dim, Attention(dim = patch_dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
#                 PreNorm(patch_dim, FeedForward(dim = patch_dim, dropout = ff_dropout)),
#             ]))

#         self.layers = layers

#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(patch_dim),
#             nn.Linear(patch_dim, num_classes)
#         )

#     def forward(self, x):
#         b, _, h, w, patch_size, image_size = *x.shape, self.patch_size, self.image_size
#         assert divisible_by(h, patch_size) and divisible_by(w, patch_size), f'height {h} and width {w} of input must be divisible by the patch size'

#         num_patches_h = h // patch_size
#         num_patches_w = w // patch_size
#         n = num_patches_w * num_patches_h

#         pixels = self.to_pixel_tokens(x)        

#         patches = repeat(self.patch_tokens[:(n + 1)], 'n d -> b n d', b = b)

#         patches += rearrange(self.patch_pos_emb[:(n + 1)], 'n d -> () n d')
#         pixels += rearrange(self.pixel_pos_emb, 'n d -> () n d')

#         for pixel_attn, pixel_ff, pixel_to_patch_residual, patch_attn, patch_ff in self.layers:

#             pixels = pixel_attn(pixels) + pixels
#             pixels = pixel_ff(pixels) + pixels

#             patches_residual = pixel_to_patch_residual(pixels)

#             patches_residual = rearrange(patches_residual, '(b h w) d -> b (h w) d', h = num_patches_h, w = num_patches_w)
#             patches_residual = F.pad(patches_residual, (0, 0, 1, 0), value = 0) # cls token gets residual of 0
#             patches = patches + patches_residual   #将视觉词的特征聚合成patch级别的特征,local

#             patches = patch_attn(patches) + patches
#             patches = patch_ff(patches) + patches  #计算图像块之间的注意力，global

#         cls_token = patches[:, 0]  #全局
#         return pixels, patches, patches_residual

# class MultiHeaded(nn.Module):
#     def __init__(self, config: Config):
#         super(MultiHeaded, self).__init__()
#         self.embed_dim = 512
#         self.num_heads = 1
#         assert self.embed_dim % self.num_heads == 0
#         self.head_dim = self.embed_dim // self.num_heads
        
#         self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
#         self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
#         self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
#         self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
#     def forward(self, text_embeds, video_embeds):
#         """
#         Input
#             text_embeds: num_texts x embed_dim
#             video_embeds: num_vids x num_frames x embed_dim
#         Output
#             o: num_vids x num_texts x embed_dim
#         """
#         # print("text_embeds:",text_embeds.shape) # text_embeds: torch.Size([64, 512])   text_embeds: torch.Size([64, 257, 512])
#         # print("video_embeds:",video_embeds.shape) #video_embeds: torch.Size([64, 257, 512])  video_embeds: torch.Size([64, 512])
#         #cccccccc
#         # num_texts, _ = text_embeds.shape
#         num_texts = text_embeds.shape[0]
#         # num_texts x embed_dim

#         q = self.q_proj(text_embeds)
#         # print("q",q.shape) #q torch.Size([64, 77, 512])
#         # print(num_texts, self.num_heads, self.head_dim)  64 1 512
#         # q = q.reshape(num_texts, self.num_heads, self.head_dim)
#         # print(f"Original shape of q: {q.size()}")
  
#         q = q.reshape(num_texts, 1, self.head_dim)
#         # print(q.shape)

#         # print(f"Reshaped shape of q: {q.size()}")
#         # num_heads x head_dim x num_texts
#         q = q.permute(1,2,0)

#         num_vids, num_frames, _ = video_embeds.shape
#         # num_vids x num_frames x embed_dim
#         k = self.k_proj(video_embeds)
#         k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
#         # print(k.shape)
#         # num_vids x num_heads x num_frames x head_dim
#         k = k.permute(0,2,1,3)

#         # num_vids x num_frames x embed_dim
#         v = self.v_proj(video_embeds)
#         v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
#         # num_vids x num_heads x head_dim x num_frames
#         v = v.permute(0,2,3,1)

#         # num_vids x num_heads x num_frames x num_texts
#         # print(k.shape)
#         # print(q.shape)
#         attention_logits = k @ q
#         attention_logits = attention_logits / math.sqrt(self.head_dim)
#         attention_weights = F.softmax(attention_logits, dim=2)

#         # num_vids x num_heads x head_dim x num_texts
#         attention = v @ attention_weights
#         # num_vids x num_texts x num_heads x head_dim
#         attention = attention.permute(0,3,1,2)

#         # attention = attention.reshape(num_vids, num_texts, self.embed_dim)
#         attention = attention.view(num_vids, num_texts, -1)

#         # num_vids x num_texts x embed_dim
#         o = self.out_proj(attention)
#         # print(o.shape)
#         return o

# def sim_matrix_training(text_embeds, vid_embeds_pooled, pooling_type):
#     """
#     Computes the similarity matrix using pooled video frames
    
#     Output
#         sims: num_texts x num_vids
#     """
#     text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
#     vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

#     if pooling_type == 'avg':
#         sims = torch.mm(text_embeds, vid_embeds_pooled.t())
        
#     else:
#         # num_texts x embed_dim x num_vids
#         vid_embeds_pooled = vid_embeds_pooled.permute(1,2,0)
#         # num_texts x 1 x embed_dim
#         text_embeds = text_embeds.unsqueeze(1)
#         # print(text_embeds.shape) torch.Size([64, 1, 512])
#         # print(vid_embeds_pooled.shape)         torch.Size([64, 512, 64]) 
#         sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)
#     # print(sims.shape) torch.Size([64, 64])

#     return sims


# def sim_matrix_training2(text_embeds, vid_embeds_pooled, pooling_type):
#     """
#     Computes the similarity matrix using pooled video frames
    
#     Output
#         sims: num_texts x num_vids
#     """
#     text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
#     vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

#     if pooling_type == 'avg':
#         sims = torch.mm(text_embeds, vid_embeds_pooled.t())
        
#     else:
#         # num_texts x embed_dim x num_vids
#         # vid_embeds_pooled = vid_embeds_pooled.permute(1,2,0)
#         # # num_texts x 1 x embed_dim
#         # text_embeds = text_embeds.unsqueeze(1)
        
#         # sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)
#         # text_embeds = text_embeds.unsqueeze(1)
#         # num_texts x embed_dim x num_vids
#         vid_embeds_pooled = vid_embeds_pooled.permute(1, 2, 0)
#                 # sims = torch.matmul(text_embeds, vid_embeds_pooled).squeeze(1)
#         # 批次矩阵乘法
#         # print(text_embeds.shape) #torch.Size([64, 1, 257, 512])
#         # print(vid_embeds_pooled.shape) #torch.Size([64, 512, 64])
#         # sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze()
#         # sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze().squeeze()
#         sims = torch.bmm(text_embeds, vid_embeds_pooled).view(text_embeds.size(0), -1)
#         sims =sims[:,:64]
#     return sims

class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class IRRA(nn.Module):
    def __init__(self, args, config: Config, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        self.global_loss = objectives.Global_Loss()
        # self.EntLoss = objectives.EntLoss(args)
        # self.EntLoss2 = objectives.EntLoss2()


        # self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size) # CLIP
        # self.embed_dim = base_cfg['embed_dim']
        # # self.embed_dim = 512

        # 调用 siglip 模型
        self.base_model, _, _ = open_clip.create_model_and_transforms('local-dir:/home/hpc/LAB-data/disk-4T/syc_data/irra-v1/irra/IRRA-main_v2/ViT-L-16-SigLIP2-256')
        self.embed_dim = 1024
        self.base_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
        self.similarityNorm = nn.Softmax(dim=2)

        self.block = ResidualAttention(num_layers=1,
                                    #    d_model=512,
                                       d_model=1024,
                                       n_head=16,
                                       att_type='cross')

        self.cr_loss_fun = CRLoss()
        # self.compute_loss = Loss()

        # 原
        # self.bottleneck_image = nn.BatchNorm1d(512)
        # self.bottleneck_image.bias.requires_grad_(False)
        # self.bottleneck_image.apply(weights_init_kaiming)
        # self.bottleneck_text = nn.BatchNorm1d(512)
        # self.bottleneck_text.bias.requires_grad_(False)
        # self.bottleneck_text.apply(weights_init_kaiming)
        # self.attention = Attention_ast(code_len=512)

        # VIT-B
        # self.bottleneck_image = nn.BatchNorm1d(768)
        # self.bottleneck_image.bias.requires_grad_(False)
        # self.bottleneck_image.apply(weights_init_kaiming)
        # self.bottleneck_text = nn.BatchNorm1d(768)
        # self.bottleneck_text.bias.requires_grad_(False)
        # self.bottleneck_text.apply(weights_init_kaiming)
        # self.attention = Attention_ast(code_len=512)

        # VIT-L
        self.bottleneck_image = nn.BatchNorm1d(1024)
        self.bottleneck_image.bias.requires_grad_(False)
        self.bottleneck_image.apply(weights_init_kaiming)
        self.bottleneck_text = nn.BatchNorm1d(1024)
        self.bottleneck_text.bias.requires_grad_(False)
        self.bottleneck_text.apply(weights_init_kaiming)
        self.attention = Attention_ast(code_len=512)

        # self.TripletLoss = objectives.TripletLoss()
        # self.mvm = MultiViewMatching()

        # self.kl_div = torch.nn.KLDivLoss()
        self.softmax = torch.nn.Softmax(dim=1)
        # self.sim_tranloc_w = nn.Linear(512, 64)
        # self.sim_tranglo_w = nn.Linear(512, 64)
        # self.sim_tranglo_w = nn.Linear(512, 512)
        # self.sim_w = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        # self.ICR_module = nn.ModuleList([InterCorrelationReasoning(64) for i in range(3)])
        image_size =256
        patch_size =16
        unfold_args=None
        channels=3
        pixel_dim=24
        pixel_size=4
        unfold_args = default(unfold_args, (pixel_size, pixel_size, 0))
        unfold_args = (*unfold_args, 0) if len(unfold_args) == 2 else unfold_args
        kernel_size, stride, padding = unfold_args
        pixel_width = unfold_output_size(patch_size, kernel_size, stride, padding)
        num_pixels = pixel_width ** 2
        patch_dim=512

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.tnt_model = TNT().to(device).to(torch.float32)
        # self.tnt_model = TNT().to(device)
        
        
        
        num_patch_tokens = (image_size // patch_size) ** 2
        self.to_pixel_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1 = patch_size, p2 = patch_size),
            nn.Unfold(kernel_size = kernel_size, stride = stride, padding = padding),
            Rearrange('... c n -> ... n c'),
            nn.Linear(channels * kernel_size ** 2, pixel_dim)
        )

        self.patch_pos_emb = nn.Parameter(torch.randn(num_patch_tokens + 1, patch_dim))
        self.pixel_pos_emb = nn.Parameter(torch.randn(num_pixels, pixel_dim))
        self.patch_tokens = nn.Parameter(torch.randn(num_patch_tokens + 1, patch_dim))

        
        layers = nn.ModuleList([])
        for _ in range(3):
            pixel_to_patch = nn.Sequential(
                nn.LayerNorm(pixel_dim),
                Rearrange('... n d -> ... (n d)'),
                nn.Linear(pixel_dim * num_pixels, patch_dim),
            )

            layers.append(nn.ModuleList([
                PreNorm(pixel_dim, Attention(dim = pixel_dim, heads = 8, dim_head = 64, dropout = 0.1)),
                PreNorm(pixel_dim, FeedForward(dim = pixel_dim, dropout = 0.1)),
                pixel_to_patch,
                PreNorm(patch_dim, Attention(dim = patch_dim, heads = 8, dim_head = 64, dropout = 0.1)),
                PreNorm(patch_dim, FeedForward(dim = patch_dim, dropout = 0.1)),
            ]))
        # self.vgmf_gate = VGMF_Fusion()
        self.layers = layers
        self.layer_norm1 = nn.LayerNorm(self.embed_dim).half()
        self.layer_norm2 = nn.LayerNorm(self.embed_dim).half()
        self.layer_norm3 = nn.LayerNorm(self.embed_dim).half()
        self.layer_norm4 = nn.LayerNorm(self.embed_dim).half()
        self.logit_scale1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.09))  #可调
        dropout = 0.3 #可调 0.3
        self.dropout = nn.Dropout(dropout)
        # self.poolattn = MultiHeaded(config)
        # self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.loss_fct = CrossEn()
         # for coarse-grained constrast weights
        # self.global_mat_weight = nn.parameter.Parameter(torch.eye(self.embed_dim), requires_grad=True)
        # self.global_mat_weight_1 = nn.parameter.Parameter(torch.eye(self.embed_dim), requires_grad=True)

        # # for cross-grained constrast weights
        # self.word_logit_weight = nn.parameter.Parameter(torch.eye(20), requires_grad=True)
        # self.frame_logit_weight = nn.parameter.Parameter(torch.eye(25), requires_grad=True)        

        # # for fine-grained constrast weights
        # self.local_mat_weight = nn.parameter.Parameter(torch.eye(self.embed_dim), requires_grad=True)
        # self.local_mat_weight1 = nn.parameter.Parameter(torch.eye(self.embed_dim), requires_grad=True)
        # self.frame_mat_weight = nn.parameter.Parameter(torch.eye(100), requires_grad=True)
        # self.word_mat_weight = nn.parameter.Parameter(torch.eye(30), requires_grad=True)
        # self.frame_mat_weight2 = nn.parameter.Parameter(torch.eye(100), requires_grad=True)
        # self.word_mat_weight2 = nn.parameter.Parameter(torch.eye(30), requires_grad=True)
        # self.pixel_mat_weight = nn.parameter.Parameter(torch.eye(25), requires_grad=True)    
        # self.pixel_mat_weight2 = nn.parameter.Parameter(torch.eye(25), requires_grad=True)    

        # if 'id' in args.loss_names:
        #     self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        #     nn.init.normal_(self.classifier.weight.data, std=0.001)
        #     nn.init.constant_(self.classifier.bias.data, val=0.0)

        # Initialize CVE
        cve_config = {}
        self.cve = ContrastiveVisualEnhancer(
            embed_dim=self.embed_dim,
            output_dim=self.embed_dim,
            residual_coefficient=cve_config.get('residual_coefficient', 0.1),
            attention_percentile=cve_config.get('attention_percentile', 90.0),
        )

        # Initialize DCC
        dcc_config = {}
        self.dcc = DiscriminativeConceptCalibrator(
            embed_dim=self.embed_dim,
            lambda_weight=getattr(self.args, 'dcc_lambda_weight', 0.5),
            temperature=getattr(self.args, 'dcc_temperature', 0.01),
        )


    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def encode_image(self, image):
        # print('xxxxx')
        # print(image.shape)
        
        # #clip
        # x = self.base_model.encode_image(image)

        # return x[0][:, 0, :].float()
        # # return x.float() # for CLIP ResNet visual model

        # siglip 
        _, x, _ = self.base_model.encode_image(image)
        return x

    def encode_text(self, text):
        # print('1111',text)

        # # clip
        # x,_ = self.base_model.encode_text(text)

        # # print(x.shape)
        # return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

        # siglip 
        _, x, _ = self.base_model.encode_text(text)
        return x

    def forward(self, batch):
        ret = dict()
                
        images = batch['images']
        # print(images.shape)  torch.Size([64, 3, 256, 256])
        caption_ids = batch['caption_ids']
        labels = batch['image_ids']
        # phrase = batch['phrase_vectors']
        
        # print('images.shape',images.shape)      # ([64, 3, 256, 256])
        # print('caption_ids.shape',caption_ids.shape)    # ([64, 77])
        # print("images.dtype ", images.dtype)    # float32
        # print("caption_ids.dtype ", caption_ids.dtype)  # int64
        # print("caption_ids.shape ", caption_ids.shape)  # int64
        # print("caption_ids ", caption_ids)  # int64

        # # CLIP 生成特征
        # image_feats, image_att_scores, text_feats, text_att_scores = self.base_model(images, caption_ids)

        image_feats, img_global_init, image_att_scores = self.base_model.encode_image(images)
        cls_attention = image_att_scores[-1].mean(dim=1)
        img_global, cve_info = self.cve(
            cls_token=img_global_init,
            patch_tokens=image_feats,
            cls_attention=cls_attention,
            cls_projected=img_global_init,
        )

        text_feats, text_global_init, text_att_scores = self.base_model.encode_text(caption_ids)
        eot_attention = text_att_scores[-1].mean(dim=1)
        # token_mask create
        seq_len = caption_ids.shape[1]
        pad_id = 0  # 如果你的 pad 不是 0，把这里改掉
        is_pad = (caption_ids == pad_id)                 # (B, seq_len) bool
        has_pad = is_pad.any(dim=1)               # (B,) bool
        # 第一个 pad 的位置；若不存在 pad，argmax 会返回 0，需要修正
        first_pad_pos = is_pad.float().argmax(dim=1)  # (B,)
        # 若不存在 pad，则让 first_pad_pos = seq_len（表示整条有效）
        first_pad_pos = torch.where(
            has_pad,
            first_pad_pos,
            torch.full_like(first_pad_pos, seq_len)
        )
        # mask：pad 之前为 1，从第一个 pad 开始为 0
        token_mask = (torch.arange(seq_len, device=caption_ids.device).unsqueeze(0) < first_pad_pos.unsqueeze(1)).float()

        text_global, dcc_info = self.dcc(
            eot_features=text_global_init,
            token_features=text_feats,
            eot_attention=eot_attention,
            token_ids=caption_ids,
            token_mask=token_mask,
            corpus_token_ids=None,
        )


        # ret.update({'image_feats': image_feats})
        # ret.update({'pids': batch['pids']})
        # ret.update({'text_feats': text_feats})
        # ret.update({'labels': labels})

        # print('text_feats', text_feats.shape)       # ([64, 77, 512])
        # print('text_att_scores', len(text_att_scores))      # 12
        # print('text_att_scores[0]', text_att_scores[-1].shape)   #([64, 77, 77])
        # print('image_feats', image_feats.shape)     # ([64, 257, 512])
        # print('image_att_scores', len(image_att_scores))    # 12
        # print('image_att_scores[0]', image_att_scores[-1].shape) # ([64, 257, 257])

        # TODO CFine CLIP
        # # global
        # #图像全局特征11111
        # img_global = self.bottleneck_image(image_feats[:, 0, :])
        # # print(img_global.shape)
        # #文本全局特征111111
        # text_global = self.bottleneck_text(text_feats[:, 1, :])

        
        # #change
        # i_feats = image_feats[:, 0, :].float()
        # # print("i_feats ----------",i_feats)
        # # i_feats = image_feats.float() # for CLIP ResNet visual model
        # t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        
        i_feats = img_global
        t_feats = text_global

        image_parts = []
        text_parts = []

        # clip
        img_score1 = image_att_scores[-1][:, 0, :]
        text_score1 = text_att_scores[-1][:, 0, :]
        temp = torch.zeros(text_score1.size(0), 1)
        score_mask_img = (torch.cat((temp, torch.ones(img_score1.size(0), img_score1.size(1) - 1)), dim=1)).to(img_score1.device)
        score_mask_text = (torch.cat((temp, torch.ones(text_score1.size(0), text_score1.size(1) - 1)), dim=1)).to(text_score1.device)
        img_score = img_score1 * score_mask_img
        text_score = text_score1 * score_mask_text
        text_masks = torch.zeros_like(caption_ids).masked_fill_(caption_ids == 0, 1).bool()

        text_feats_selected1 = []
        text_mask_selected1 = []
        text_feats_selected2 = []
        text_mask_selected2 = []
        text_feats_selected_all = []
        text_K = int(text_feats.size(1) * 0.2) #analy

        for b in range(text_feats.size(0)):
            _, idx = text_score[b].topk(text_score.size(1), largest=True, sorted=True)
            neg_idx1 = idx[:text_K]
            neg_idx2 = idx[text_K:text_K*2]
            text_feats_selected1.append(text_feats[b][neg_idx1, :])
            text_mask_selected1.append(text_masks[b][neg_idx1])
            text_feats_selected2.append(text_feats[b][neg_idx2, :])
            text_mask_selected2.append(text_masks[b][neg_idx2])
            text_feats_selected_all.append(text_feats[b][idx[:text_K*2], :])
        text_feats_selected1 = torch.stack(text_feats_selected1, dim=0)
        text_mask_selected1 = torch.stack(text_mask_selected1, dim=0)
        text_feats_selected1 = torch.cat((text_feats[:, 0, :].unsqueeze(1), text_feats_selected1), dim=1)
        text_mask_selected1 = torch.cat((text_masks[:, 0].unsqueeze(1), text_mask_selected1), dim=1)
        text_feats_selected2 = torch.stack(text_feats_selected2, dim=0)
        text_mask_selected2 = torch.stack(text_mask_selected2, dim=0)
        text_feats_selected2 = torch.cat((text_feats[:, 0, :].unsqueeze(1), text_feats_selected2), dim=1)
        text_mask_selected2 = torch.cat((text_masks[:, 0].unsqueeze(1), text_mask_selected2), dim=1)
        text_feats_selected_all = torch.stack(text_feats_selected_all, dim=0)

        # ResidualAttentionCross
        text_part1 = self.block(text_feats_selected1, text_feats, text_masks)
        text_part2 = self.block(text_feats_selected2, text_feats, text_masks)

        text_parts.append(text_part1)
        text_parts.append(text_part2)

        image_feats_selected1 = []
        image_feats_selected2 = []
        image_feats_selected_all = []
    
        img_K = int(image_feats.size(1) * 0.4) # 原来0.1 analy 0.1, 0.2, 0.3, 0.4, 0.5
        # print(img_K)   25
        # img_K = 20
        for b in range(image_feats.size(0)):
            # print("img_score",img_score.size(1))
            _, idx = img_score[b].topk(img_score.size(1), largest=True, sorted=True)
            neg_idx1 = idx[:img_K]
            # print("neg_idx1",neg_idx1)
            # print(image_feats[b].shape)
            neg_idx2 = idx[img_K:img_K*2]
            image_feats_selected1.append(image_feats[b][neg_idx1, :])
            image_feats_selected2.append(image_feats[b][neg_idx2, :])
            image_feats_selected_all.append(image_feats[b][idx[:img_K*2], :])
        image_feats_selected1 = torch.stack(image_feats_selected1, dim=0)
        image_feats_selected1 = torch.cat((image_feats[:, 0, :].unsqueeze(1), image_feats_selected1), dim=1)
        image_feats_selected2 = torch.stack(image_feats_selected2, dim=0)
        image_feats_selected2 = torch.cat((image_feats[:, 0, :].unsqueeze(1), image_feats_selected2), dim=1)
        image_feats_selected_all = torch.stack(image_feats_selected_all, dim=0)

        # ResidualAttentionCross通过注意力提取图像的局部特征1和2
        image_part1 = self.block(image_feats_selected1, image_feats)
        image_part2 = self.block(image_feats_selected2, image_feats)

        image_parts.append(image_part1)
        image_parts.append(image_part2)

        # print('img_global',img_global)
        img_output = (img_global,)
        # print('9999999',img_output)
        text_output = (text_global,)

        # img_output = img_global
        # text_output = text_global
        # print('1111',img_output.shape)  
        # print('2222',text_output.shape)
        
        for j in range(len(image_parts)):
            #同模态跨粒度 tg vg 多层次特征学习模块
            img_output = img_output + (self.bottleneck_image(image_parts[j][:, 0, :]),)
            text_output = text_output + (self.bottleneck_text(text_parts[j][:, 0, :]),)
        
        # img_f = torch.stack(img_output, dim=1)
        # text_f = torch.stack(text_output, dim=1)
        # print('1111',img_f.shape)  #torch.Size([64, 3, 512])
        # print('2222',text_f.shape)  #torch.Size([64, 3, 512])

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            # print(i_feats.shape)
            # print('i_feats',i_feats)
            # ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)*3.0}) #可
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)}) #lry改
        
        if 'sdm' in self.current_task:
            # ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)*3.0}) #可
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)*1.8}) #lry改

        # if 'id' in self.current_task:
        #     image_logits = self.classifier(i_feats.half()).float()
        #     text_logits = self.classifier(t_feats.half()).float()
        #     ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])})

        #TODO cfine- Cross-Similarity       
        if 'cmpm' in self.current_task:
            # fine-grain  
            # 图像全局特征22222          
            G_img_token = image_feats[:, 0, :].unsqueeze(1)
            # print(G_img_token.shape)  torch.Size([64, 1, 512])
            # G_img_token = image_feats
            # G_img_token = i_feats2.unsqueeze(1)
            # 图像局部特征11111  
            L_img_token = image_feats_selected_all
            # B = L_img_token.size(0)
            # 文本全局特征22222 
            G_text_token = text_feats[:, 0, :].unsqueeze(1)
            # print(G_text_token.shape)  torch.Size([64, 1, 512])
            # G_text_token = text_feats
            # G_text_token = t_feats2.unsqueeze(1)
            # 文本局部特征11111 
            L_text_token = text_feats_selected_all
            # print('1',L_text_token.shape)  torch.Size([64, 30, 512])
            # print('1',L_img_token.shape)  #torch.Size([64, 50, 512]

            G_img_token_norm = G_img_token / G_img_token.norm(dim=-1, keepdim=True)
            L_img_token_norm = L_img_token / L_img_token.norm(dim=-1, keepdim=True)
            # print('2',L_img_token_norm.shape)  torch.Size([64, 50, 512]

            G_text_token_norm = G_text_token / G_text_token.norm(dim=-1, keepdim=True)
            L_text_token_norm = L_text_token / L_text_token.norm(dim=-1, keepdim=True)
            
            # ---------------------------
            # Correspondence Discovery
            # ---------------------------
            #patch-word 和 patch-text之间的相似度,特征对应关系
            L_img_token = image_feats_selected1[:, 1:, :]
            #六猪 L_img_token torch.Size([64, 25, 512])
            # print('L_img_token',L_img_token.shape)  L_img_token torch.Size([64, 25, 512])
            L_text_token = text_feats_selected1[:, 1:, :]
            # print('L_img_token ',L_img_token.shape)
            # print('L_text_token ',L_text_token.shape)
            # print('L_text_token',L_text_token.shape) L_text_token torch.Size([64, 15, 512])
            L_img_token_norm1 = L_img_token / L_img_token.norm(dim=-1, keepdim=True)
            L_text_token_norm1 = L_text_token / L_text_token.norm(dim=-1, keepdim=True)
            _b, _, _ = L_text_token.shape
            # L_text_token_norm1=L_text_token_norm1.to(torch.float16)
            L_text_token_norm1=L_text_token_norm1
            vidwordSim = torch.bmm(L_img_token_norm1, L_text_token_norm1.permute(0, 2, 1))
            vidwordSim = self.similarityNorm(vidwordSim)
            # ---------- posWord -------
            _, idxWord = vidwordSim.topk(3, dim=2, largest=True, sorted=True)
            posWord = []
            for _batch in range(idxWord.shape[0]):
                posWord.append(L_text_token[_batch, idxWord[_batch], :])
            posWord = torch.stack(posWord)
            posWord = torch.mean(posWord, dim=2)
            posWord_norm = posWord / posWord.norm(dim=-1, keepdim=True)
            # ---------- posClip -------
            _, idxVid = vidwordSim.topk(3, dim=1, largest=True, sorted=True)
            posClip = []
            for _batch in range(idxVid.shape[0]):
                posClip.append(L_img_token[_batch, idxVid[_batch], :])
            posClip = torch.stack(posClip)
            posClip = torch.mean(posClip, dim=1)
            posClip_norm = posClip / posClip.norm(dim=-1, keepdim=True)
            # posWord_norm_r 是选择后的文本局部特征
            # posClip_norm_l 是选择后的图像局部特征
            # piexl_word sim
            L_img_token_norm_l1 = L_img_token_norm1.unsqueeze(1).repeat(1, _b, 1, 1)
            posWord_norm_r = posWord_norm.unsqueeze(0).repeat(_b, 1, 1, 1)
            posClip_norm_l = posClip_norm.unsqueeze(1).repeat(1, _b, 1, 1)
            L_text_token_norm_r1 = L_text_token_norm1.unsqueeze(0).repeat(_b, 1, 1, 1)

            #compute gobal image---local text and local image---global text
            # 跨模态跨粒度  sim_cs图像局部和文本全局  图像全局和文本局部 【缺:phrase和图像全局、patch和文本全局】
            sim_cs1 = objectives.compute_sim_cs(G_img_token_norm, L_text_token_norm, L_img_token_norm, G_text_token_norm)
             
            # print("sram ==================")
            # print("sram ", sram.shape)
            # sim_cs2 = objectives.compute_sim_cs (G_img_token_norm, L_text_token_norm, patches, G_text_token_norm)
            # sim_cs = sim_cs1 + sim_cs2
            
            # sim = objectives.compute_cmpm(G_img_token_norm, L_text_token_norm, L_img_token_norm, G_text_token_norm, L_img_token_norm_l1, posWord_norm_r, posClip_norm_l, L_text_token_norm_r1)
            # print('1xxxxxxxxxxx',sim.shape)  64,64
            sim_loss = self.cr_loss_fun(sim_cs1, labels, semi=False)
            # sim_loss = self.cr_loss_fun(sim, labels, semi=False)
            # ret.update({'cmpm_loss':sim_loss*0.4})
            ret.update({'cmpm_loss':sim_loss})

        if 'cfine' in self.current_task:
            #跨模态同粒度
            #句子-global级别 / word-sub-patch是sim_cd[有]  跨模态同粒度  细粒度对应关系模块 + 短语-patch  [缺:段落和patch]
            # print('G_img_token_norm.shape ',G_img_token_norm.shape) # [64, 1, 512]
            # print('G_text_token_norm.shape ',G_text_token_norm.shape) # [64, 1, 512]
            sim_global = objectives.compute_cmp(G_img_token_norm , G_text_token_norm)
            # print('L_img_token_norm_l1.shape ',L_img_token_norm_l1.shape) # [64, 64, 25, 512]
            # print('posWord_norm_r.shape ',posWord_norm_r.shape) # [64, 64, 25, 512]
            # print('posClip_norm_l.shape ',posClip_norm_l.shape) # [64, 64, 15, 512]
            # print('L_text_token_norm_r1.shape ',L_text_token_norm_r1.shape) # [64, 64, 15, 512]
            sim_cd = objectives.compute_sim_cd (L_img_token_norm_l1, posWord_norm_r, posClip_norm_l, L_text_token_norm_r1)
            # print('sim_cd ',sim_cd) 
            # print('sim_globa ',sim_global) 
            # print('sim_cd.shape ',sim_cd.shape) # [64, 64]
            # print('sim_globa.shape ',sim_global.shape) # [64, 64]
            # print('11',L_img_token.shape)  #64,25,512

            # 切片
            # L_img_token_output = L_img_token[:,:1,:]
            # phrase_output = phrase[:,:1,:]


            # sram = objectives.compute_cmp(L_img_token_output, phrase_output)
            # print("sram.shape1 ", sram.shape)
            # print("sram1 ", sram)

            # 均值
            # L_img_token_mean = L_img_token.mean(dim=1, keepdim=True)
            # phrase_mean = phrase.mean(dim=1, keepdim=True)
            # sram = objectives.compute_cmp(L_img_token_mean, phrase_mean)

            # sram = objectives.phrase_region_score(L_img_token, phrase)
            # print("sram.shape2 ", sram.shape)
            # print("sram ", sram)

            sim_1 = sim_global + sim_cd 
            # sim_1 = sim_global + sim_cd 
            # self.cr_loss_fun是triplet loss
            sim_global = self.cr_loss_fun(sim_1, labels, semi=False)
            # print("sim_global ", sim_global)
            # print("sim_global.shape ", sim_global.shape)
            ret.update({'cfine_loss':sim_global*0.8})

        # if 'id' in self.current_task:
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #     # 跨模态同粒度
        #     # sub_patch_features, patch_features, patches_residual= self.tnt_model(images.to(torch.float16))   # sub_patch_features [16384, 16, 24]   patch_features [64, 257, 512]
        #     # tnt_model = TNT().to(device)     
        #     # sub_patch_features, patch_features, patches_residual= tnt_model(images)   # sub_patch_features [16384, 16, 24]   patch_features [64, 257, 512]  patches_residual [64, 257, 512]
        #     ## ============================
        #     x = images
        #     b, _, h, w, patch_size, image_size = *x.shape, 16, 256
        #     assert divisible_by(h, patch_size) and divisible_by(w, patch_size), f'height {h} and width {w} of input must be divisible by the patch size'
        #     num_patches_h = h // patch_size
        #     num_patches_w = w // patch_size
        #     n = num_patches_w * num_patches_h
        #     pixels = self.to_pixel_tokens(x.to(torch.float16))
        #     # print('1',pixels.shape)  1 torch.Size([16384, 16, 24])
        #     patches = repeat(self.patch_tokens[:(n + 1)], 'n d -> b n d', b = b)    # 增强后的图片局部patch特征 patch_features [64, 257, 512]
        #     patches += rearrange(self.patch_pos_emb[:(n + 1)], 'n d -> () n d')
        #     pixels += rearrange(self.pixel_pos_emb, 'n d -> () n d')
        #     # print('2',pixels.shape)  2 torch.Size([16384, 16, 24])
        #     from torch.cuda.amp import autocast
        #     with autocast():
        #         for pixel_attn, pixel_ff, pixel_to_patch_residual, patch_attn, patch_ff in self.layers:

        #             pixels2 =  pixel_attn(pixels.to(torch.float16))
        #             pixels = pixels2 + pixels
        #             pixels = pixel_ff(pixels) + pixels

        #             patches_residual = pixel_to_patch_residual(pixels)  # 聚合后的像素特征

        #             patches_residual = rearrange(patches_residual, '(b h w) d -> b (h w) d', h = num_patches_h, w = num_patches_w)
        #             patches_residual = F.pad(patches_residual, (0, 0, 1, 0), value = 0) # cls token gets residual of 0
        #             patches = patches + patches_residual   #将视觉词的特征聚合成patch级别的特征,local

        #             patches = patch_attn(patches) + patches
        #             patches = patch_ff(patches) + patches  #计算图像块之间的注意力，global
        #     # cls_token = patches[:, 0]  #全局
        #     # ## ========================

        #     # W = phrase.to(torch.float32)      #[64, 257, 512]   W 短语级特征

        #     # # # 同模态跨粒度：图片全局-图片patch
        #     # img_g = G_img_token.to(torch.float16)           # [64, 1, 512]      图片全局特征
        #     # img_l = L_img_token.to(torch.float32)           # [64, 25, 512]     图片局部特征
        #     # frame_logit_weight = nn.parameter.Parameter(torch.eye(25), requires_grad=True).to(device).to(torch.float32)
        #     # img_l = patches_residual                 # 图像sub_patch特征 [64, 257, 512]
        #     # frame_logit_weight = nn.parameter.Parameter(torch.eye(257), requires_grad=True).to(device).to(torch.float16)
        #     # # # # 特征矩阵相乘方法1：torch.matmul(特征矩阵1, 特征矩阵2)
        #     # # sentence_frame_logits = logit_scale * torch.sum(torch.matmul(img_g, img_l.permute(0, 2, 1)) \
        #     # #     * torch.softmax(torch.matmul(torch.softmax(torch.matmul(img_g, img_l.permute(0, 2, 1)) / 1e-2, dim=-1), frame_logit_weight) / 1e-2, dim=-1), dim=-1).t()
        #     # # # print(sentence_frame_logits)
        #     # # # # 特征矩阵相乘方法2：F.softmax(特征矩阵1, dim=1) @ F.softmax(特征矩阵2, dim=1)
        #     # sentence_frame_logits = logit_scale * torch.sum((F.softmax(img_g, dim=1) @ F.softmax(img_l.permute(0, 2, 1), dim=1) / 0.07) \
        #         # * torch.softmax(torch.matmul(torch.softmax((F.softmax(img_g, dim=1) @ F.softmax(img_l.permute(0, 2, 1), dim=1)) / 1e-2, dim=-1), frame_logit_weight) / 1e-2, dim=-1), dim=-1).t()

        #     # # # 同模态跨粒度：文本全局-名词短语
        #     # word_g2 = G_text_token.to(torch.float32)        # [64, 1, 512]      句子特征
        #     # word_l = L_text_token.to(torch.float32)         # [64, 15, 512]     单词特征
        #     # frame_logit_weight = nn.parameter.Parameter(torch.eye(15), requires_grad=True).to(device).to(torch.float32)
        #     # # word_l = phrase.to(torch.float32)               # [64, 257, 512]   W 短语级特征   
        #     # # frame_logit_weight = nn.parameter.Parameter(torch.eye(257), requires_grad=True).to(device).to(torch.float32)
        #     # # # 特征矩阵相乘方法1：torch.matmul(特征矩阵1, 特征矩阵2)
        #     # # sentence_frame_logits = logit_scale * torch.sum(torch.matmul(word_g2, word_l.permute(0, 2, 1)) \
        #     #     # * torch.softmax(torch.matmul(torch.softmax(torch.matmul(word_g2, word_l.permute(0, 2, 1)) / 1e-2, dim=-1), frame_logit_weight) / 1e-2, dim=-1), dim=-1).t()
        #     # # # 特征矩阵相乘方法2：F.softmax(特征矩阵1, dim=1) @ F.softmax(特征矩阵2, dim=1)
        #     # sentence_frame_logits = logit_scale * torch.sum((F.softmax(word_g2, dim=1) @ F.softmax(word_l.permute(0, 2, 1), dim=1) / 0.07) \
        #     #     * torch.softmax(torch.matmul(torch.softmax((F.softmax(word_g2, dim=1) @ F.softmax(word_l.permute(0, 2, 1), dim=1)) / 1e-2, dim=-1), frame_logit_weight) / 1e-2, dim=-1), dim=-1).t()

        #     # # 跨模态跨粒度：图片全局-名词短语
        #     # img_g = G_img_token.to(torch.float32)           # [64, 1, 512]      图片全局特征
        #     # # word_l = phrase.to(torch.float32)               # [64, 257, 512]    短语级特征
        #     # # frame_logit_weight = nn.parameter.Parameter(torch.eye(257), requires_grad=True).to(device).to(torch.float32)
        #     # word_l = L_text_token.to(torch.float32)         # [64, 15, 512]     单词特征
        #     # frame_logit_weight = nn.parameter.Parameter(torch.eye(15), requires_grad=True).to(device).to(torch.float32)
        #     # # 特征矩阵相乘方法1：torch.matmul(特征矩阵1, 特征矩阵2)
        #     # # sentence_frame_logits = logit_scale * torch.sum(torch.matmul(img_g, word_l.permute(0, 2, 1)) \
        #     # #     * torch.softmax(torch.matmul(torch.softmax(torch.matmul(img_g, word_l.permute(0, 2, 1)) / 1e-2, dim=-1), frame_logit_weight) / 1e-2, dim=-1), dim=-1).t()
        #     # # 特征矩阵相乘方法2：F.softmax(特征矩阵1, dim=1) @ F.softmax(特征矩阵2, dim=1)
        #     # sentence_frame_logits = logit_scale * torch.sum((F.softmax(img_g, dim=1) @ F.softmax(word_l.permute(0, 2, 1), dim=1) / 0.07) \
        #     #     * torch.softmax(torch.matmul(torch.softmax((F.softmax(img_g, dim=1) @ F.softmax(word_l.permute(0, 2, 1), dim=1)) / 1e-2, dim=-1), frame_logit_weight) / 1e-2, dim=-1), dim=-1).t()



        #     # # 粗粒度-粗粒度
        #     # # 视频-句子
        #     # img_g = G_img_token.to(torch.float32).squeeze(1)          # [64, 1, 512] -> [64, 512] 图片全局特征（去掉第二维）
        #     # word_g1 = G_text_token.to(torch.float32).squeeze(1)       # [64, 1, 512] -> [64, 512] 句子特征（去掉第二维）
        #     # # 特征矩阵相乘方法1：torch.matmul(特征矩阵1, 特征矩阵2)
        #     # video_sentence_logits = logit_scale * torch.matmul(torch.matmul(word_g1, self.global_mat_weight), torch.matmul(img_g,self.global_mat_weight_1).t() )
        #     # # 特征矩阵相乘方法2：F.softmax(特征矩阵1, dim=1) @ F.softmax(特征矩阵2, dim=1) / 温度系数
        #     # # video_sentence_logits = F.softmax(torch.matmul(word_g1, self.global_mat_weight), dim=1) @ F.softmax(torch.matmul(img_g,self.global_mat_weight_1).t(), dim=1) / 0.07

        #     # # 跨粒度：
        #     # # 图片细粒度-文本粗粒度，帧-句子
        #     # img_l = L_img_token.to(torch.float32)           # [64, 25, 512]     图片局部特征
        #     # frame_logit_weight = nn.parameter.Parameter(torch.eye(25), requires_grad=True).to(device).to(torch.float32)
        #     # # img_l = patch_features                          # [64, 257, 512]    增强图片局部patch特征
        #     # # frame_logit_weight = nn.parameter.Parameter(torch.eye(257), requires_grad=True).to(device).to(torch.float32)
        #     img_l = patches_residual                 # 图像sub_patch特征 [64, 257, 512]
        #     frame_logit_weight = nn.parameter.Parameter(torch.eye(257), requires_grad=True).to(device).to(torch.float16)
        #     word_g2 = G_text_token.to(torch.float16)        # [64, 1, 512]      句子特征
        #     # # 特征矩阵相乘方法1：torch.matmul(特征矩阵1, 特征矩阵2)
        #     # sentence_frame_logits = logit_scale * torch.sum(torch.matmul(word_g2, img_l.permute(0, 2, 1)) \
        #         # * torch.softmax(torch.matmul(torch.softmax(torch.matmul(word_g2, img_l.permute(0, 2, 1)) / 1e-2, dim=-1), frame_logit_weight) / 1e-2, dim=-1), dim=-1).t()
        #     # # 特征矩阵相乘方法2：F.softmax(特征矩阵1, dim=1) @ F.softmax(特征矩阵2, dim=1)
        #     sentence_frame_logits = logit_scale * torch.sum((F.softmax(word_g2, dim=1) @ F.softmax(img_l.permute(0, 2, 1), dim=1) / 0.07) \
        #         * torch.softmax(torch.matmul(torch.softmax((F.softmax(word_g2, dim=1) @ F.softmax(img_l.permute(0, 2, 1), dim=1)) / 1e-2, dim=-1), frame_logit_weight) / 1e-2, dim=-1), dim=-1).t()


        #     # # 细粒度-细粒度
        #     # # 像素-短语/单词
        #     # # img_l = L_img_token.to(torch.float32)           # [64, 25, 512]     图片局部patch特征
        #     # # img_l = patch_features                          # [64, 257, 512]    增强图片局部patch特征
        #     # img_l = patches_residual.to(torch.float32)                        # [64, 257, 512]     聚合之后的sub_patch特征
        #     # word_l = L_text_token.to(torch.float32)         # [64, 15, 512]     单词特征
        #     # # word_l = phrase.to(torch.float32)               # [64, 257, 512]   W 短语级特征
        #     # bs_video, num_pixels, dim_video = img_l.shape   # bs_video: 64, num_pixels: 25, dim_video: 512
        #     # bs_text, num_words, dim_text = word_l.shape     # bs_text: 64, num_words: 15, dim_text: 512
        #     # embed_dim = 512 # 来自clip，通常是512或768
        #     # local_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True).to(device).to(torch.float32)
        #     # local_mat_weight1 = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True).to(device).to(torch.float32)
        #     # word_mat_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True).to(device).to(torch.float32)     # [257, 257]
        #     # pixel_mat_weight = nn.parameter.Parameter(torch.eye(num_pixels), requires_grad=True).to(device).to(torch.float32)   # [25, 25]
        #     # word_mat_weight2 = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True).to(device).to(torch.float32)  
        #     # pixel_mat_weight2 = nn.parameter.Parameter(torch.eye(num_pixels), requires_grad=True).to(device).to(torch.float32)
        #     # # 特征矩阵相乘方法1：torch.matmul(特征矩阵1, 特征矩阵2)
        #     # # fine_grained_sim_scores = torch.matmul(torch.matmul(word_l.reshape(-1, dim_text), local_mat_weight), torch.matmul(img_l.reshape(-1, dim_video),local_mat_weight1).t()).reshape(bs_text, num_words, bs_video, num_pixels)  # [64, 257, 64, 25] [bs_text, num_words, bs_video, num_frames]
        #     # # 特征矩阵相乘方法2：F.softmax(特征矩阵1, dim=1) @ F.softmax(特征矩阵2, dim=1)
        #     # fine_grained_sim_scores = (torch.matmul(F.softmax(torch.matmul(word_l.reshape(-1, dim_text), local_mat_weight), dim=1), F.softmax(torch.matmul(img_l.reshape(-1, dim_video),local_mat_weight1).t(), dim=1)) / 0.07).reshape(bs_text, num_words, bs_video, num_pixels)  # [64, 257, 64, 25] [bs_text, num_words, bs_video, num_frames]
        #     # word_level_logit = torch.sum(torch.softmax(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=1).permute(0,2,3,1), word_mat_weight)/1e-2, dim = -1).permute(0,3,1,2) * fine_grained_sim_scores, dim=1)               # [bs_text, bs_video, num_frames]
        #     # frame_level_logit = torch.sum(torch.softmax(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=-1), pixel_mat_weight)/1e-2, dim = -1) * fine_grained_sim_scores, dim=-1)       
        #     # sent2frame_logits = torch.sum(torch.softmax(torch.matmul(torch.softmax(word_level_logit/1e-2, dim=-1),pixel_mat_weight2)/1e-2, dim = -1) * word_level_logit, dim=-1)                                # [bs_text, bs_video]
        #     # video2word_logits = torch.sum(torch.softmax(torch.matmul(torch.softmax(frame_level_logit/1e-2, dim=1).permute(0,2,1), word_mat_weight2)/1e-2, dim = -1).permute(0,2,1) * frame_level_logit, dim=1)  # [bs_text, bs_video]
        #     # pixel_word_score = (sent2frame_logits + video2word_logits) / 2  # [64, 64]

        #     # logits = video_sentence_logits
        #     logits = sentence_frame_logits
        #     # logits = pixel_word_score
        #     # logits = (video_sentence_logits + pixel_word_score) / 2
        #     # logits = video_sentence_logits*0.25 + sentence_frame_logits*0.25 + pixel_word_score*0.5
        #     # logits = (video_sentence_logits + sentence_frame_logits + pixel_word_score) / 3
        #     # logits = video_sentence_logits + sentence_frame_logits + pixel_word_score
        #     # logits = (sentence_frame_logits + pixel_word_score) / 2

        #     # CrossEn()计算loss
        #     loss = 0.
        #     sim_loss1 = self.loss_fct(logits)
        #     sim_loss2 = self.loss_fct(logits.T)
        #     sim_loss = (sim_loss1 + sim_loss2) / 2
        #     loss += sim_loss

        #     # CRLoss()计算loss
        #     # loss = 0.
        #     # sim1 = self.cr_loss_fun(pixel_word_score, labels, semi=False)
        #     # sim2 = self.cr_loss_fun(pixel_word_score.T, labels, semi=False)
        #     # sim_loss = (sim1 + sim2) / 2
        #     # loss += sim_loss

        #     ret.update({'id_loss': loss*0.5})
        #     # ret.update({'id_loss': loss})

        # if 'id' in self.current_task:
        #     #跨模态同粒度
        #     R  = L_img_token
        #     W = phrase

        #     # print("R ", R)
        #     sim_i_2_t = objectives.get_similarity(patch_part=R, word_part=W)
        #     sim_t_2_i = sim_i_2_t.t()
        #     print("sim_i_2_t ", sim_i_2_t)
        #     # print("sim_i_2_t.shape ", sim_i_2_t.shape)
        #     # print("sim_t_2_i ", sim_t_2_i)
        #     # print("sim_t_2_i.shape ", sim_t_2_i.shape)

        #     sim_1 = self.cr_loss_fun(sim_i_2_t, labels, semi=False)
        #     sim_2 = self.cr_loss_fun(sim_t_2_i, labels, semi=False)
        #     # print("sim_1 ", sim_1)
        #     # print("sim_2 ", sim_2)
        #     # print("sim_1.shape ", sim_1.shape)
        #     # print("sim_2.shape ", sim_2.shape)

        #     sim_3 = (sim_1 + sim_2) / 2

        #     # print("sim_3.shape ", sim_3.shape)
        #     # print("sim_3 ", sim_3)
        #     # ret.update({'id_loss':sim_3*0.4})
        #     # ret.update({'id_loss':sim_3*0.5})
        #     ret.update({'id_loss':sim_3})

        #     # loss_i_2_t = objectives.align_loss(sim_i_2_t)     
        #     # loss_t_2_i = objectives.align_loss(sim_t_2_i)

        #     # align_loss = (loss_t_2_i + loss_i_2_t) / 2
        #     # print("align_loss ", align_loss)
        #     # print("align_loss.shape ", align_loss.shape)
        #     # ret.update({'id_loss':align_loss})

        # if 'xpool' in self.current_task:
        #     sentence_output = text_global
        #     video_output = img_global
        #     frame_features = L_img_token
        #     visual_pixel_output = L_img_token
        #     word_features = L_text_token_norm
        #     video_sentence_logits = logit_scale * torch.matmul(torch.matmul(sentence_output.to(torch.float32), self.global_mat_weight), torch.matmul(video_output.to(torch.float32),self.global_mat_weight_1).t() )
        #     # print('11',video_sentence_logits.shape)   torch.Size([64, 64])

        #     # sentence-frame score
        #     sentence_frame_logits = logit_scale * torch.sum(torch.matmul(sentence_output.to(torch.float32), frame_features.to(torch.float32).permute(0, 2, 1)) \
        #         * torch.softmax(torch.matmul(torch.softmax(torch.matmul(sentence_output.to(torch.float32), frame_features.to(torch.float32).permute(0, 2, 1)) / 1e-2, dim=-1), self.frame_logit_weight) / 1e-2, dim=-1), dim=-1).t()

        #     # frame-word score
        #     bs_video, num_pixels, dim_video = visual_pixel_output.shape
        #     bs_text, num_words, dim_text = word_features.shape

        #     fine_grained_sim_scores = torch.matmul(torch.matmul(word_features.to(torch.float32).view(-1, dim_text), self.local_mat_weight), torch.matmul(visual_pixel_output.to(torch.float32).view(-1, dim_video),self.local_mat_weight1).t()).view(bs_text, num_words, bs_video, num_pixels)  # [bs_text, num_words, bs_video, num_frames]

        #     word_level_logit = torch.sum(torch.softmax(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=1).permute(0,2,3,1), self.word_mat_weight)/1e-2, dim = -1).permute(0,3,1,2) * fine_grained_sim_scores, dim=1)  
        #     # print('11',word_level_logit.shape)   ([64, 64, 25])          # [bs_text, bs_video, num_frames]
        #     # frame_level_logit = torch.sum(torch.softmax(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=-1), self.pixel_mat_weight)/1e-2, dim = -1) * fine_grained_sim_scores, dim=-1)    
        #     # Step 1: 对 fine_grained_sim_scores 按照最后一个维度进行 softmax
        #     initial_softmax_similarity = torch.softmax(fine_grained_sim_scores / 1e-2, dim=-1)

        #     # Step 2: initial_softmax_similarity 与 pixel_mat_weight 进行矩阵乘法
        #     # print("initial_softmax_similarity shape:", initial_softmax_similarity.shape)
        #     # print("self.pixel_mat_weight shape:", self.pixel_mat_weight.shape)

        #     weighted_similarity = torch.matmul(initial_softmax_similarity, self.pixel_mat_weight)

        #     # Step 3: 对 weighted_similarity 进行 softmax
        #     final_softmax_similarity = torch.softmax(weighted_similarity / 1e-2, dim=-1)

        #     # Step 4: 计算最终的 frame_level_logit
        #     frame_level_logit = torch.sum(final_softmax_similarity * fine_grained_sim_scores, dim=-1)
   

        #     sent2frame_logits = torch.sum(torch.softmax(torch.matmul(torch.softmax(word_level_logit/1e-2, dim=-1),self.pixel_mat_weight2)/1e-2, dim = -1) * word_level_logit, dim=-1)                                # [bs_text, bs_video]
        #     video2word_logits = torch.sum(torch.softmax(torch.matmul(torch.softmax(frame_level_logit/1e-2, dim=1).permute(0,2,1), self.word_mat_weight2)/1e-2, dim = -1).permute(0,2,1) * frame_level_logit, dim=1)  # [bs_text, bs_video]

        #     pixel_word_score = (sent2frame_logits + video2word_logits) / 2
        #     # print('1',pixel_word_score.shape)

        #     logits = (video_sentence_logits + sentence_frame_logits + pixel_word_score) / 3
        #     # loss1 = self.cr_loss_fun(video_sentence_logits, labels, semi=False)
        #     # loss2 = self.cr_loss_fun(sentence_frame_logits, labels, semi=False)
        #     # loss3 = self.cr_loss_fun(pixel_word_score, labels, semi=False)
        #     # loss = loss2 + loss3
        #     loss = 0.
        #     sim_loss1 = self.loss_fct(logits)
        #     sim_loss2 = self.loss_fct(logits.T)
        #     sim_loss = (sim_loss1 + sim_loss2) / 2
        #     loss += sim_loss
        #     ret.update({'xpool_loss':loss*0.5})

        # if 'xpool' in self.current_task:
        #     sentence_output = text_global
        #     video_output = img_global
        #     frame_features = L_img_token_norm
        #     visual_pixel_output = L_img_token_norm#sub-patch
        #     word_features = L_text_token_norm
        #     video_sentence_logits = logit_scale * torch.matmul(torch.matmul(sentence_output.to(torch.float32), self.global_mat_weight), torch.matmul(video_output.to(torch.float32),self.global_mat_weight_1).t() )

        #     # sentence-frame score
        #     # sentence_output = sentence_output.unsqueeze(1)  # 变为 (64, 1, 512)
        #     # video_output = video_output.unsqueeze(1) 
        #     sentence_frame_logits = torch.matmul(sentence_output.to(torch.float32), frame_features.to(torch.float32).permute(0, 2, 1)) / 1e-2
        #     # print("sentence_output shape1:", sentence_output.shape)  sentence_output shape1: torch.Size([64, 512])
        #     # print("frame_features shape after permute1:", frame_features.permute(0, 2, 1).shape)   frame_features shape after permute1: torch.Size([64, 512, 50])

        #     # sentence_frame_logits = torch.softmax(sentence_frame_logits, dim=-1)
        #     # sentence_frame_logits = sentence_frame_logits[:, 0, :]  #[64, 50
        #     # sentence_frame_logits = torch.matmul(sentence_frame_logits, self.frame_logit_weight) / 1e-2
        #     sentence_frame_logits = torch.matmul(sentence_output.to(torch.float32), frame_features.to(torch.float32).permute(0, 2, 1)) / 1e-2

        #     # sentence_frame_logits = logit_scale * torch.sum(torch.matmul(sentence_output.to(torch.float32), frame_features.to(torch.float32).permute(0, 2, 1)) \
        #     #     * torch.softmax(torch.matmul(torch.softmax(torch.matmul(sentence_output.to(torch.float32), frame_features.to(torch.float32).permute(0, 2, 1)) / 1e-2, dim=-1), self.frame_logit_weight) / 1e-2, dim=-1), dim=-1).t()

        #     # frame-word score
        #     bs_video, num_pixels, dim_video = visual_pixel_output.shape
        #     bs_text, num_words, dim_text = word_features.shape

        #     fine_grained_sim_scores = torch.matmul(torch.matmul(word_features.to(torch.float32).view(-1, dim_text), self.local_mat_weight), torch.matmul(visual_pixel_output.to(torch.float32).view(-1, dim_video),self.local_mat_weight1).t()).view(bs_text, num_words, bs_video, num_pixels)  # [bs_text, num_words, bs_video, num_frames]
        #     #Bi-ISA
        #     word_level_logit = torch.sum(torch.softmax(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=1).permute(0,2,3,1), self.word_mat_weight)/1e-2, dim = -1).permute(0,3,1,2) * fine_grained_sim_scores, dim=1)               # [bs_text, bs_video, num_frames]
        #     frame_level_logit = torch.sum(torch.softmax(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=-1), self.pixel_mat_weight)/1e-2, dim = -1) * fine_grained_sim_scores, dim=-1)       

        #     sent2frame_logits = torch.sum(torch.softmax(torch.matmul(torch.softmax(word_level_logit/1e-2, dim=-1),self.pixel_mat_weight2)/1e-2, dim = -1) * word_level_logit, dim=-1)                                # [bs_text, bs_video]
        #     video2word_logits = torch.sum(torch.softmax(torch.matmul(torch.softmax(frame_level_logit/1e-2, dim=1).permute(0,2,1), self.word_mat_weight2)/1e-2, dim = -1).permute(0,2,1) * frame_level_logit, dim=1)  # [bs_text, bs_video]

        #     pixel_word_score = (sent2frame_logits + video2word_logits) / 2
        #     sentence_frame_logits = sentence_frame_logits[:, :64]
        #     sentence_frame_logits = sentence_frame_logits.mean(dim=2)

            
        #     # print("sentence_output shape:", sentence_output.shape)  # 应该是 [64, embed_dim]
        #     # print("frame_features shape:", frame_features.shape)  # 应该是 [64, num_frames, embed_dim]

        #     # print("video_sentence_logits shape:", video_sentence_logits.shape)
        #     # print("sentence_frame_logits shape:", sentence_frame_logits.shape)
        #     # print("pixel_word_score shape:", pixel_word_score.shape)



        #     logits = (video_sentence_logits + sentence_frame_logits + pixel_word_score) / 3
        #     # loss = 0.
        #     sim_loss1 = self.loss_fct(logits)
        #     sim_loss2 = self.loss_fct(logits.T)
        #     sim_loss = (sim_loss1 + sim_loss2) / 2
        #     # loss += sim_loss

        #     ret.update({'xpool_loss':sim_loss*0.5})
        
        # if 'mlm' in self.current_task:
        #     #同模态跨粒度
        #     #cfine里的cmpm cmpc loss 【缺】
        #     # sim_global = objectives.compute_cmp(img_f , text_f)
        #     # img_output1 = torch.stack(img_output, dim=0)
        #     # text_output1 = torch.stack(text_output, dim=0)
        #     # print('text_output123455:',text_output)
        #     sim_global1 = self.global_loss(img_output, text_output, img_f, text_f, labels)
        #     # sim_global2 = self.cr_loss_fun(sim_global1, labels, semi=False)
        #     ret.update({'mlm_loss':sim_global1*0.4})
 
        # if 'id' in self.current_task:
        #     #跨模态跨粒度对齐
        #     #sim-pt（单词-图像，句子和patch【有】）  跨粒度特征细化模块  phrase-图像，sub-patch-句子【缺】
        #     sim_23 = objectives.compute_cmm(phrase , G_img_token_norm)
        #     sim_23 = self.cr_loss_fun(sim_23, labels, semi=False)
        #     ret.update({'id_loss':sim_23})     
             
        # # TODO TNT
        # if 'LLM' in self.current_task:
        #     x=images
        #     # print('1',x.shape)  torch.Size([64, 3, 256, 256])
        #     # sim_lo = objectives.compute_cmpm_loss(i_l, L_text_token)  # [B]
        #     b, _, h, w, patch_size, image_size = *x.shape, 16, 256

        #     assert divisible_by(h, patch_size) and divisible_by(w, patch_size), f'height {h} and width {w} of input must be divisible by the patch size'

        #     num_patches_h = h // patch_size
        #     num_patches_w = w // patch_size
        #     n = num_patches_w * num_patches_h

        #     pixels = self.to_pixel_tokens(x.to(torch.float16))
        #     # print('1',pixels.shape)  1 torch.Size([16384, 16, 24])
        #     patches = repeat(self.patch_tokens[:(n + 1)], 'n d -> b n d', b = b)

        #     patches += rearrange(self.patch_pos_emb[:(n + 1)], 'n d -> () n d')
        #     pixels += rearrange(self.pixel_pos_emb, 'n d -> () n d')
        #     # print('2',pixels.shape)  2 torch.Size([16384, 16, 24])

        #     from torch.cuda.amp import autocast
        #     with autocast():
        #         for pixel_attn, pixel_ff, pixel_to_patch_residual, patch_attn, patch_ff in self.layers:
                    

        #             pixels2 =  pixel_attn(pixels.to(torch.float16))
        #             pixels = pixels2 + pixels
        #             pixels = pixel_ff(pixels) + pixels

        #             patches_residual = pixel_to_patch_residual(pixels)

        #             patches_residual = rearrange(patches_residual, '(b h w) d -> b (h w) d', h = num_patches_h, w = num_patches_w)
        #             patches_residual = F.pad(patches_residual, (0, 0, 1, 0), value = 0) # cls token gets residual of 0
        #             patches = patches + patches_residual   #将视觉词的特征聚合成patch级别的特征,local

        #             patches = patch_attn(patches) + patches
        #             patches = patch_ff(patches) + patches  #计算图像块之间的注意力，global

        #     # cls_token = patches[:, 0]  #全局
        #     # # patches = patches.device
        #     patches = torch.mean(patches, dim=1)
        #     patches = patches.to(torch.float16)
        #     # print('11111',patches.shape)   torch.Size([64, 512])


        #     L_text_token_norm = torch.mean(L_text_token_norm, dim=1)
        #     L_text_token_norm = L_text_token_norm.to(torch.float16)
        #     # print(patches)  #torch.Size([64, 512])
        #     # print(L_text_token_norm) # torch.Size([64, 512])
        #     # 图像局部和文本局部  跨模态同粒度
        #     sim1 = objectives.compute_cmpm_loss(patches, G_text_token.to(torch.float16).squeeze(1) , labels)
        #     # print(sim1.shape)  ]torch.Size([])

        #     sim_loss = self.cr_loss_fun(sim1, labels, semi=False)
        #     ret.update({'LLM_loss':sim_loss*0.5})

        # if 'mlm' in self.current_task:
        #     x=images
        #     # print('1',x.shape)
        #     # sim_lo = objectives.compute_cmpm_loss(i_l, L_text_token)  # [B]
        #     b, _, h, w, patch_size, image_size = *x.shape, 16, 256

        #     assert divisible_by(h, patch_size) and divisible_by(w, patch_size), f'height {h} and width {w} of input must be divisible by the patch size'

        #     num_patches_h = h // patch_size
        #     num_patches_w = w // patch_size
        #     n = num_patches_w * num_patches_h

        #     pixels = self.to_pixel_tokens(x.to(torch.float16))
        #     # print(pixels.shape)
        #     patches = repeat(self.patch_tokens[:(n + 1)], 'n d -> b n d', b = b)

        #     patches += rearrange(self.patch_pos_emb[:(n + 1)], 'n d -> () n d')
        #     pixels += rearrange(self.pixel_pos_emb, 'n d -> () n d')

        #     for pixel_attn, pixel_ff, pixel_to_patch_residual, patch_attn, patch_ff in self.layers:

        #         pixels = pixel_attn(pixels) + pixels
        #         pixels = pixel_ff(pixels) + pixels

        #         patches_residual = pixel_to_patch_residual(pixels)

        #         patches_residual = rearrange(patches_residual, '(b h w) d -> b (h w) d', h = num_patches_h, w = num_patches_w)
        #         patches_residual = F.pad(patches_residual, (0, 0, 1, 0), value = 0) # cls token gets residual of 0
        #         patches = patches + patches_residual   #将视觉词的特征聚合成patch级别的特征,local

        #         patches = patch_attn(patches) + patches
        #         patches = patch_ff(patches) + patches  #计算图像块之间的注意力，global

        #     patches = torch.mean(patches, dim=1)
        #     patches = patches.to(torch.float16)


        #     L_text_token_norm = torch.mean(L_text_token_norm, dim=1)
        #     L_text_token_norm = L_text_token_norm.to(torch.float16)
        #     # print(patches)  #torch.Size([64, 512])
        #     # print(L_text_token_norm) # torch.Size([64, 512])
        #     sim1 = objectives.compute_cmpm_loss(patches, L_text_token_norm, labels)
        #     # print(sim1.shape)  ]torch.Size([])

        #     sim_loss = self.cr_loss_fun(sim1, labels, semi=False)

        # TODO  AMFMN feature dunamic fusion
        # if 'id' in self.current_task:
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     tnt_model = TNT().to(device)
        #     # tnt_model = TNT()
     
        #     sub_patch_features, patch_features = tnt_model(images)
        #     # print('1',patch_features.shape)
        #     # print('1',patch_features.shape)
        #     patch_features = F.adaptive_avg_pool1d(patch_features.transpose(1, 2), 30).transpose(1, 2)

        #     # sim_pc= objectives.compute_cmpm_loss(patch_features,L_text_token_norm,labels)
        #     patch_features = patch_features.mean(dim=1)  # [64, 512]
        #     L_text_token_norm = L_text_token_norm.mean(dim=1)    # [64, 512]

        #     # 2. 归一化向量以便使用余弦相似度
        #     patch_features = F.normalize(patch_features, p=2, dim=-1).detach()
        #     L_text_token_norm = F.normalize(L_text_token_norm, p=2, dim=-1).detach()

        #     # 3. 计算批量之间的相似度，得到 (64, 64) 的矩阵
        #     sim_pc = torch.mm(patch_features.half(), L_text_token_norm.t()) 
        #     # sim_pc = F.cosine_similarity(patch_features.unsqueeze(2), L_text_token_norm.unsqueeze(1), dim=-1)
        #     # print('1',sim_pc.shape) torch.Size([64, 30, 30])
        #     # print('1',labels.shape)
        #     # sim_losspc = self.cr_loss_fun(sim_pc, labels, semi=False)
        #     sim_pc = torch.diag(sim_pc)
        #     sim_losspc = F.mse_loss(sim_pc, labels)
        #     ret.update({'id_loss':sim_losspc})

            # print("sub-patch特征形状:", sub_patch_features.shape)  torch.Size([16384, 16, 24])
            # print("patch特征形状:", patch_features.shape)  torch.Size([64, 257, 512])
            # Ft = self.cross_attention_s(L_img_token_norm, L_text_token_norm)
            # L_img_token_norm=L_img_token_norm.mean(dim=1)
            # out =  self.vgmf_gate (L_img_token_norm, L_text_token_norm)
            # sim_loss2 = self.cr_loss_fun(out, labels, semi=False)
            # ret.update({'id_loss':sim_loss2})

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    # convert_weights(model)
    return model
