""" CLIP Model
Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from collections import OrderedDict
import logging
import math
import os
from types import SimpleNamespace
from typing import List, Tuple, Union
import hashlib
import urllib
from tqdm import tqdm
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoProcessor, AutoModel, AutoTokenizer
# from transformers import SiglipImageProcessor, SiglipVisionModel



logger = logging.getLogger("IRRA.model")

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}

def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.positional_embedding = nn.Parameter(torch.randn((spacial_dim[0] * spacial_dim[1]) + 1, embed_dim)/ embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        spacial_dim = (
            input_resolution[0] // 32,
            input_resolution[1] // 32,
        )
        self.attnpool = AttentionPool2d(spacial_dim, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# class ResidualAttentionBlock(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
#         super().__init__()

#         self.attn = nn.MultiheadAttention(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2 = LayerNorm(d_model)
#         self.attn_mask = attn_mask

#     def attention(self, x: torch.Tensor):
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

#     def forward(self, x: torch.Tensor):
#         x = x + self.attention(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x

class ResidualAttentionBlock(nn.Module):
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
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)
        # return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x_, attention_score = self.attention(self.ln_1(x))
        x = x + x_
        x = x + self.mlp(self.ln_2(x))
        return x, attention_score


# class Transformer(nn.Module):
#     def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
#         super().__init__()
#         self.width = width
#         self.layers = layers
#         self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

#     def forward(self, x: torch.Tensor):
#         return self.resblocks(x)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        # return self.resblocks(x)
        attention_score = []
        for i in range(self.layers):
            x, score = self.resblocks[i](x)
            attention_score.append(score)
        return x, attention_score


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], patch_size: int, stride_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution # (384, 128)
        self.num_x = (input_resolution[1] - patch_size) // stride_size + 1
        self.num_y = (input_resolution[0] - patch_size) // stride_size + 1
        num_patches = self.num_x * self.num_y

        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)

        scale = width ** -0.5 # 1/sqrt(768)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


    def forward(self, x: torch.Tensor):
        # print('XXXShape',x.shape)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # print('XXXXXXX',x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, att_score = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj
    
        return x, att_score



class CLIP(nn.Module):
    def __init__(self,                      
                 embed_dim: int,
                 # vision
                 image_resolution: Union[int, Tuple[int, int]],
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 stride_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64  
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                stride_size=stride_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        # print('encode_image',self.visual(image.type(self.dtype)))
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x,att_score = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        x = x @ self.text_projection

        return x,att_score
    
    # def encode_text(self, text):
    #     # 1. Token embedding
    #     x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
    #     print(f"1. After token_embedding: {x.shape}")
    #     # 2. Add positional embedding
    #     pos_emb = self.positional_embedding.type(self.dtype)
    #     print(f"Positional embedding shape: {pos_emb.shape}")
    #     x = x + pos_emb
    #     print(f"2. After adding positional embedding: {x.shape}")
    #     # 3. Permute for transformer
    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     print(f"3. After permute (NLD->LND): {x.shape}")
    #     # 4. Transformer processing
    #     x, att_score = self.transformer(x)
    #     print(f"4. After transformer: {x.shape}")
    #     print(f"Attention scores shape: {att_score.shape if isinstance(att_score, torch.Tensor) else [t.shape for t in att_score]}")
    #     # 5. Permute back
    #     x = x.permute(1, 0, 2)  # LND -> NLD
    #     print(f"5. After permute (LND->NLD): {x.shape}")
    #     # 6. Layer normalization
    #     x = self.ln_final(x).type(self.dtype)
    #     print(f"6. After layer norm: {x.shape}")
    #     # 7. Text projection
    #     print(f"Text projection weight shape: {self.text_projection.shape}")
    #     x = x @ self.text_projection
    #     print(f"7. Final output shape: {x.shape}")
    #     return x, att_score

    def forward(self, image, text):
        # print('image111',image)
        image_features,image_att_scores = self.encode_image(image)
        # print('image111',image_features)
        text_features,text_att_scores = self.encode_text(text)

        # # normalized features
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        # # shape = [global_batch_size, global_batch_size]
        # return logits_per_image, logits_per_text

        return image_features,image_att_scores, text_features,text_att_scores
    
    
    def load_param(self, state_dict):
        # 将pretrained_dict里不属于model_dict的键剔除掉
        param_dict =  {k: v for k, v in state_dict.items() if k in self.state_dict()}

        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if k == 'visual.positional_embedding' and v.shape != self.visual.positional_embedding.shape:
                v = resize_pos_embed(v, self.visual.positional_embedding, self.visual.num_y, self.visual.num_x)
            elif k == 'positional_embedding' and v.shape != self.positional_embedding.shape:
                v = resize_text_pos_embed(v, self.context_length)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print(f'===========================ERROR occur in copy {k}, {v.shape}=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
    


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb = posemb.unsqueeze(0)
    posemb_new = posemb_new.unsqueeze(0)

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb.squeeze(0)


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj", "mcq_proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()
                    # if hasattr(attr, 'weight') and attr.weight is not None:
                    #     attr.weight = attr.weight.half()  # 将权重转换为 FP16
                    # if hasattr(attr, 'bias') and attr.bias is not None:
                    #     attr.bias = attr.bias.half()  # 如果存在 bias，将其转换为 FP16


    model.apply(_convert_weights_to_fp16)


def build_CLIP_from_openai_pretrained(name: str, image_size: Union[int, Tuple[int, int]], stride_size: int, jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    
    image_size: Union[int, Tuple[int, int]]
        Input image size, in Re-ID task, image size commonly set to 384x128, instead of 224x224

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")
    print("---------model_path--------")
    print(model_path)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu")
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    state_dict = state_dict or model.state_dict()

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model_cfg = {
        'embed_dim': embed_dim,
        'image_resolution': image_resolution,
        'vision_layers': vision_layers, 
        'vision_width': vision_width, 
        'vision_patch_size': vision_patch_size,
        'context_length': context_length, 
        'vocab_size': vocab_size, 
        'transformer_width': transformer_width, 
        'transformer_heads': transformer_heads, 
        'transformer_layers': transformer_layers
    }


    # modify image resolution to adapt Re-ID task
    model_cfg['image_resolution'] = image_size
    model_cfg['stride_size'] = stride_size
    logger.info(f"Load pretrained {name} CLIP model with model config: {model_cfg}")
    model = CLIP(**model_cfg)

    # covert model to fp16
    # convert_weights(model)

    # resize modified pos embedding
    model.load_param(state_dict)
    return model, model_cfg

# -----SIGLIP2
# class SIGLIP2(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  # vision
#                  image_resolution: Union[int, Tuple[int, int]],
#                  vision_layers: Union[Tuple[int, int, int, int], int],
#                  vision_width: int,
#                  vision_patch_size: int,
#                  stride_size: int,
#                  # text
#                  context_length: int,
#                  vocab_size: int,
#                  transformer_width: int,
#                  transformer_heads: int,
#                  transformer_layers: int
#                  ):
#         super().__init__()
#         self.context_length = context_length

#         ckpt = "google/siglip2-base-patch16-256"
#         # self.model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
#         self.model = AutoModel.from_pretrained(ckpt).eval()
#         self.model = self.model.to("cuda:0")  # 指定第一个 GPU
#         self.processor = AutoProcessor.from_pretrained(ckpt)


#         self.token_embedding = nn.Embedding(vocab_size, transformer_width)
#         self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
#         self.transformer = Transformer(
#             width=transformer_width,
#             layers=transformer_layers,
#             heads=transformer_heads,
#             attn_mask=self.build_attention_mask()
#         )
    
    
#     def encode_image(self, image):
#         inputs = self.processor(images=[image], return_tensors="pt").to(self.model.device)
#         print(3)
#         with torch.no_grad():
#             image_embeddings = self.model.get_image_features(**inputs)
#             print(4)
#         return image_embeddings

#     def encode_text(self, text):
#         x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

#         x = x + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x,att_score = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)

#         x = x @ self.text_projection
#         return x,att_score


#     def build_attention_mask(self):
#         # lazily create causal attention mask, with full attention between the vision tokens
#         # pytorch uses additive attention mask; fill with -inf
#         mask = torch.empty(self.context_length, self.context_length)
#         mask.fill_(float("-inf"))
#         mask.triu_(1)  # zero out the lower diagonal
#         return mask

#     def forward(self, image, text):
#         # print('image111',image)
#         image_features,image_att_scores = self.encode_image(image)
#         # print('image111',image_features)
#         # print('text123123213213:',text)
#         text_features,text_att_scores = self.encode_text(text)
#         # print('text_features12324:',text_features)

#         return image_features,image_att_scores, text_features,text_att_scores


def build_SIGLIP2_from_openai_pretrained(name: str, image_size: Union[int, Tuple[int, int]], stride_size: int, jit: bool = False, download_root: str = None):
# def build_SIGLIP2_from_openai_pretrained():
    # model = SIGLIP2()
    # model_cfg = {
    #     'embed_dim': embed_dim,
    #     'image_resolution': image_resolution,
    #     'vision_layers': vision_layers, 
    #     'vision_width': vision_width, 
    #     'vision_patch_size': vision_patch_size,
    #     'context_length': context_length, 
    #     'vocab_size': vocab_size, 
    #     'transformer_width': transformer_width, 
    #     'transformer_heads': transformer_heads, 
    #     'transformer_layers': transformer_layers
    # }
    # model = SIGLIP2(**model_cfg)

    # model = AutoModel.from_pretrained("google/siglip2-base-patch16-256")
    # processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-256")
    args = SimpleNamespace(
        mm_vision_select_layer=11,
        mm_vision_select_feature='patch',
        unfreeze_mm_vision_tower=False,
        num_workers=4
    )

    model = SigLIPVisionTextTower(vision_tower="google/siglip2-base-patch16-256", args=args)

    model_cfg = model.get_model_config()

    # model_cfg = {
    #     'embed_dim': embed_dim,
    #     'image_resolution': image_resolution,
    #     'vision_layers': vision_layers, 
    #     'vision_width': vision_width, 
    #     'vision_patch_size': vision_patch_size,
    #     'context_length': context_length, 
    #     'vocab_size': vocab_size, 
    #     'transformer_width': transformer_width, 
    #     'transformer_heads': transformer_heads, 
    #     'transformer_layers': transformer_layers
    # }


    # modify image resolution to adapt Re-ID task
    model_cfg['image_resolution'] = image_size
    model_cfg['stride_size'] = stride_size

    return model, model_cfg



class SigLIPVisionTextTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):   # vision_tower 代表加载的模型的名称
        super().__init__()
        device="cuda:0"
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.target_hidden_size = 512
        self.target_num_patches = 256  # 257 = 1 (CLS) + 256 patches
        self.device2 = device

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            vision_model = SiglipVisionModel.from_pretrained(self.vision_tower_name)
            self.cfg_only = vision_model.config 

        self.tokenizer = AutoTokenizer.from_pretrained(self.vision_tower_name)
        self.text_model = AutoModel.from_pretrained(self.vision_tower_name, output_attentions=True, torch_dtype=self.dtype).to(device)
        # 提取文本模型部分 (如果存在)
        if hasattr(self.text_model, 'text_model'):
            self.text_model = self.text_model.text_model
        self.text_model.eval()  # 设置为评估模式
        # 获取模型支持的最大序列长度
        self.max_length = self.text_model.config.max_position_embeddings
        
        # 投影层 (用于调整输出维度到512)
        self.projection = None
        hidden_size = self.text_model.config.hidden_size
        if hidden_size != 512:
            self.projection = nn.Linear(hidden_size, 512).to(device)

    def get_model_config(self):
        # 从视觉模型配置中获取参数
        vision_config = self.vision_tower.config
        # vision_config = model.config.vision_config
        text_model = self.text_model
        
        model_cfg = {
            # 视觉部分参数
            'vision_width': vision_config.hidden_size,  # 视觉特征维度 (如1152)
            'vision_layers': vision_config.num_hidden_layers,  # Transformer层数
            'vision_patch_size': vision_config.patch_size,  # Patch大小 (如14)
            'image_resolution': vision_config.image_size,  # 输入图像分辨率 (如336)
            
            # 文本部分参数
            'transformer_width': text_model.config.hidden_size,  # 文本特征维度
            'transformer_heads': text_model.config.num_attention_heads,  # 注意力头数
            'transformer_layers': text_model.config.num_hidden_layers,  # Transformer层数
            # 'context_length': text_model.max_length,  # 最大序列长度
            'context_length': getattr(text_model, 'max_length', 77),  # 如果没有 max_length 属性，使用默认值 77
            'vocab_size': text_model.config.vocab_size,  # 词表大小
            
            # 共享参数
            # 'embed_dim': text_model.target_hidden_size  # 投影后的统一维度(512)
            'embed_dim': 512  # 投影后的统一维度(512)
        }
        
        # 计算网格大小 (用于验证)
        # grid_size = vision_config.image_size // vision_config.patch_size
        # assert grid_size ** 2 == text_model.target_num_patches, "Patch数量不匹配"
        # 计算视觉模型的 patch 数量
        # grid_size = vision_config.image_size // vision_config.patch_size
        # num_patches = grid_size ** 2
        # assert num_patches == vision_config.num_patches, "Patch数量不匹配"

        
        return model_cfg

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, skipping.')
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(
            self.vision_tower_name,
            size={"height": 256, "width": 256}
        )
        self.vision_tower = SiglipVisionModel.from_pretrained(
            self.vision_tower_name,
            device_map=device_map,
            output_attentions=True,  # 确保模型返回注意力分数
            torch_dtype=torch.float16  # 或 torch.float16，根据你的需求
            # torch_dtype=torch.float32  # 或 torch.float16，根据你的需求
        )
        
        # 投影层
        original_hidden_size = self.vision_tower.config.hidden_size
        self.projection = nn.Linear(original_hidden_size, self.target_hidden_size).to(
            device=self.device2,
            dtype=self.vision_tower.dtype
        )
            
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # 获取特征
        print("image_forward_outs.hidden_states ", len(image_forward_outs.hidden_states))
        image_features = image_forward_outs.hidden_states[self.select_layer]
        
        if self.select_feature == 'patch':
            image_features = image_features[:, :257]  # [B, 257, orig_hidden_size]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        
        # 应用投影
        image_features = self.projection(image_features)  # [B, 257, 512]
        
        # 获取注意力分数
        attentions = image_forward_outs.attentions  # 元组，每层一个 [B, heads, seq_len, seq_len]
        # print("self.select_layer ", self.select_layer)
        # print("len(attentions) ", len(attentions))  # 12:0-11
        selected_attentions = attentions[self.select_layer]  # [B, heads, 257, 257]
        
        return image_features, selected_attentions
    
    def encode_image(self, image):
        if type(image) is list:
            image_features = []
            attentions = []
            for image in image:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device2, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                    output_attentions=True  # 确保返回注意力
                )
                img_feat, attn = self.feature_select(image_forward_out)
                image_features.append(img_feat.to(image.dtype))
                attentions.append(attn.to(image.dtype))
        else:
            print("image.dtype ", image.dtype)
            print("vision_tower.dtype ", self.vision_tower.dtype)
            print("self.dtype ", self.dtype)
            image_forward_outs = self.vision_tower(
                image.to(device=self.device2, dtype=self.dtype),
                output_hidden_states=True,
                output_attentions=True
            )
            image_features, attentions = self.feature_select(image_forward_outs)
            image_features = image_features.to(image.dtype)
            attentions = attentions.to(image.dtype)
        
        # 返回字典，包含特征和注意力
        return image_features,  attentions # [B, 257, 512]     # [B, heads, 257, 257]

    # def encode_text(self, text):
    #     print("text ",text)
    #     print("text.shape ",text.shape)
    #     print("Type of text:", type(text))
    #     # Tokenization
    #     print("self.max_length ", self.max_length)
    #     inputs = self.tokenizer(
    #         text, 
    #         padding="max_length", 
    #         max_length=self.max_length,
    #         truncation=True, 
    #         return_tensors="pt"
    #     ).to(self.device2)
    def encode_text(self, inputs):
        if isinstance(inputs, torch.Tensor):
            inputs = {
                "input_ids": inputs.to(self.device2).long(),
                "attention_mask": (inputs != self.tokenizer.pad_token_id).to(self.device2)
            }
    
        # 特征提取
        with torch.no_grad():
            outputs = self.text_model(**inputs, output_attentions=True)
        # 投影到512维 (如果需要)
        features = outputs.last_hidden_state
        if self.projection is not None:
            features = self.projection(features)
        return features, outputs.attentions
    

    @torch.no_grad()
    def forward(self, image, text):
        print("self.image.shape ", image.shape)
        print("self.text.shape ", text.shape)
        # 确保输入图像数据类型与模型一致
        image = image.to(device=self.device2, dtype=self.dtype)  # 关键修改
        text = text.to(device=self.device2, dtype=self.dtype)  # 关键修改
        # print('image111',image)
        image_features,image_att_scores = self.encode_image(image)
        # print('image111',image_features)
        # print('text123123213213:',text)
        text_features,text_att_scores = self.encode_text(text)
        # print('text_features12324:',text_features)
        return image_features,image_att_scores, text_features,text_att_scores

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.config.vision_config.hidden_size, device=self.device2, dtype=self.dtype) #    V   

    @property
    def dtype(self):
        return self.vision_tower.dtype # torch.float16  V

    @property
    def device(self):
        return self.vision_tower.device # device(type='cuda', index=0)  V

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config # CLIPVisionConfig - SiglipConfig   V
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.vision_config.hidden_size # 1152 V

    @property
    def num_patches_per_side(self):
        return self.config.vision_config.image_size // self.config.vision_config.patch_size # (348 // 14) = 27  V

    @property
    def num_patches(self):
        return (self.config.vision_config.image_size // self.config.vision_config.patch_size) ** 2



from transformers.image_utils import load_image

if __name__ == '__main__':
    print(11111111111)
    model = build_SIGLIP2_from_openai_pretrained()
    image = load_image("/home/hpc/LAB-data/disk-4T/syc_data/irra-v1/irra/IRRA-main_v2/000000000285.jpg")
    image_embeddings = model.encode_image(image=image)
    print(22222222222)
    print(image_embeddings.shape)

    # image_embeddings = model.get_image_features(**inputs)    