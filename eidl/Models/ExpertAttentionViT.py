from typing import Iterable

import einops
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x,  *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None, *args, **kwargs):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            mask = F.pad(mask, (x.shape[-2] - mask.shape[-1], 0), value=True)  # class token is always valid
            square_mask = torch.einsum('bi,bj->bij', mask, mask)
            square_mask = einops.repeat(square_mask, 'b ... -> b h ...', h=self.heads)
            dots = dots.masked_fill(~square_mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, *args, **kwargs):
        for attn, ff in self.layers:
            out, alpha = attn(x, *args, **kwargs)
            x = out + x
            x = ff(x) + x
        return x, alpha  # last layer


# class ViT(nn.Module):
#     def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
#         super().__init__()
#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(patch_size)
#
#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
#
#         num_patches = (image_height // patch_height) * (image_width // patch_width)
#         patch_dim = channels * patch_height * patch_width
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#             nn.Linear(patch_dim, dim),
#         )
#
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)
#
#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
#
#         self.pool = pool
#         self.to_latent = nn.Identity()
#
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )
#
#     def forward(self, img):
#         x = self.to_patch_embedding(img)
#         b, n, _ = x.shape
#
#         cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embedding[:, :(n + 1)]
#         x = self.dropout(x)
#
#         x = self.transformer(x)
#
#         x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
#
#         x = self.to_latent(x)
#         return self.mlp_head(x)

class ViT_LSTM(nn.Module):
    def __init__(self, *, image_size, num_classes, embed_dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., weak_interaction=True, num_patches=None, patch_size=None):
        """

        Parameters
        ----------
        image_size: can either be a list of int, or a list of list of int, in the later case, it will be treated as the sizes for the subimages
        num_classes
        embed_dim
        depth
        heads
        mlp_dim
        pool
        channels
        dim_head
        dropout
        emb_dropout
        weak_interaction
        num_patches: int: number of patches in each dimension, using this parameter will override patch_size and will create the
        the same number of patches across height and width
        patch_size: tuple: tuple of two integers, in pixels (height, width)
        """
        super().__init__()
        self.depth = depth
        self.heads = heads

        self.num_dim_fixation = 2
        self.rnn_hidden_dim = 64
        self.num_layers = 2

        self.image_size = image_size
        # if type(image_size[0]) is list:


        assert num_patches is not None or patch_size is not None, 'Either num_patches or patch_size must be specified.'
        if num_patches is not None:
            print("computing number of patches from num_patches")
            assert not isinstance(image_size[0], Iterable), "subimages is not supported when num_patches is specified"
            image_width, image_height = image_size
            self.patch_height, self.patch_width = int(image_height / num_patches), int(image_width / num_patches)
            self.grid_size = num_patches, num_patches
            assert image_height % self.grid_size[1] == 0 and image_width % self.grid_size[0] == 0, 'Image dimensions must be divisible by the number of patches.'
            self.num_patches = (image_height // self.patch_height) * (image_width // self.patch_width)

        elif patch_size is not None:
            print("computing number of patches from patch_size and image size")
            self.patch_height, self.patch_width = patch_size
            self.grid_size = [(int(w / self.patch_height), int(h / self.patch_width)) for w, h in image_size]
            assert all([w % self.grid_size[i][0] == 0 and h % self.grid_size[i][1] == 0 for i, (w, h) in enumerate(image_size)]), 'Image dimensions must be divisible by the patch size.'
            self.num_patches = sum([w * h for (w, h) in self.grid_size])
        patch_dim = channels * self.patch_height * self.patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.Linear(patch_dim, embed_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(embed_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.weak_interaction = weak_interaction
        if self.weak_interaction:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(embed_dim + self.num_layers * self.rnn_hidden_dim),
                nn.Linear(embed_dim + self.num_layers * self.rnn_hidden_dim, num_classes)
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, num_classes)
            )

        self.lstm = nn.LSTM(self.num_dim_fixation, self.rnn_hidden_dim, num_layers=self.num_layers, bidirectional=False,
                            batch_first=True)

    def forward(self, img, collapse_attention_matrix=True, *args, **kwargs):
        x = self.to_patch_embedding(img)
        return self._encode(x, *args, **kwargs)


    def _encode(self, x, collapse_attention_matrix, fixation_sequence=None, *args, **kwargs):
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x, att_matrix = self.transformer(x, *args, **kwargs)
        if collapse_attention_matrix:
            att_matrix = att_matrix[:, :, 1:, 1:]
            att_matrix = att_matrix / torch.sum(att_matrix, dim=3, keepdim=True)
            att_matrix = torch.sum(att_matrix, dim=2)
        # att_matrix = att_matrix / torch.sum(att_matrix, dim=2,keepdim= True)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # rnn_input = rnn_utils.pad_sequence(seq, batch_first=True)
        # output = self.lstm(rnn_input)

        if self.weak_interaction:
            assert fixation_sequence is not None
            rnn_outputs, hiddens = self.lstm(fixation_sequence)
            hidden, cell = hiddens
            x = torch.concat([x] + [hidden[i, :, :] for i in range(self.num_layers)], dim=1)
        else:
            x = self.to_latent(x)

        return self.mlp_head(x), att_matrix

    # def test(self, img):
    #     x = self.to_patch_embedding(img)
    #     b, n, _ = x.shape
    #
    #     cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     x += self.pos_embedding[:, :(n + 1)]
    #     x = self.dropout(x)
    #
    #     x, att_matrix = self.transformer(x)
    #     att_matrix = att_matrix[:, :, 1:, 1:]
    #     att_matrix = att_matrix / torch.sum(att_matrix, dim=3, keepdim=True)
    #     att_matrix = torch.sum(att_matrix, dim=2)
    #     # att_matrix = att_matrix / torch.sum(att_matrix, dim=2,keepdim= True)
    #     # note test does not have lstm and sequence
    #     x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
    #
    #     return self.mlp_head(x), att_matrix

    def get_grid_size(self):
        return self.grid_size
