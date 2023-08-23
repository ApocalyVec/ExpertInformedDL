import torch
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, in_channels: int = 3, patch_size: int = 16, embedding_size: int = 768,
                 mode="conv"):
        """
        patch embedding mode can be either conv or linear

        this implementation is inpired by
        https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
        https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
        :param in_channels:
        :param patch_size:
        :param embedding_size:
        :param mode:
        """
        self.patch_size = patch_size
        super().__init__()
        if mode == 'linear':
            self.patch_embedding = nn.Sequential(
                # break-down the image in s1 x s2 patches and flat them
                Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
                nn.Linear(patch_size * patch_size * in_channels, embedding_size)
            )
        elif mode == 'conv':
            self.patch_embedding = nn.Sequential(
                # using a conv layer instead of a linear one -> performance gains
                nn.Conv2d(in_channels, embedding_size, kernel_size=patch_size, stride=patch_size),
                Rearrange('b e (h) (w) -> b (h w) e'),
            )
        else:
            raise Exception("Unsupported patch embedding mode, please use conv or linear.")
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, embedding_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape

        x = self.patch_embedding(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)

        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 512, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # self.keys = nn.Linear(emb_size, emb_size)
        # self.queries = nn.Linear(emb_size, emb_size)
        # self.values = nn.Linear(emb_size, emb_size)
        self.qkv = nn.Linear(emb_size, emb_size * 3)

        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        # queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        # keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        # values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),  # the Gaussian Error Linear Units function
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, n_classes, emb_size: int = 768):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))


class ViT(nn.Sequential):
    def __init__(self,
                 n_classes: int,
                 img_size: int,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 depth: int = 12,
                 **kwargs):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, in_channels, patch_size, emb_size)
        self.transformer_encoder = TransformerEncoder(depth=depth, emb_size=emb_size, **kwargs)
        self.classification_head = ClassificationHead(n_classes, emb_size)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        return F.softmax(self.classification_head(x))
