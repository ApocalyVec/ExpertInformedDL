import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.utils.rnn as rnn_utils

from eidl.Models.ExpertAttentionViT import ViT_LSTM


class ViT_LSTM_subimage(nn.Module):
    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        image_size
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
        self.ViT_LSTM = ViT_LSTM(*args, **kwargs)

    def forward(self, img, *args, **kwargs):
        x = self.to_patch_embedding(img)
        return self.ViT_LSTM._encode(x, *args, **kwargs)


    def get_grid_size(self):
        return self.ViT_LSTM.grid_size
