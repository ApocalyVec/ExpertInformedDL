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
        self.ViT = ViT_LSTM(*args, **kwargs)
        self.depth = self.ViT.depth
        self.patch_height, self.patch_width = self.ViT.patch_height, self.ViT.patch_width

    def forward(self, img, collapse_attention_matrix=True, *args, **kwargs):
        '''

        Parameters
        ----------
        img
        collapse_attention_matrix
        args
        kwargs

        Returns
        -------

        '''

        # apply patch embedding to each subimage

        # flatten the mask for the attention layer
        subimage_xs = [self.ViT.to_patch_embedding(x) for x in img['subimages']]
        mask = [torch.flatten(m, 1, 2) for m in img['masks']]
        mask = torch.cat(mask, dim=1)
        # concatenate the subimage patches
        x = torch.cat(subimage_xs, dim=1)

        return self.ViT._encode(x, collapse_attention_matrix=collapse_attention_matrix, mask=mask, *args, **kwargs)


    def get_grid_size(self):
        return self.ViT.get_grid_size()

