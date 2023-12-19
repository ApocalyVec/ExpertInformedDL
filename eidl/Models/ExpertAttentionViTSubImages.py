import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.utils.rnn as rnn_utils

from eidl.Models.ExpertAttentionViT import ViT_LSTM


class ViT_LSTM_subimage(ViT_LSTM):
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
        super().__init__(*args, **kwargs)

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
        subimage_xs = [self.to_patch_embedding(x) for x in img['subimages']]
        masks = [m for m in img['masks']]
        # concatenate the subimage patches

        x = torch.cat(subimage_xs, dim=1)
        x = self.to_patch_embedding(img)

        # TODO check if the patch masks are correct

        # TODO check the mask in the attention class

        # TODO change the AOI to match the subiamge patches

        return self._encode(x, *args, **kwargs)


