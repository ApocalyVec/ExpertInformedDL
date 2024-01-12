from typing import Iterable

import torch

import numpy as np

from eidl.Models.ExpertAttentionViT import Attention as ExpertAttentionViTAttention

from timm.models.vision_transformer import VisionTransformer

from eidl.Models.ExtensionTimmViTSubimage import ExtensionTimmViTSubimage
from eidl.utils.torch_utils import any_image_to_tensor


def rollout(depth, attentions, discard_ratio, head_fusion, normalize=True, return_raw_attention=False, *args, **kwargs):
    """

    Parameters
    ----------
    depth
    attentions
    discard_ratio
    head_fusion
    return_raw_attention: if false, the classification token's attention will be normalized and returned as a mask
                          if true, the raw attention will be returned
    normalize: if true, the attention will be normalized by dividing by the maximum attention
    args
    kwargs

    Returns
    -------

    """
    # check if depth is an interable
    if isinstance(depth, int):
        depth = [depth]
    elif isinstance(depth, Iterable) and all([np.issubdtype(type(d), np.integer) for d in depth]):
        pass
    else:
        raise ValueError("depth must be an integer or an iterable of integers")

    rolls = []

    result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    with torch.no_grad():
        for i, attention in enumerate(attentions):
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-2), attention_heads_fused.size(-1)).to(attention_heads_fused.device)
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
            if i in depth:
                if not return_raw_attention:
                    rtn = result[0, 0, 1:]  # Look at the total attention between the class token, # and the image patches
                else:
                    rtn = result[0]
                if normalize and rtn.max() > 0:
                    rtn = rtn / torch.max(rtn)

                rolls.append(rtn.detach().cpu().numpy())
                depth = [x for x in depth if x != i]
                if len(depth) == 0:
                    break

    if len(rolls) == 1:
        return rolls[0]

    return rolls

class VITAttentionRollout:
    def __init__(self, model, device, attention_layer_name='attn_drop', head_fusion="mean",
                 discard_ratio=0.9, vit_model=None, *args, **kwargs):
        """


        Parameters
        ----------
        model
        device
        attention_layer_name
        head_fusion
        discard_ratio: discard the bottom x% of the attention
        args
        kwargs
        """
        self.model = model
        self.vit_model = vit_model
        self.device = device
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

        self.attention_layer_count = 0
        for name, module in self.model.named_modules():
            if attention_layer_name in name or isinstance(module, ExpertAttentionViTAttention):
                module.register_forward_hook(self.get_attention)
                self.attention_layer_count += 1
        if self.attention_layer_count == 0:
            raise ValueError("No attention layer in the given model")
        if self.attention_layer_count != self.model.depth:
            raise ValueError(f"Model depth ({self.model.depth}) does not match attention layer count {self.attention_layer_count}")
        self.attentions = []

        self.is_reset_fuse_attn = isinstance(self.model, ExtensionTimmViTSubimage) or isinstance(self.vit_model, VisionTransformer)
        if self.is_reset_fuse_attn and isinstance(self.model, ExtensionTimmViTSubimage):
            self.block_count = len(self.model.vision_transformer.blocks)
            self.vit_model = self.model.vision_transformer
        elif self.is_reset_fuse_attn and isinstance(self.vit_model, VisionTransformer):
            self.block_count = len(self.vit_model.blocks)

    def get_attention(self, module, input, output):
        if isinstance(module, ExpertAttentionViTAttention):
            attention_output = output[1]
        else:
            attention_output = output
        self.attentions.append(attention_output)

    def __call__(self, depth, in_data, *args, **kwargs):
        if np.max(depth) > self.attention_layer_count:
            raise ValueError(f"Given depth ({depth}) is greater than the number of attenion layers in the model ({self.attention_layer_count})")
        self.attentions = []

        if self.is_reset_fuse_attn:
            # not use fuse attention so that the attn_drop is called
            for i in range(self.block_count):
                self.vit_model.blocks[i].attn.fused_attn = False

        output = self.model(in_data, *args, **kwargs)

        if self.is_reset_fuse_attn:
            # revert the fuse attention
            for i in range(self.block_count-1):
                self.vit_model.blocks[i].attn.fused_attn = True

        return rollout(depth, self.attentions, self.discard_ratio, self.head_fusion, *args, **kwargs)